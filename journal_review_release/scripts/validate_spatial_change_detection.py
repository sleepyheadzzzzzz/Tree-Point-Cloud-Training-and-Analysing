#!/usr/bin/env python3
"""Validate spatial change-detection rasters and one source-grid sample."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window


SPECIES = {
    1: "General_Conifer",
    2: "General_Broadleaf",
    3: "Acer",
    4: "Alnus",
    5: "Betula",
    6: "Pinus",
    7: "Prunus",
    8: "Quercus",
    9: "Sorbus",
    10: "Tilia",
    11: "Ulmus",
}
ENVIRONMENT = {
    1: ("Density_15", "Density_21"),
    2: ("Mono_Rate_", "Mono_Rat_1"),
    3: ("svf15_17", "svf21_23"),
    4: ("RA15_17", "RA21_23"),
    5: ("LST15_17", "LST21_23"),
}


def pixel(dataset, band: int, row: int, column: int):
    return dataset.read(band, window=Window(column, row, 1, 1))[0, 0].item()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True, type=Path)
    parser.add_argument("--spatial-output", required=True, type=Path)
    args = parser.parse_args()
    output = args.spatial_output
    grid = json.loads((output / "grid_metadata.json").read_text(encoding="utf-8"))
    usecols = ["X", "Y"]
    for start, end in ENVIRONMENT.values():
        usecols.extend([start, end])
    sample = pd.read_csv(args.input_csv, usecols=usecols, nrows=1).iloc[0]
    column = round(
        (float(sample["X"]) - float(grid["min_x_center"]))
        / float(grid["resolution"])
    )
    row = round(
        (float(grid["max_y_center"]) - float(sample["Y"]))
        / float(grid["resolution"])
    )

    paths = {
        "environment": output / "environment_change_2015_2023.tif",
        "growth": output / "relative_growth_change_pp_2015_2023.tif",
        "level": output / "suitability_level_change_2015_2023.tif",
        "reliability": output / "reliability_intersection_2015_2023.tif",
        "early_growth": output / "relative_growth_pct_15_17.tif",
        "late_growth": output / "relative_growth_pct_21_23.tif",
        "early_level": output / "suitability_level_15_17.tif",
        "late_level": output / "suitability_level_21_23.tif",
        "early_reliability": output / "reliability_15_17.tif",
        "late_reliability": output / "reliability_21_23.tif",
    }
    for path in paths.values():
        if not path.exists() or path.stat().st_size == 0:
            raise AssertionError(f"Missing or empty output: {path}")

    datasets = {name: rasterio.open(path) for name, path in paths.items()}
    try:
        expected_shape = (int(grid["height"]), int(grid["width"]))
        expected_crs = grid["crs"]
        for name in ["environment", "growth", "level", "reliability"]:
            dataset = datasets[name]
            if dataset.shape != expected_shape:
                raise AssertionError(f"{name}: wrong shape {dataset.shape}")
            if dataset.crs.to_string() != expected_crs:
                raise AssertionError(f"{name}: wrong CRS {dataset.crs}")
        if datasets["environment"].count != 5:
            raise AssertionError("Environmental change must have five bands")
        if datasets["growth"].count != 11 or datasets["level"].count != 11:
            raise AssertionError("Growth and level change must have 11 bands")
        if datasets["reliability"].count != 1:
            raise AssertionError("Reliability intersection must have one band")

        for band, (start, end) in ENVIRONMENT.items():
            expected = float(sample[end]) - float(sample[start])
            actual = pixel(datasets["environment"], band, row, column)
            if not np.isclose(actual, expected, atol=1e-4):
                raise AssertionError(
                    f"Environmental band {band}: {actual} != {expected}"
                )

        for band, species in SPECIES.items():
            early = pixel(datasets["early_growth"], band, row, column)
            late = pixel(datasets["late_growth"], band, row, column)
            change = pixel(datasets["growth"], band, row, column)
            if not np.isclose(change, late - early, atol=1e-5):
                raise AssertionError(f"{species}: growth change mismatch")
            early_class = pixel(datasets["early_level"], band, row, column)
            late_class = pixel(datasets["late_level"], band, row, column)
            class_change = pixel(datasets["level"], band, row, column)
            if class_change != late_class - early_class:
                raise AssertionError(f"{species}: level change mismatch")

        early_reliable = pixel(datasets["early_reliability"], 1, row, column)
        late_reliable = pixel(datasets["late_reliability"], 1, row, column)
        intersection = pixel(datasets["reliability"], 1, row, column)
        expected_intersection = int(early_reliable == 1 and late_reliable == 1)
        if intersection != expected_intersection:
            raise AssertionError("Reliability intersection mismatch")
    finally:
        for dataset in datasets.values():
            dataset.close()

    summary = {
        "status": "PASS",
        "sample_grid_row": row,
        "sample_grid_column": column,
        "crs": grid["crs"],
        "shape": [int(grid["height"]), int(grid["width"])],
        "environment_bands": 5,
        "species_bands": 11,
        "checks": [
            "file existence and nonzero size",
            "CRS, shape, and band counts",
            "source CSV to environmental-difference agreement",
            "late-minus-early growth agreement for all species",
            "late-minus-early suitability agreement for all species",
            "two-period reliability intersection agreement",
        ],
    }
    (output / "change_detection_validation.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
