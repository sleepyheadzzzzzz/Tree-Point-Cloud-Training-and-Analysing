#!/usr/bin/env python3
"""Validate spatial relative-growth CSV and GeoTIFF deliverables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio


PERIODS = ["15_17", "17_21", "21_23"]
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


def count_csv_rows(path: Path) -> int:
    newline_count = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            newline_count += block.count(b"\n")
    return max(0, newline_count - 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    output = args.output
    grid = json.loads((output / "grid_metadata.json").read_text(encoding="utf-8"))
    run = json.loads((output / "run_metadata.json").read_text(encoding="utf-8"))
    references = pd.read_csv(output / "reference_predictions.csv").set_index(
        "Species_Code"
    )
    expected_descriptions = tuple(
        f"Sp{code}_{name}" for code, name in SPECIES.items()
    )
    expected_rows = int(run["run"]["rows_processed"])
    csv_path = output / "spatial_relative_growth_predictions.csv"
    csv_rows = count_csv_rows(csv_path)
    if csv_rows != expected_rows:
        raise AssertionError(f"CSV rows {csv_rows:,} != expected {expected_rows:,}")

    sample = pd.read_csv(csv_path, nrows=1).iloc[0]
    summary_rows = []
    reliability_summary = []
    expected_transform = rasterio.transform.from_origin(
        float(grid["min_x_center"]) - float(grid["resolution"]) / 2.0,
        float(grid["max_y_center"]) + float(grid["resolution"]) / 2.0,
        float(grid["resolution"]),
        float(grid["resolution"]),
    )
    sample_column = int(
        round(
            (float(sample["X"]) - float(grid["min_x_center"]))
            / float(grid["resolution"])
        )
    )
    sample_row = int(
        round(
            (float(grid["max_y_center"]) - float(sample["Y"]))
            / float(grid["resolution"])
        )
    )

    for period in PERIODS:
        paths = {
            "percentage": output / f"relative_growth_pct_{period}.tif",
            "deviation": output / f"environmental_deviation_pp_{period}.tif",
            "level": output / f"suitability_level_{period}.tif",
            "reliability": output / f"reliability_{period}.tif",
        }
        for path in paths.values():
            if not path.exists() or path.stat().st_size == 0:
                raise AssertionError(f"Missing or empty raster: {path}")

        with (
            rasterio.open(paths["percentage"]) as percentage,
            rasterio.open(paths["deviation"]) as deviation,
            rasterio.open(paths["level"]) as level,
            rasterio.open(paths["reliability"]) as reliability,
        ):
            for dataset in [percentage, deviation, level, reliability]:
                if dataset.crs.to_string() != grid["crs"]:
                    raise AssertionError(f"Unexpected CRS in {dataset.name}")
                if dataset.shape != (int(grid["height"]), int(grid["width"])):
                    raise AssertionError(f"Unexpected shape in {dataset.name}")
                if not dataset.transform.almost_equals(expected_transform):
                    raise AssertionError(f"Unexpected transform in {dataset.name}")

            for dataset in [percentage, deviation, level]:
                if dataset.count != len(SPECIES):
                    raise AssertionError(f"Unexpected band count in {dataset.name}")
                if dataset.descriptions != expected_descriptions:
                    raise AssertionError(f"Unexpected band descriptions in {dataset.name}")
            if reliability.count != 3:
                raise AssertionError("Reliability raster must have three bands")

            reliable = reliability.read(1)
            populated = reliable != reliability.nodata
            populated_count = int(populated.sum())
            if populated_count != expected_rows:
                raise AssertionError(
                    f"{period}: populated cells {populated_count:,} "
                    f"!= rows {expected_rows:,}"
                )
            strict_count = int((reliable == 1).sum())
            robust = reliability.read(2)
            robust_count = int((robust == 1).sum())
            outside_count = reliability.read(3)
            reliability_summary.append(
                {
                    "Period": period,
                    "Populated_Cells": populated_count,
                    "Reliable_MinMax_Cells": strict_count,
                    "Reliable_MinMax_Fraction": strict_count / populated_count,
                    "Within_P01_P99_Cells": robust_count,
                    "Within_P01_P99_Fraction": robust_count / populated_count,
                    "Mean_Outside_MinMax_Feature_Count": float(
                        outside_count[populated].mean()
                    ),
                }
            )

            for band_index, species_code in enumerate(SPECIES, start=1):
                pct = percentage.read(band_index, masked=True)
                dev = deviation.read(band_index, masked=True)
                levels = level.read(band_index, masked=True)
                if int(pct.count()) != expected_rows:
                    raise AssertionError(
                        f"{period} species {species_code}: wrong valid count"
                    )
                sample_pct = float(
                    percentage.read(
                        band_index,
                        window=((sample_row, sample_row + 1), (sample_column, sample_column + 1)),
                    )[0, 0]
                )
                csv_value = float(sample[f"Pct_{period}_Sp{species_code}"])
                if not np.isclose(sample_pct, csv_value, atol=5.1e-4):
                    raise AssertionError(
                        f"CSV/raster mismatch: {period} species {species_code}"
                    )
                sample_dev = float(
                    deviation.read(
                        band_index,
                        window=((sample_row, sample_row + 1), (sample_column, sample_column + 1)),
                    )[0, 0]
                )
                expected_dev = sample_pct - float(
                    references.loc[
                        species_code, "Reference_Annual_Growth_Percent"
                    ]
                )
                if not np.isclose(sample_dev, expected_dev, atol=1.0e-4):
                    raise AssertionError(
                        f"Deviation mismatch: {period} species {species_code}"
                    )
                valid_levels = levels.compressed()
                if valid_levels.min() < 1 or valid_levels.max() > 7:
                    raise AssertionError(
                        f"Suitability levels outside 1-7: {period} species {species_code}"
                    )
                summary_rows.append(
                    {
                        "Period": period,
                        "Species_Code": species_code,
                        "Species_Name": SPECIES[species_code],
                        "Valid_Cells": int(pct.count()),
                        "Annual_Growth_Percent_Min": float(pct.min()),
                        "Annual_Growth_Percent_Mean": float(pct.mean()),
                        "Annual_Growth_Percent_Max": float(pct.max()),
                        "Environmental_Deviation_PP_Min": float(dev.min()),
                        "Environmental_Deviation_PP_Mean": float(dev.mean()),
                        "Environmental_Deviation_PP_Max": float(dev.max()),
                        "Suitability_Level_Min": int(valid_levels.min()),
                        "Suitability_Level_Max": int(valid_levels.max()),
                    }
                )

    summary = pd.DataFrame(summary_rows)
    reliability_table = pd.DataFrame(reliability_summary)
    summary.to_csv(output / "raster_value_summary.csv", index=False)
    reliability_table.to_csv(output / "reliability_summary.csv", index=False)
    report = {
        "status": "PASS",
        "csv_rows": csv_rows,
        "expected_rows": expected_rows,
        "crs": grid["crs"],
        "resolution": grid["resolution"],
        "shape": [grid["height"], grid["width"]],
        "species_band_count": len(SPECIES),
        "periods": PERIODS,
        "checks": [
            "CSV row count",
            "GeoTIFF file existence and nonzero size",
            "CRS, transform, shape, nodata, and band labels",
            "valid raster-cell count",
            "CSV-to-raster coordinate/value agreement",
            "environmental-deviation arithmetic",
            "suitability levels constrained to 1-7",
        ],
    }
    (output / "VALIDATION.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))
    print(reliability_table.to_string(index=False))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
