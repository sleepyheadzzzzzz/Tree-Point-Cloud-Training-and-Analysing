#!/usr/bin/env python3
"""Rasterize the daytime-noise values used for prediction, including 40 dB floor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True, type=Path)
    parser.add_argument("--spatial-output", required=True, type=Path)
    parser.add_argument("--chunk-size", type=int, default=250_000)
    args = parser.parse_args()

    grid = json.loads(
        (args.spatial_output / "grid_metadata.json").read_text(encoding="utf-8")
    )
    height = int(grid["height"])
    width = int(grid["width"])
    cell_count = height * width
    noise = np.full(cell_count, -9999.0, dtype=np.float32)
    replacement = np.full(cell_count, 255, dtype=np.uint8)
    replacement_count = 0
    min_x = float(grid["min_x_center"])
    max_y = float(grid["max_y_center"])
    resolution = float(grid["resolution"])

    for chunk in pd.read_csv(
        args.input_csv,
        usecols=["X", "Y", "noise"],
        chunksize=args.chunk_size,
    ):
        x = pd.to_numeric(chunk["X"], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(chunk["Y"], errors="coerce").to_numpy(dtype=float)
        source_noise = pd.to_numeric(
            chunk["noise"], errors="coerce"
        ).to_numpy(dtype=float)
        valid_xy = np.isfinite(x) & np.isfinite(y)
        row = np.rint((max_y - y) / resolution).astype(np.int64)
        column = np.rint((x - min_x) / resolution).astype(np.int64)
        valid_xy &= (
            (row >= 0)
            & (row < height)
            & (column >= 0)
            & (column < width)
        )
        replaced = ~np.isfinite(source_noise) | (source_noise <= -9990.0)
        used_noise = source_noise.copy()
        used_noise[replaced] = 40.0
        flat = row[valid_xy] * width + column[valid_xy]
        noise[flat] = used_noise[valid_xy].astype(np.float32)
        replacement[flat] = replaced[valid_xy].astype(np.uint8)
        replacement_count += int(np.sum(replaced & valid_xy))

    with rasterio.open(
        args.spatial_output / "relative_growth_pct_15_17.tif"
    ) as template:
        base_profile = template.profile
    noise_profile = {
        **base_profile,
        "count": 1,
        "dtype": "float32",
        "nodata": -9999.0,
        "compress": "DEFLATE",
        "predictor": 3,
        "zlevel": 6,
    }
    mask_profile = {
        **base_profile,
        "count": 1,
        "dtype": "uint8",
        "nodata": 255,
        "compress": "DEFLATE",
        "predictor": 2,
        "zlevel": 6,
    }
    with rasterio.open(
        args.spatial_output / "noise_day_used_db.tif", "w", **noise_profile
    ) as destination:
        destination.write(noise.reshape(height, width), 1)
        destination.set_band_description(1, "Daytime_noise_used_dB")
        destination.update_tags(
            1,
            quiet_floor_db=40,
            note="source missing/sentinel values replaced with 40 dB",
        )
    with rasterio.open(
        args.spatial_output / "noise_40db_replacement_mask.tif",
        "w",
        **mask_profile,
    ) as destination:
        destination.write(replacement.reshape(height, width), 1)
        destination.set_band_description(1, "Noise_replaced_with_40dB")
        destination.update_tags(
            1,
            value_1="source missing/sentinel replaced with 40 dB",
            value_0="original finite source noise retained",
        )

    summary = {
        "populated_cells": int(grid["input_rows"]),
        "noise_cells_replaced_with_40db": replacement_count,
        "replacement_fraction": replacement_count / int(grid["input_rows"]),
        "quiet_noise_floor_db": 40.0,
    }
    (args.spatial_output / "noise40_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
