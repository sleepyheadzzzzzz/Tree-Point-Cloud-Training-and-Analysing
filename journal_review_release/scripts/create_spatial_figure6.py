#!/usr/bin/env python3
"""Create a manuscript Figure 6 from validated relative-growth rasters."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import BoundaryNorm
from rasterio.enums import Resampling


PERIODS = [
    ("15_17", "2015-2017"),
    ("17_21", "2017-2021"),
    ("21_23", "2021-2023"),
]


def read_display(path: Path, band: int, maximum_pixels: int = 850):
    with rasterio.open(path) as source:
        scale = max(source.height / maximum_pixels, source.width / maximum_pixels, 1)
        height = max(1, round(source.height / scale))
        width = max(1, round(source.width / scale))
        data = source.read(
            band,
            out_shape=(height, width),
            resampling=Resampling.nearest,
        ).astype(float)
        nodata = source.nodata
    if nodata is not None:
        data[data == nodata] = np.nan
    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--species-band", type=int, default=2)
    args = parser.parse_args()

    percentages = []
    levels = []
    reliabilities = []
    for period, _ in PERIODS:
        percentages.append(
            read_display(
                args.analysis / f"relative_growth_pct_{period}.tif",
                args.species_band,
            )
        )
        levels.append(
            read_display(
                args.analysis / f"suitability_level_{period}.tif",
                args.species_band,
            )
        )
        reliabilities.append(
            read_display(args.analysis / f"reliability_{period}.tif", 1)
        )

    pooled = np.concatenate(
        [
            values[(reliability == 1) & np.isfinite(values)]
            for values, reliability in zip(percentages, reliabilities)
        ]
    )
    vmin, vmax = np.quantile(pooled, [0.01, 0.99])
    level_cmap = plt.get_cmap("viridis", 7)
    level_norm = BoundaryNorm(np.arange(0.5, 8.5, 1), level_cmap.N)

    figure, axes = plt.subplots(2, 3, figsize=(15.5, 10.3))
    top_image = None
    bottom_image = None
    for column, ((_, label), percentage, level, reliability) in enumerate(
        zip(PERIODS, percentages, levels, reliabilities)
    ):
        percentage = percentage.copy()
        percentage[reliability != 1] = np.nan
        level = level.copy()
        level[(reliability != 1) | (level <= 0)] = np.nan
        top_image = axes[0, column].imshow(
            percentage,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )
        bottom_image = axes[1, column].imshow(
            level,
            cmap=level_cmap,
            norm=level_norm,
            interpolation="nearest",
        )
        axes[0, column].set_title(label, fontsize=12, weight="bold")
        for row in range(2):
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
            for spine in axes[row, column].spines.values():
                spine.set_visible(False)
    axes[0, 0].set_ylabel(
        "Predicted annual relative growth", fontsize=11, weight="bold"
    )
    axes[1, 0].set_ylabel("Fixed suitability level", fontsize=11, weight="bold")
    top_colorbar = figure.colorbar(
        top_image,
        ax=axes[0, :],
        orientation="horizontal",
        fraction=0.035,
        pad=0.02,
        aspect=40,
    )
    top_colorbar.set_label("Annual relative carbon growth (%)")
    bottom_colorbar = figure.colorbar(
        bottom_image,
        ax=axes[1, :],
        orientation="horizontal",
        fraction=0.035,
        pad=0.02,
        ticks=np.arange(1, 8),
        aspect=40,
    )
    bottom_colorbar.set_label("Suitability level (fixed training-derived scale)")
    figure.suptitle(
        "General broadleaf spatial diagnosis under a fixed 10 m park-tree scenario",
        fontsize=15,
        weight="bold",
        y=0.985,
    )
    figure.text(
        0.5,
        0.006,
        (
            "The same pooled no-period XGBoost model and thresholds are used in all "
            "periods. Grey/blank cells are outside the development min-max domain."
        ),
        ha="center",
        fontsize=9,
    )
    figure.subplots_adjust(
        left=0.05,
        right=0.985,
        top=0.94,
        bottom=0.08,
        hspace=0.22,
        wspace=0.08,
    )
    figure.savefig(args.output, dpi=300, bbox_inches="tight", facecolor="#EFEFEF")
    plt.close(figure)
    print(f"Created {args.output}")


if __name__ == "__main__":
    main()
