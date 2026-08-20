#!/usr/bin/env python3
"""Regenerate three standalone figures without content-obscuring overlays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap


COUNT_COLORS = [
    "#f2f2f2",
    "#edf8e9",
    "#c7e9c0",
    "#a1d99b",
    "#74c476",
    "#41ab5d",
    "#238b45",
    "#006d2c",
    "#005322",
    "#003717",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--confusion", type=Path, required=True)
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--map-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "axes.edgecolor": "#4a4a4a",
            "axes.linewidth": 0.8,
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "text.color": "#222222",
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def save_all(fig: plt.Figure, output: Path, stem: str) -> None:
    fig.savefig(output / f"{stem}.png", dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(output / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(output / f"{stem}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def calibration_figure(predictions: pd.DataFrame, diagnostics: pd.Series, output: Path) -> None:
    actual = predictions["Actual_LogSGR"].to_numpy(float)
    predicted = predictions["Predicted_LogSGR"].to_numpy(float)
    intercept = float(diagnostics["Calibration_Intercept"])
    slope = float(diagnostics["Calibration_Slope"])
    r2 = float(diagnostics["Test_R2_LogSGR"])

    fig, ax = plt.subplots(figsize=(7.2, 6.6), constrained_layout=True)
    hb = ax.hexbin(
        predicted,
        actual,
        gridsize=60,
        mincnt=1,
        cmap="viridis",
        bins="log",
        rasterized=True,
    )
    lower = float(min(actual.min(), predicted.min()))
    upper = float(max(actual.max(), predicted.max()))
    padding = 0.03 * (upper - lower)
    limits = (lower - padding, upper + padding)
    x_line = np.asarray(limits)
    calibration, = ax.plot(
        x_line,
        intercept + slope * x_line,
        color="#d95f02",
        linewidth=2.0,
        alpha=0.88,
        zorder=3,
        label="Calibration",
    )
    # Draw the reference as a dashed overlay so it remains distinguishable even
    # when the nearly ideal calibration line closely overlaps it.
    one_to_one, = ax.plot(
        x_line,
        x_line,
        color="#4a4a4a",
        linewidth=1.5,
        linestyle=(0, (5, 3)),
        zorder=4,
        label="1:1",
    )
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Predicted log-SGR")
    ax.set_ylabel("Observed log-SGR")
    ax.set_title("Locked-test calibration", loc="left", pad=34)
    ax.text(
        0.0,
        1.025,
        f"R² = {r2:.3f}    Calibration intercept = {intercept:.3f}    slope = {slope:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        clip_on=False,
    )
    ax.legend(
        handles=[one_to_one, calibration],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=2,
        frameon=False,
    )
    colorbar = fig.colorbar(hb, ax=ax, pad=0.035, fraction=0.055)
    colorbar.set_label("Observation density (log scale)")
    save_all(fig, output, "Figure_clean_locked_test_calibration")


def confusion_figure(confusion: pd.DataFrame, diagnostics: pd.Series, output: Path) -> None:
    matrix = confusion.to_numpy(dtype=float)
    row_totals = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, row_totals, out=np.zeros_like(matrix), where=row_totals > 0)
    exact = float(diagnostics["Suitability_Exact_Accuracy"])
    within = float(diagnostics["Suitability_Within_One"])
    kappa = float(diagnostics["Suitability_Quadratic_Kappa"])

    fig, ax = plt.subplots(figsize=(7.2, 6.6), constrained_layout=True)
    image = ax.imshow(
        normalized,
        cmap="Blues",
        vmin=0,
        vmax=max(0.01, float(normalized.max())),
        interpolation="nearest",
    )
    threshold = 0.50 * float(normalized.max())
    for row in range(7):
        for column in range(7):
            value = normalized[row, column]
            ax.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=9,
                color="white" if value > threshold else "#1e2a35",
            )
    ax.set_title("Fixed-threshold suitability agreement", loc="left", pad=34)
    ax.text(
        0.0,
        1.025,
        f"Exact = {exact:.1%}    Within ±1 level = {within:.1%}    Quadratic κ = {kappa:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        clip_on=False,
    )
    ax.set_xlabel("Predicted suitability level")
    ax.set_ylabel("Observed suitability level")
    ax.set_xticks(range(7), range(1, 8))
    ax.set_yticks(range(7), range(1, 8))
    ax.set_xticks(np.arange(-0.5, 7, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 7, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)
    colorbar = fig.colorbar(image, ax=ax, pad=0.035, fraction=0.055)
    colorbar.set_label("Row proportion")
    save_all(fig, output, "Figure_clean_suitability_agreement")


def map_figure(map_data: Path, output: Path) -> None:
    archive = np.load(map_data, allow_pickle=False)
    count = archive["suitable_genus_count"].astype(float)
    valid = archive["valid_any"].astype(bool)
    metadata = json.loads(str(archive["metadata_json"]))
    count[~valid] = np.nan
    resolution_x, resolution_y = metadata["resolution"]
    height, width = count.shape
    width_m = width * resolution_x
    height_m = height * resolution_y

    cmap = ListedColormap(COUNT_COLORS)
    cmap.set_bad("#d9d9d9")
    norm = BoundaryNorm(np.arange(-0.5, 10.5, 1.0), cmap.N)
    fig, ax = plt.subplots(figsize=(6.5, 9.0), constrained_layout=True)
    image = ax.imshow(
        count,
        cmap=cmap,
        norm=norm,
        origin="upper",
        extent=[0, width_m, 0, height_m],
        interpolation="nearest",
    )
    ax.set_title("Number of suitable genera", loc="left", pad=34)
    ax.text(
        0.0,
        1.025,
        "Suitability levels 5–7; retained patches >10 m²; observed maximum = 8",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.5,
        clip_on=False,
    )
    ax.set_xlabel("Easting from western edge (m)")
    ax.set_ylabel("Northing from southern edge (m)")
    ax.set_aspect("equal")
    colorbar = fig.colorbar(
        image,
        ax=ax,
        ticks=range(10),
        boundaries=np.arange(-0.5, 10.5, 1.0),
        pad=0.035,
        fraction=0.07,
    )
    colorbar.set_label("Suitable genera (0–9)")
    colorbar.ax.text(
        0.5,
        -0.045,
        "9 not observed",
        transform=colorbar.ax.transAxes,
        ha="center",
        va="top",
        fontsize=8,
    )
    save_all(fig, output, "Figure_clean_suitable_genus_count_map")


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    set_style()
    predictions = pd.read_csv(args.predictions)
    confusion = pd.read_csv(args.confusion, index_col=0)
    diagnostics = pd.read_csv(args.diagnostics).iloc[0]
    calibration_figure(predictions, diagnostics, args.output)
    confusion_figure(confusion, diagnostics, args.output)
    map_figure(args.map_data, args.output)
    print(f"Saved clean figures to {args.output.resolve()}")


if __name__ == "__main__":
    main()
