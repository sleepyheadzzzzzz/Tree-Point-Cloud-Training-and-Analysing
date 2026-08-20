#!/usr/bin/env python3
"""Render the six-panel environmental-change figure from prepared arrays."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--figure-data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def robust_symmetric_limit(data: np.ndarray, quantile: float = 0.99) -> float:
    finite = np.abs(data[np.isfinite(data)])
    if finite.size == 0:
        return 1.0
    return max(float(np.quantile(finite, quantile)), np.finfo(float).eps)


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    source = np.load(args.figure_data)
    panels = [
        (source["panel_a"], "A  Tree density", "Δ trees within 25 m", False),
        (source["panel_b"], "B  Monoculture rate", "Δ proportion", True),
        (source["panel_c"], "C  Sky-view factor", "Δ proportion", False),
        (source["panel_d"], "D  Solar radiation", "Δ source units", True),
        (source["panel_e"], "E  Land-surface temperature", "Δ °C", True),
        (source["panel_f"], "F  Daytime noise", "Δ dB", True),
    ]
    standard_palette = LinearSegmentedColormap.from_list(
        "negative_red_positive_green",
        ["#b2182b", "#ef8a62", "#fffdf5", "#7fbf7b", "#1b7837"],
        N=256,
    )
    reversed_palette = standard_palette.reversed(
        name="negative_green_positive_red"
    )

    figure, axes = plt.subplots(2, 3, figsize=(15.6, 10.2), facecolor="white")
    for axis, (data, title, label, reverse_colours) in zip(axes.flat, panels):
        limit = robust_symmetric_limit(data)
        image = axis.imshow(
            data,
            cmap=reversed_palette if reverse_colours else standard_palette,
            norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
            interpolation="nearest",
        )
        axis.set_title(title, fontsize=11.5, weight="bold", loc="left", pad=6)
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_visible(False)
        colorbar = figure.colorbar(
            image,
            ax=axis,
            orientation="horizontal",
            fraction=0.045,
            pad=0.025,
            extend="both",
        )
        colorbar.set_label(label, fontsize=9)
        colorbar.ax.tick_params(labelsize=8)

    figure.suptitle(
        "Mapped environmental change: 2015–2017 to 2021–2023",
        fontsize=15,
        weight="bold",
        y=0.992,
    )
    figure.text(
        0.5,
        0.008,
        (
            "Change = later minus earlier. In A and C, red = decrease and green = increase; "
            "in B, D, E and F, green = negative and red = positive. Pale colours indicate "
            "little or no change. Limits are panel-specific symmetric 99th percentiles."
        ),
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=[0, 0.035, 1, 0.965])

    stem = args.output / "Figure7_environment_change_with_noise_2015_2023"
    outputs = []
    for suffix, kwargs in (
        (".png", {"dpi": 600}),
        (".pdf", {}),
        (".svg", {}),
    ):
        path = stem.with_suffix(suffix)
        figure.savefig(path, bbox_inches="tight", facecolor="white", **kwargs)
        outputs.append(str(path))
    plt.close(figure)
    print("\n".join(outputs))


if __name__ == "__main__":
    main()
