#!/usr/bin/env python3
"""Create a clockwise radial P05-P95 SHAP-impact diagram.

The zero ring separates negative (inward) from positive (outward) model
contributions. Sector hue denotes the requested environmental category;
dark/light/grey denote positive, negative, or nonlinear feature-SHAP
association, respectively.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


VARIABLES = [
    ("View factor", "avg_svf", "Structure"),
    ("Diversity", "Mono_Rate", "Structure"),
    ("Density", "Density25", "Structure"),
    ("Park condition", "type_Puisto", "Structure"),
    ("Moraine", "soil_moraine", "Earth / soil"),
    ("Infill", "soil_infill", "Earth / soil"),
    ("Bedrock", "soil_bedrock", "Earth / soil"),
    ("LST", "avg_LST", "Atmosphere"),
    ("Radiation", "avg_radiation", "Atmosphere"),
    ("Noise", "avg_noise_day", "Atmosphere"),
    ("Illumination", "lightemiss", "Atmosphere"),
]

DEEP = {
    "Structure": "#1F7A4D",
    "Earth / soil": "#D84A36",
    "Atmosphere": "#187CAD",
}
LIGHT = {
    "Structure": "#9CCFAE",
    "Earth / soil": "#F3A187",
    "Atmosphere": "#9BD5E6",
}
NONLINEAR = "#B9BEC5"
BINARY = {"type_Puisto", "soil_moraine", "soil_infill", "soil_bedrock"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--statistics", required=True, type=Path)
    parser.add_argument("--deciles", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def monotonicity_score(deciles: pd.DataFrame, feature: str) -> float:
    values = deciles.loc[deciles["Feature"].eq(feature)].copy()
    if values.empty or len(values) <= 2:
        return 1.0
    x = values["Feature_Median"].to_numpy(dtype=float)
    if feature == "Mono_Rate":
        x = 1.0 - x
    values = values.assign(_x=x).sort_values("_x")
    differences = np.diff(values["Mean_SHAP"].to_numpy(dtype=float))
    total = float(np.abs(differences).sum())
    if total == 0:
        return 0.0
    upward = float(differences[differences > 0].sum())
    downward = float(-differences[differences < 0].sum())
    return max(upward, downward) / total


def build_plot_data(statistics: pd.DataFrame, deciles: pd.DataFrame) -> pd.DataFrame:
    overall = statistics.loc[statistics["Group"].eq("Overall")].set_index("Feature")
    records = []
    for position, (label, feature, category) in enumerate(VARIABLES, start=1):
        row = overall.loc[feature]
        rho = float(row["SHAP_Feature_Spearman"])
        contrast = float(row["High_minus_Low_SHAP"])
        if feature == "Mono_Rate":
            rho = -rho
            contrast = -contrast
        monotonicity = monotonicity_score(deciles, feature)
        if feature in BINARY:
            direction = "Positive association" if contrast >= 0 else "Negative association"
        elif abs(rho) < 0.25 or monotonicity < 0.65:
            direction = "Nonlinear association"
        else:
            direction = "Positive association" if rho > 0 else "Negative association"
        color = (
            NONLINEAR
            if direction == "Nonlinear association"
            else DEEP[category]
            if direction == "Positive association"
            else LIGHT[category]
        )
        records.append(
            {
                "Clockwise_Order": position,
                "Display_Variable": label,
                "Model_Feature": feature,
                "Category": category,
                "SHAP_P05": float(row["SHAP_P05"]),
                "SHAP_P95": float(row["SHAP_P95"]),
                "Robust_SHAP_Range_P05_P95": float(
                    row["SHAP_Robust_Range_P05_P95"]
                ),
                "Mean_Absolute_SHAP": float(row["Mean_Absolute_SHAP"]),
                "High_minus_Low_SHAP_for_Display_Variable": contrast,
                "Feature_SHAP_Spearman_for_Display_Variable": rho,
                "Decile_Monotonicity_Score": monotonicity,
                "Association_Class": direction,
                "Plot_Color": color,
            }
        )
    return pd.DataFrame(records)


def save_figure(data: pd.DataFrame, output_dir: Path) -> None:
    n = len(data)
    step = 2 * np.pi / n
    theta = np.arange(n) * step
    p05 = data["SHAP_P05"].to_numpy(float)
    p95 = data["SHAP_P95"].to_numpy(float)
    max_abs = float(max(np.abs(p05).max(), np.abs(p95).max()))
    zero_radius = 1.0
    impact_span = 0.68
    radial_scale = impact_span / max_abs
    bottoms = zero_radius + p05 * radial_scale
    heights = (p95 - p05) * radial_scale

    fig = plt.figure(figsize=(11.2, 11.2), facecolor="white")
    ax = fig.add_subplot(111, projection="polar")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_ylim(0.0, 2.17)
    ax.spines["polar"].set_visible(False)
    ax.set_facecolor("white")

    # Radial SHAP scale, including the zero ring.
    tick_values = np.arange(-0.15, 0.151, 0.05)
    tick_radii = zero_radius + tick_values * radial_scale
    valid = (tick_radii > 0) & (tick_radii < 1.72)
    ax.set_yticks(tick_radii[valid])
    ax.set_yticklabels(
        ["0" if abs(v) < 1e-12 else f"{v:+.2f}" for v in tick_values[valid]],
        fontsize=8.5,
        color="#5F6872",
    )
    ax.set_rlabel_position(205)
    ax.yaxis.grid(True, color="#D7DCE1", linewidth=0.7, linestyle=(0, (2, 3)))
    ax.xaxis.grid(True, color="#E2E6EA", linewidth=0.7, linestyle=(0, (2, 3)))
    ax.set_xticks(theta)
    ax.set_xticklabels([])
    ax.plot(np.linspace(0, 2 * np.pi, 720), np.full(720, zero_radius),
            color="#30363B", linewidth=1.7, zorder=4)

    ax.bar(
        theta,
        heights,
        width=step * 0.76,
        bottom=bottoms,
        color=data["Plot_Color"].tolist(),
        edgecolor="white",
        linewidth=1.2,
        alpha=0.98,
        zorder=3,
    )

    # Variable labels, kept horizontal for publication readability.
    for angle, label in zip(theta, data["Display_Variable"]):
        degrees = np.degrees(angle)
        if degrees < 1 or degrees > 359:
            ha = "center"
        elif 0 < degrees < 180:
            ha = "left"
        else:
            ha = "right"
        ax.text(angle, 1.79, label, ha=ha, va="center", fontsize=11,
                color="#30363B", fontweight="medium")

    # Requested category arcs in the outer ring.
    group_specs = [
        ("Structure", 0, 4),
        ("Earth / soil", 4, 3),
        ("Atmosphere", 7, 4),
    ]
    for group, start, count in group_specs:
        center = (start + (count - 1) / 2) * step
        ax.bar(
            center,
            0.055,
            width=count * step - 0.035,
            bottom=1.93,
            color=DEEP[group],
            edgecolor="none",
            zorder=2,
        )
        ax.text(center, 2.08, group, ha="center", va="center",
                fontsize=12.5, color=DEEP[group], fontweight="bold")

    fig.suptitle(
        "Categorized environmental SHAP impact",
        y=0.965,
        fontsize=19,
        fontweight="bold",
        color="#202428",
    )
    fig.text(
        0.5,
        0.918,
        "Three-soil pooled XGBoost · independent-test P05–P95 SHAP range",
        ha="center",
        fontsize=11.5,
        color="#59636D",
    )
    fig.text(
        0.5,
        0.073,
        "Inward from zero = negative contribution    ·    Outward from zero = positive contribution",
        ha="center",
        fontsize=10.5,
        color="#48515A",
    )

    legend_handles = [
        Patch(facecolor="#34434D", label="Dark hue: positive association"),
        Patch(facecolor="#B7D9E1", label="Light hue: negative association"),
        Patch(facecolor=NONLINEAR, label="Grey: nonlinear association"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.013),
        ncol=3,
        frameon=False,
        fontsize=9.5,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "Figure_radial_environmental_SHAP_three_soil"
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    statistics = pd.read_csv(args.statistics)
    deciles = pd.read_csv(args.deciles)
    deciles = deciles.loc[deciles["Group"].eq("Overall")]
    data = build_plot_data(statistics, deciles)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.output_dir / "radial_SHAP_figure_data.csv", index=False)
    save_figure(data, args.output_dir)
    print(data.to_string(index=False))


if __name__ == "__main__":
    main()
