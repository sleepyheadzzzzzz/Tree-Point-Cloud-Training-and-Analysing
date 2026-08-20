#!/usr/bin/env python3
"""Create a species-specific categorized environmental SHAP composite.

The figure converts the pooled XGBoost independent-test SHAP subsets for nine
genera into three environmental domains. Feature magnitude is the robust
P05-P95 SHAP range. Association classes use the same rules as the pooled
radial diagram in ``create_radial_shap_diagram.py``.

The output is descriptive of the fitted model and is not a causal effect
estimate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Polygon, Rectangle
import numpy as np
import pandas as pd


GENERA = ["Acer", "Alnus", "Betula", "Pinus", "Prunus", "Quercus", "Sorbus", "Tilia", "Ulmus"]

# Clockwise order and display semantics match the pooled radial figure.
VARIABLES = [
    ("View factor", "avg_svf", "Structure"),
    ("Diversity", "Mono_Rate", "Structure"),
    ("Density", "Density25", "Structure"),
    ("Park condition", "type_Puisto", "Structure"),
    ("Moraine", "soil_moraine", "Soil"),
    ("Infill", "soil_infill", "Soil"),
    ("Bedrock", "soil_bedrock", "Soil"),
    ("LST", "avg_LST", "Atmosphere"),
    ("Radiation", "avg_radiation", "Atmosphere"),
    ("Noise", "avg_noise_day", "Atmosphere"),
    ("Illumination", "lightemiss", "Atmosphere"),
]

CATEGORY_ORDER = ["Atmosphere", "Structure", "Soil"]
DEEP = {"Atmosphere": "#187CAD", "Structure": "#1F7A4D", "Soil": "#D84A36"}
LIGHT = {"Atmosphere": "#9BD5E6", "Structure": "#9CCFAE", "Soil": "#F3A187"}
NONLINEAR = "#B9BEC5"
BINARY = {"type_Puisto", "soil_moraine", "soil_infill", "soil_bedrock"}
NONLINEAR_RHO_THRESHOLD = 0.25
NONLINEAR_MONOTONICITY_THRESHOLD = 0.65
BORDERLINE_MONOTONICITY_TOLERANCE = 0.01
STRONG_DIRECTIONAL_RHO = 0.50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--statistics", required=True, type=Path)
    parser.add_argument("--deciles", required=True, type=Path)
    parser.add_argument("--metrics", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--rotation-degrees-clockwise", type=float, default=0.0)
    parser.add_argument("--contiguous-wedges", action="store_true")
    parser.add_argument("--hide-category-background", action="store_true")
    parser.add_argument(
        "--figure-stem",
        default="Figure5_species_categorized_SHAP_profiles_zero_centered",
    )
    return parser.parse_args()


def monotonicity_score(deciles: pd.DataFrame, group: str, feature: str) -> float:
    values = deciles.loc[
        deciles["Group"].eq(group) & deciles["Feature"].eq(feature)
    ].copy()
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


def classify_continuous_association(
    rho: float, contrast: float, monotonicity: float
) -> tuple[str, str, str]:
    """Classify and audit a continuous dependence pattern.

    The original classification is retained for audit. A narrow safeguard
    prevents a strongly directional feature from being labelled nonlinear
    solely because its decile-monotonicity score is within 0.01 of the 0.65
    threshold and its Spearman and end-contrast signs agree.
    """
    sign_label = "Positive association" if rho > 0 else "Negative association"
    previous = (
        "Nonlinear association"
        if abs(rho) < NONLINEAR_RHO_THRESHOLD
        or monotonicity < NONLINEAR_MONOTONICITY_THRESHOLD
        else sign_label
    )
    if abs(rho) < NONLINEAR_RHO_THRESHOLD:
        return (
            "Nonlinear association",
            previous,
            f"Weak overall direction: |Spearman| < {NONLINEAR_RHO_THRESHOLD:.2f}",
        )
    if monotonicity < NONLINEAR_MONOTONICITY_THRESHOLD:
        borderline_floor = (
            NONLINEAR_MONOTONICITY_THRESHOLD
            - BORDERLINE_MONOTONICITY_TOLERANCE
        )
        signs_agree = contrast != 0 and np.sign(rho) == np.sign(contrast)
        if (
            monotonicity >= borderline_floor
            and abs(rho) >= STRONG_DIRECTIONAL_RHO
            and signs_agree
        ):
            return (
                sign_label,
                previous,
                "Borderline monotonicity retained as directional because "
                "|Spearman| >= 0.50 and Spearman/end-contrast signs agree",
            )
        return (
            "Nonlinear association",
            previous,
            f"Material decile reversal: monotonicity < {NONLINEAR_MONOTONICITY_THRESHOLD:.2f}",
        )
    return sign_label, previous, "Directional Spearman and decile pattern"


def build_feature_data(statistics: pd.DataFrame, deciles: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    indexed = statistics.set_index(["Group", "Feature"])
    for group in GENERA:
        for position, (label, feature, category) in enumerate(VARIABLES, start=1):
            row = indexed.loc[(group, feature)]
            rho = float(row["SHAP_Feature_Spearman"])
            contrast = float(row["High_minus_Low_SHAP"])
            if feature == "Mono_Rate":
                rho = -rho
                contrast = -contrast
            monotonicity = monotonicity_score(deciles, group, feature)
            if feature in BINARY:
                direction = "Positive association" if contrast >= 0 else "Negative association"
                previous_direction = direction
                classification_reason = "Binary indicator: sign of high-minus-low SHAP contrast"
            else:
                direction, previous_direction, classification_reason = (
                    classify_continuous_association(rho, contrast, monotonicity)
                )
            color = (
                NONLINEAR
                if direction == "Nonlinear association"
                else DEEP[category]
                if direction == "Positive association"
                else LIGHT[category]
            )
            p05 = float(row["SHAP_P05"])
            p95 = float(row["SHAP_P95"])
            records.append(
                {
                    "Genus": group,
                    "N_Test_Rows": int(row["N_SHAP"]),
                    "Clockwise_Order": position,
                    "Display_Variable": label,
                    "Model_Feature": feature,
                    "Category": category,
                    "SHAP_P05": p05,
                    "SHAP_P95": p95,
                    "Absolute_P05_Plus_P95": abs(p05) + abs(p95),
                    "Robust_SHAP_Range_P05_P95": float(row["SHAP_Robust_Range_P05_P95"]),
                    "Mean_Absolute_SHAP": float(row["Mean_Absolute_SHAP"]),
                    "High_minus_Low_SHAP_for_Display_Variable": contrast,
                    "Feature_SHAP_Spearman_for_Display_Variable": rho,
                    "Decile_Monotonicity_Score": monotonicity,
                    "Previous_Association_Class": previous_direction,
                    "Association_Class": direction,
                    "Classification_Changed_After_Audit": direction != previous_direction,
                    "Classification_Reason": classification_reason,
                    "Plot_Color": color,
                }
            )
    data = pd.DataFrame(records)
    totals = data.groupby("Genus")["Absolute_P05_Plus_P95"].transform("sum")
    data["Feature_Share_of_Genus_Absolute_SHAP_Impact"] = (
        data["Absolute_P05_Plus_P95"] / totals
    )
    return data


def build_category_data(feature_data: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        feature_data.groupby(["Genus", "Category"], as_index=False)
        .agg(
            Category_Absolute_SHAP_Impact=("Absolute_P05_Plus_P95", "sum"),
            Category_Robust_SHAP_Range_Sum=("Robust_SHAP_Range_P05_P95", "sum"),
            Category_Mean_Absolute_SHAP_Sum=("Mean_Absolute_SHAP", "sum"),
            N_Features=("Model_Feature", "size"),
            N_Test_Rows=("N_Test_Rows", "first"),
        )
    )
    totals = grouped.groupby("Genus")["Category_Absolute_SHAP_Impact"].transform("sum")
    grouped["Category_Composition_Share"] = grouped["Category_Absolute_SHAP_Impact"] / totals

    for association, label in [
        ("Positive association", "Positive_Sensitivity"),
        ("Negative association", "Negative_Sensitivity"),
        ("Nonlinear association", "Nonlinear_Sensitivity"),
    ]:
        values = (
            feature_data.loc[feature_data["Association_Class"].eq(association)]
            .groupby(["Genus", "Category"])["Absolute_P05_Plus_P95"]
            .sum()
        )
        grouped[label] = [float(values.get((g, c), 0.0)) for g, c in zip(grouped["Genus"], grouped["Category"])]

    grouped["Category_Rank_within_Genus"] = (
        grouped.groupby("Genus")["Category_Absolute_SHAP_Impact"]
        .rank(method="dense", ascending=False)
        .astype(int)
    )
    grouped["Category"] = pd.Categorical(
        grouped["Category"], categories=CATEGORY_ORDER, ordered=True
    )
    return grouped.sort_values(["Genus", "Category"]).reset_index(drop=True)


def load_metrics(path: Path) -> pd.DataFrame:
    metrics = pd.read_csv(path)
    keep = metrics.loc[
        metrics["Split"].eq("Test")
        & metrics["Model"].eq("XGB")
        & metrics["Group"].isin(GENERA),
        ["Group", "N_Rows", "N_Trees", "R2_LogSGR", "RMSE_LogSGR", "MAE_LogSGR"],
    ].copy()
    return keep.rename(columns={"Group": "Genus", "R2_LogSGR": "Pooled_XGB_Within_Genus_Test_R2"})


def draw_score_ring(
    ax: plt.Axes,
    genus_data: pd.DataFrame,
    category_data: pd.DataFrame,
    global_abs_limit: float,
    rotation_degrees_clockwise: float,
    contiguous_wedges: bool,
    show_category_background: bool,
) -> None:
    genus_data = genus_data.sort_values("Clockwise_Order")
    n = len(genus_data)
    step = 2 * np.pi / n
    rotation = np.deg2rad(rotation_degrees_clockwise)
    theta = np.arange(n) * step + rotation
    p05 = genus_data["SHAP_P05"].to_numpy(float)
    p95 = genus_data["SHAP_P95"].to_numpy(float)
    zero_radius = 0.82
    impact_span = 0.38
    radial_scale = impact_span / global_abs_limit
    bottoms = zero_radius + p05 * radial_scale
    heights = (p95 - p05) * radial_scale

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 1.54 if contiguous_wedges else 1.57)
    ax.set_axis_off()

    background_width = step if contiguous_wedges else step * 0.82
    wedge_width = step if contiguous_wedges else step * 0.78
    wedge_edgecolor = "none" if contiguous_wedges else "white"
    wedge_linewidth = 0.0 if contiguous_wedges else 0.55
    if show_category_background:
        category_background = [DEEP[c] for c in genus_data["Category"]]
        ax.bar(theta, np.full(n, impact_span * 2), width=background_width,
               bottom=zero_radius - impact_span,
               color=category_background, alpha=0.08, edgecolor="none", zorder=1)
    ax.bar(theta, heights, width=wedge_width, bottom=bottoms,
           color=genus_data["Plot_Color"].tolist(), edgecolor=wedge_edgecolor,
           linewidth=wedge_linewidth, zorder=3)
    circle_theta = np.linspace(0, 2 * np.pi, 360)
    ax.plot(circle_theta, np.full(360, zero_radius), color="#202428", linewidth=1.35, zorder=5)

    group_specs = [("Structure", 0, 4), ("Soil", 4, 3), ("Atmosphere", 7, 4)]
    scores = category_data.set_index("Category")["Category_Absolute_SHAP_Impact"]
    for category, start, count in group_specs:
        center = (start + (count - 1) / 2) * step + rotation
        group_width = count * step if contiguous_wedges else count * step - 0.035
        ax.bar(center, 0.036, width=group_width, bottom=1.30,
               color=DEEP[category], edgecolor="none", zorder=2)
        label_radius = 1.43 if contiguous_wedges else 1.49
        ax.text(center, label_radius, f"{scores.loc[category]:.2f}",
                ha="center", va="center", fontsize=8.8, fontweight="bold",
                color=DEEP[category], zorder=5)


TREE_STYLE = {
    "Pinus": {"shape": "conical", "crown_low": 0.27, "crown_high": 0.96, "width": 0.30, "clusters": 170, "seed": 11},
    "Betula": {"shape": "column", "crown_low": 0.34, "crown_high": 0.94, "width": 0.24, "clusters": 135, "seed": 22},
    "Prunus": {"shape": "round", "crown_low": 0.38, "crown_high": 0.88, "width": 0.38, "clusters": 150, "seed": 33},
    "Quercus": {"shape": "broad", "crown_low": 0.35, "crown_high": 0.91, "width": 0.43, "clusters": 190, "seed": 44},
    "Acer": {"shape": "round", "crown_low": 0.38, "crown_high": 0.92, "width": 0.37, "clusters": 165, "seed": 55},
    "Alnus": {"shape": "irregular", "crown_low": 0.31, "crown_high": 0.94, "width": 0.29, "clusters": 145, "seed": 66},
    "Ulmus": {"shape": "vase", "crown_low": 0.34, "crown_high": 0.94, "width": 0.38, "clusters": 175, "seed": 77},
    "Sorbus": {"shape": "airy", "crown_low": 0.39, "crown_high": 0.92, "width": 0.31, "clusters": 120, "seed": 88},
    "Tilia": {"shape": "heart", "crown_low": 0.33, "crown_high": 0.94, "width": 0.38, "clusters": 180, "seed": 99},
}


def crown_half_width(shape: str, y_norm: np.ndarray, maximum: float) -> np.ndarray:
    if shape == "conical":
        return maximum * (0.20 + 0.90 * (1.0 - y_norm))
    if shape == "column":
        return maximum * (0.70 + 0.22 * np.sin(np.pi * y_norm))
    if shape == "broad":
        return maximum * (0.42 + 0.72 * np.sin(np.pi * y_norm) ** 0.72)
    if shape == "vase":
        return maximum * (0.36 + 0.78 * y_norm ** 0.65)
    if shape == "heart":
        return maximum * (0.38 + 0.72 * np.sin(np.pi * y_norm) + 0.18 * y_norm)
    if shape == "irregular":
        return maximum * (0.58 + 0.32 * np.sin(2.4 * np.pi * y_norm + 0.5) + 0.14 * y_norm)
    if shape == "airy":
        return maximum * (0.47 + 0.62 * np.sin(np.pi * y_norm) ** 0.9)
    return maximum * (0.40 + 0.67 * np.sin(np.pi * y_norm) ** 0.75)


def draw_tree(ax: plt.Axes, genus: str) -> None:
    style = TREE_STYLE[genus]
    rng = np.random.default_rng(style["seed"])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()

    # Ground and soil texture.
    ax.add_patch(Rectangle((0, 0), 1, 0.13, facecolor="#F6F5F2", edgecolor="none", zorder=0))
    ax.plot([0, 1], [0.13, 0.13], color="#667078", linewidth=0.75, zorder=8)
    dots_x = rng.uniform(0.02, 0.98, 44)
    dots_y = rng.uniform(0.02, 0.11, 44)
    ax.scatter(dots_x, dots_y, s=rng.uniform(1.0, 4.0, 44), facecolors="none",
               edgecolors="#C9C5BD", linewidths=0.35, zorder=1)
    ax.add_patch(Ellipse((0.50, 0.135), 0.40, 0.035, facecolor="#CDD0D2", alpha=0.30, edgecolor="none", zorder=1))

    trunk_top = style["crown_low"] + 0.24
    ax.add_patch(Polygon(
        [(0.475, 0.13), (0.525, 0.13), (0.513, trunk_top), (0.493, trunk_top)],
        closed=True, facecolor="#777B7E", edgecolor="#4F5559", linewidth=0.45, zorder=3,
    ))
    ax.plot([0.493, 0.505], [0.14, trunk_top], color="#D9DADB", linewidth=1.1, alpha=0.55, zorder=4)
    if genus == "Betula":
        for yy in np.linspace(0.20, trunk_top - 0.02, 7):
            ax.plot([0.480, 0.515], [yy, yy + rng.uniform(-0.005, 0.005)], color="#4B4F52", linewidth=0.6, zorder=5)

    # Branch skeleton gives the silhouette a dimensional, rather than icon-flat, appearance.
    for _ in range(14):
        y0 = rng.uniform(style["crown_low"], trunk_top)
        side = rng.choice([-1.0, 1.0])
        length = rng.uniform(0.10, style["width"] * 0.85)
        y1 = y0 + rng.uniform(0.08, 0.27)
        ax.plot([0.502, 0.502 + side * length], [y0, min(y1, style["crown_high"])],
                color="#5C6266", linewidth=rng.uniform(0.35, 0.9), alpha=0.70, zorder=2)

    n = style["clusters"]
    y_norm = rng.beta(1.45, 1.35, n)
    y = style["crown_low"] + y_norm * (style["crown_high"] - style["crown_low"])
    half_width = crown_half_width(style["shape"], y_norm, style["width"])
    x = 0.50 + rng.normal(0, 0.47, n) * half_width
    keep = np.abs(x - 0.50) <= half_width
    x, y, y_norm, half_width = x[keep], y[keep], y_norm[keep], half_width[keep]
    order = np.argsort(y)
    for idx in order:
        shade = int(np.clip(168 + 60 * (x[idx] - 0.50) / max(style["width"], 0.01) + rng.normal(0, 15), 92, 220))
        radius = rng.uniform(0.018, 0.040) * (0.85 if genus == "Pinus" else 1.0)
        width = radius * rng.uniform(1.3, 2.1)
        height = radius * rng.uniform(0.75, 1.35)
        ax.add_patch(Ellipse((x[idx], y[idx]), width, height,
                             angle=rng.uniform(-45, 45),
                             facecolor=f"#{shade:02x}{shade:02x}{shade:02x}",
                             edgecolor="#74797C", linewidth=0.18,
                             alpha=rng.uniform(0.28, 0.68), zorder=5))

    # Fine twig points improve the textured, localized-model appearance.
    point_count = 230 if genus != "Sorbus" else 150
    py_norm = rng.uniform(0.02, 0.98, point_count)
    py = style["crown_low"] + py_norm * (style["crown_high"] - style["crown_low"])
    pw = crown_half_width(style["shape"], py_norm, style["width"])
    px = 0.50 + rng.uniform(-1.0, 1.0, point_count) * pw
    ax.scatter(px, py, s=rng.uniform(0.15, 1.3, point_count), color="#545A5E", alpha=0.36, linewidths=0, zorder=6)


def add_summary_panel(ax: plt.Axes, category_data: pd.DataFrame) -> None:
    ax.set_axis_off()
    ax.text(0.02, 0.985, "Normalized environmental composition", transform=ax.transAxes,
            ha="left", va="top", fontsize=12.5, fontweight="bold", color="#202428")

    legend_ax = ax.inset_axes([0.00, 0.10, 0.28, 0.80])
    legend_ax.set_axis_off()
    legend_ax.add_patch(Rectangle((0.01, 0.01), 0.97, 0.98, facecolor="white",
                                  edgecolor="#B7BCC1", linewidth=0.8))
    legend_ax.text(0.08, 0.91, "Domain", fontsize=9.5, fontweight="bold", color="#2F353A")
    for idx, category in enumerate(CATEGORY_ORDER):
        yy = 0.78 - idx * 0.10
        legend_ax.add_patch(Rectangle((0.08, yy - 0.022), 0.075, 0.040,
                                      facecolor=DEEP[category], edgecolor="none"))
        legend_ax.text(0.19, yy, category, va="center", fontsize=8.6, color=DEEP[category], fontweight="bold")
    legend_ax.text(0.08, 0.45, "Association", fontsize=9.5, fontweight="bold", color="#2F353A")
    legend_ax.add_patch(Rectangle((0.08, 0.35), 0.075, 0.040, facecolor="#34434D", edgecolor="none"))
    legend_ax.text(0.19, 0.37, "Positive", va="center", fontsize=8.4, color="#343A40")
    legend_ax.add_patch(Rectangle((0.08, 0.27), 0.075, 0.040, facecolor="#B7D9E1", edgecolor="none"))
    legend_ax.text(0.19, 0.29, "Negative", va="center", fontsize=8.4, color="#343A40")
    legend_ax.add_patch(Rectangle((0.08, 0.19), 0.075, 0.040, facecolor=NONLINEAR, edgecolor="none"))
    legend_ax.text(0.19, 0.21, "Nonlinear", va="center", fontsize=8.4, color="#343A40")
    legend_ax.plot([0.08, 0.155], [0.125, 0.125], color="#202428", linewidth=1.5)
    legend_ax.text(0.19, 0.125, "SHAP = 0", va="center", fontsize=8.4, color="#343A40")
    legend_ax.text(0.08, 0.035, "Inward: negative\nOutward: positive", va="bottom", fontsize=7.3, color="#66717A")

    bar_ax = ax.inset_axes([0.33, 0.13, 0.65, 0.74])
    order = ["Tilia", "Ulmus", "Alnus", "Acer", "Quercus", "Prunus", "Betula", "Pinus", "Sorbus"]
    pivot = (
        category_data.pivot(index="Genus", columns="Category", values="Category_Composition_Share")
        .reindex(order)[CATEGORY_ORDER]
    )
    y = np.arange(len(pivot))
    left = np.zeros(len(pivot))
    for category in CATEGORY_ORDER:
        values = pivot[category].to_numpy(float) * 100.0
        bar_ax.barh(y, values, left=left, height=0.82, color=DEEP[category], alpha=0.70,
                    edgecolor="white", linewidth=0.45)
        for yy, x0, width in zip(y, left, values):
            if width >= 8:
                bar_ax.text(x0 + width / 2, yy, f"{width:.0f}%", ha="center", va="center",
                            fontsize=6.8, color="white", fontweight="bold")
        left += values
    bar_ax.set_xlim(0, 100)
    bar_ax.set_yticks(y)
    bar_ax.set_yticklabels(order, fontsize=8.3)
    bar_ax.invert_yaxis()
    bar_ax.set_xticks([0, 25, 50, 75, 100])
    bar_ax.set_xticklabels(["0", "25", "50", "75", "100%"], fontsize=7.3, color="#5A636B")
    bar_ax.xaxis.grid(True, linestyle=(0, (2, 3)), color="#D5DADF", linewidth=0.6)
    bar_ax.set_axisbelow(True)
    for spine in ["top", "right", "left"]:
        bar_ax.spines[spine].set_visible(False)
    bar_ax.spines["bottom"].set_color("#9AA1A7")
    bar_ax.tick_params(axis="y", length=0)
    bar_ax.tick_params(axis="x", length=2)
    ax.text(0.66, 0.035, "Share of each genus' aggregated absolute SHAP impact", transform=ax.transAxes,
            ha="center", va="bottom", fontsize=8.2, color="#616B74")


def create_figure(
    feature_data: pd.DataFrame,
    category_data: pd.DataFrame,
    metrics: pd.DataFrame,
    output_dir: Path,
    figure_stem: str,
    rotation_degrees_clockwise: float,
    contiguous_wedges: bool,
    show_category_background: bool,
) -> None:
    fig = plt.figure(figsize=(17.0, 12.2), facecolor="white")
    outer = fig.add_gridspec(
        2, 6, left=0.025, right=0.985, bottom=0.060, top=0.895,
        wspace=0.08, hspace=0.10, height_ratios=[1, 1]
    )
    summary_ax = fig.add_subplot(outer[0, 0:3])
    add_summary_panel(summary_ax, category_data)

    layout = [
        ("Pinus", outer[0, 3]), ("Betula", outer[0, 4]), ("Prunus", outer[0, 5]),
        ("Quercus", outer[1, 0]), ("Acer", outer[1, 1]), ("Alnus", outer[1, 2]),
        ("Ulmus", outer[1, 3]), ("Sorbus", outer[1, 4]), ("Tilia", outer[1, 5]),
    ]
    global_abs_limit = float(
        max(feature_data["SHAP_P05"].abs().max(), feature_data["SHAP_P95"].abs().max())
    )
    metrics_indexed = metrics.set_index("Genus")
    for genus, cell in layout:
        nested = cell.subgridspec(2, 1, height_ratios=[0.48, 0.52], hspace=-0.08)
        polar_ax = fig.add_subplot(nested[0, 0], projection="polar")
        tree_ax = fig.add_subplot(nested[1, 0])
        g_feature = feature_data.loc[feature_data["Genus"].eq(genus)]
        g_category = category_data.loc[category_data["Genus"].eq(genus)].copy()
        g_category["Category"] = g_category["Category"].astype(str)
        draw_score_ring(
            polar_ax,
            g_feature,
            g_category,
            global_abs_limit,
            rotation_degrees_clockwise,
            contiguous_wedges,
            show_category_background,
        )
        draw_tree(tree_ax, genus)
        metric = metrics_indexed.loc[genus]
        tree_ax.text(0.50, 0.055, genus, transform=tree_ax.transAxes,
                     ha="center", va="center", fontsize=11.5, fontweight="bold", color="#22272B")
        tree_ax.text(0.50, 0.008,
                     f"n = {int(metric['N_Rows']):,} rows / {int(metric['N_Trees']):,} trees",
                     transform=tree_ax.transAxes, ha="center", va="bottom",
                     fontsize=6.8, color="#6A737B")

    fig.suptitle("Species-specific sensitivity profiles and structural composition",
                 y=0.972, fontsize=20, fontweight="bold", color="#202428")
    fig.text(0.5, 0.927,
             "Pooled period-controlled XGBoost · independent-test genus subsets · robust P05-P95 SHAP ranges",
             ha="center", fontsize=11.2, color="#59636D")
    fig.text(0.5, 0.020,
             "Black circle = SHAP 0; each wedge spans P05-P95, with negative portions inward and positive portions outward. Category numbers = Σ(|P05| + |P95|).",
             ha="center", fontsize=9.0, color="#555F68")

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / figure_stem
    fig.savefig(stem.with_suffix(".png"), dpi=450, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    statistics = pd.read_csv(args.statistics)
    deciles = pd.read_csv(args.deciles)
    feature_data = build_feature_data(statistics, deciles)
    category_data = build_category_data(feature_data)
    metrics = load_metrics(args.metrics)

    plots_dir = args.output_dir / "plots"
    tables_dir = args.output_dir / "tables"
    plots_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    feature_data.to_csv(tables_dir / "species_feature_categorization.csv", index=False)
    feature_data.loc[
        feature_data["Classification_Changed_After_Audit"].astype(bool)
    ].to_csv(tables_dir / "species_feature_recolor_audit.csv", index=False)
    category_data.to_csv(tables_dir / "species_category_scores_and_composition.csv", index=False)
    metrics.to_csv(tables_dir / "species_pooled_xgb_test_performance.csv", index=False)
    score_wide = category_data.pivot(
        index="Genus", columns="Category",
        values=["Category_Absolute_SHAP_Impact", "Category_Composition_Share"],
    )
    score_wide.columns = [f"{measure}_{category}" for measure, category in score_wide.columns]
    score_wide = score_wide.reset_index()
    dominant = (
        category_data.sort_values(
            ["Genus", "Category_Absolute_SHAP_Impact"], ascending=[True, False]
        )
        .drop_duplicates("Genus")
        [["Genus", "Category", "Category_Absolute_SHAP_Impact", "Category_Composition_Share"]]
        .rename(columns={
            "Category": "Dominant_Category",
            "Category_Absolute_SHAP_Impact": "Dominant_Category_Score",
            "Category_Composition_Share": "Dominant_Category_Share",
        })
    )
    score_wide = score_wide.merge(dominant, on="Genus").merge(metrics, on="Genus")
    score_wide.to_csv(tables_dir / "species_category_profile_summary_wide.csv", index=False)
    create_figure(
        feature_data,
        category_data,
        metrics,
        plots_dir,
        args.figure_stem,
        args.rotation_degrees_clockwise,
        args.contiguous_wedges,
        not args.hide_category_background,
    )

    metadata = {
        "statistics_source": str(args.statistics),
        "deciles_source": str(args.deciles),
        "metrics_source": str(args.metrics),
        "genera": GENERA,
        "figure_geometry": {
            "rotation_degrees_clockwise": args.rotation_degrees_clockwise,
            "contiguous_wedges": args.contiguous_wedges,
            "show_category_background": not args.hide_category_background,
            "figure_stem": args.figure_stem,
        },
        "radial_definition": "The black circle is SHAP zero. Each wedge spans P05-P95; negative portions are drawn inward and positive portions outward, using one common absolute SHAP scale across genera.",
        "magnitude_definition": "Feature absolute impact is abs(P05) + abs(P95); category impact is the sum of member-feature absolute impacts.",
        "composition_definition": "Category absolute impact divided by total absolute impact across all 11 environmental predictors within genus.",
        "association_rules": {
            "continuous": (
                "Nonlinear when abs(Spearman) < 0.25 or decile monotonicity < 0.65. "
                "A borderline safeguard retains direction when monotonicity is 0.64-0.65, "
                "abs(Spearman) >= 0.50, and Spearman/end-contrast signs agree."
            ),
            "binary": "Sign of high-minus-low SHAP contrast.",
            "diversity_display": "Mono_Rate is displayed as Diversity by reversing feature values and the correlation/contrast signs.",
        },
        "scientific_scope": "Species subsets of one pooled period-controlled XGBoost model; model associations, not causal effects.",
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    changed = feature_data.loc[
        feature_data["Classification_Changed_After_Audit"].astype(bool),
        [
            "Genus",
            "Display_Variable",
            "Previous_Association_Class",
            "Association_Class",
            "Feature_SHAP_Spearman_for_Display_Variable",
            "Decile_Monotonicity_Score",
            "High_minus_Low_SHAP_for_Display_Variable",
            "Classification_Reason",
        ],
    ]
    changed_lines = [
        "- "
        + f"{row.Genus} — {row.Display_Variable}: "
        + f"{row.Previous_Association_Class} → {row.Association_Class}"
        for row in changed.itertuples(index=False)
    ]
    if not changed_lines:
        changed_lines = ["- No association colors changed after the audit."]
    run_log = "\n".join(
        [
            "# Run log — rotated contiguous species SHAP profiles",
            "",
            "## Figure revision",
            "",
            f"- Every radial profile was rotated {args.rotation_degrees_clockwise:g}° clockwise.",
            f"- Contiguous wedge geometry: {args.contiguous_wedges}.",
            f"- Pale category backgrounds shown: {not args.hide_category_background}.",
            "- The SHAP = 0 circle, P05–P95 wedge magnitudes, category totals, and normalized composition were unchanged.",
            "",
            "## Nonlinear-color audit",
            "",
            "Continuous features were checked using both feature–SHAP Spearman direction and the saved dependence-decile monotonicity score. Binary indicators remain categorical positive/negative contrasts and are not labelled nonlinear.",
            "",
            *changed_lines,
            "",
            "## Inputs",
            "",
            f"- `{args.statistics}`",
            f"- `{args.deciles}`",
            f"- `{args.metrics}`",
            "",
            "No model was retrained and no source data were changed.",
            "",
        ]
    )
    (args.output_dir / "RUN_LOG.md").write_text(run_log, encoding="utf-8")
    print(category_data.to_string(index=False))
    print("\nRecolor audit")
    print(changed.to_string(index=False))


if __name__ == "__main__":
    main()
