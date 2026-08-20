#!/usr/bin/env python3
"""Create standard SHAP dependence plots with interaction-feature coloring.

For each requested feature, the x axis is the raw feature value, the y axis is
that feature's raw SHAP value, and point color is the strongest second feature
identified by SHAP's approximate interaction-ranking method over the complete
pooled model feature matrix.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from xgboost import XGBRegressor


DEFAULT_TARGET_FEATURES = ["avg_LST", "lightemiss", "type_Puisto"]
SHAP_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "shap_red_blue", ["#008BFB", "#FF0052"]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--pipeline-script", required=True, type=Path)
    parser.add_argument("--soil-script", required=True, type=Path)
    parser.add_argument("--full-analysis", required=True, type=Path)
    parser.add_argument("--saved-observations", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--target-features",
        nargs="+",
        default=DEFAULT_TARGET_FEATURES,
        help=(
            "Model-matrix features to plot. The default reproduces the original "
            "three-panel LST, illumination, and park-context figure."
        ),
    )
    parser.add_argument(
        "--figure-stem",
        default="Figure4b_SHAP_dependence_interactions",
        help="Filename stem for the combined figure.",
    )
    parser.add_argument(
        "--interaction-candidates",
        nargs="+",
        default=None,
        help=(
            "Optional feature subset eligible for interaction coloring. "
            "Ranking is still computed by SHAP over the full model matrix, "
            "then restricted to this subset."
        ),
    )
    parser.add_argument(
        "--figure-title",
        default=None,
        help="Optional title for the combined figure.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def feature_label(feature: str, pipeline, soil) -> str:
    labels = {
        "Log_Height": "Log initial height",
        **pipeline.FEATURE_LABELS,
        **soil.SOIL_LABELS,
    }
    if feature in labels:
        return labels[feature]
    if feature.startswith("Species_"):
        return f"Species: {feature[len('Species_'):].replace('_', ' ')}"
    if feature.startswith("Period_"):
        period = feature[len("Period_"):].replace("_", "–")
        return f"Monitoring period: {period}"
    return feature.replace("_", " ")


def x_axis_label(feature: str, pipeline, soil) -> str:
    if feature == "avg_LST":
        return "Land-surface temperature (°C)"
    if feature == "lightemiss":
        return "Nighttime illumination (relative units)"
    if feature == "avg_noise_day":
        return "Daytime noise (dB)"
    if feature == "type_Puisto":
        return "Park context"
    if feature == "soil_moraine":
        return "Moraine context"
    if feature == "Mono_Rate":
        return "Monoculture rate (proportion)"
    if feature == "avg_svf":
        return "Sky-view factor (proportion)"
    return feature_label(feature, pipeline, soil)


def binary_level_labels(feature: str, pipeline, soil) -> list[str] | None:
    if feature == "type_Puisto":
        return ["Street/non-park", "Park"]
    if feature == "soil_moraine":
        return ["Other substrate", "Moraine"]
    if feature == "soil_bedrock":
        return ["Other substrate", "Bedrock"]
    if feature == "soil_infill":
        return ["Other substrate", "Fill"]
    if feature.startswith("Species_"):
        return ["Other species", feature_label(feature, pipeline, soil)]
    if feature.startswith("Period_"):
        return ["Other period", feature_label(feature, pipeline, soil)]
    return None


def target_bins(values: pd.Series, feature: str) -> tuple[pd.Series, list[str]]:
    unique = np.sort(values.dropna().unique())
    if len(unique) <= 2:
        mapping = {value: index for index, value in enumerate(unique)}
        labels = [f"{value:g}" for value in unique]
        if feature == "type_Puisto" and len(unique) == 2:
            labels = ["Street/non-park", "Park"]
        elif feature == "soil_moraine" and len(unique) == 2:
            labels = ["Other substrate", "Moraine"]
        return values.map(mapping).astype(int), labels
    if feature == "lightemiss":
        edges = [-np.inf, 45, 55, 65, 75, 85, 95, 105, 125, 165, np.inf]
        labels = [
            "30–40",
            "50",
            "60",
            "70",
            "80",
            "90",
            "100",
            "110–120",
            "130–160",
            "170–190",
        ]
        cut = pd.cut(values, bins=edges, labels=labels, include_lowest=True)
        return pd.Series(cut.cat.codes, index=values.index), labels
    if feature == "avg_noise_day" and len(unique) <= 20:
        mapping = {value: index for index, value in enumerate(unique)}
        labels = [f"{value:g}" for value in unique]
        return values.map(mapping).astype(int), labels
    cut = pd.qcut(values, q=10, duplicates="drop")
    labels = [f"Q{index}" for index in range(1, len(cut.cat.categories) + 1)]
    return pd.Series(cut.cat.codes, index=values.index), labels


def interaction_groups(values: pd.Series) -> tuple[pd.Series, dict[int, str], dict]:
    unique = np.sort(values.dropna().unique())
    if len(unique) <= 2:
        mapping = {value: index for index, value in enumerate(unique)}
        groups = values.map(mapping).astype(int)
        labels = {index: f"Value {value:g}" for value, index in mapping.items()}
        metadata = {"type": "exact", "values": unique.tolist()}
        return groups, labels, metadata
    q25, q75 = values.quantile([0.25, 0.75])
    groups = pd.Series(-1, index=values.index, dtype=int)
    groups.loc[values <= q25] = 0
    groups.loc[values >= q75] = 1
    labels = {0: f"Low (≤ {q25:.3g})", 1: f"High (≥ {q75:.3g})"}
    metadata = {"type": "quartiles", "q25": float(q25), "q75": float(q75)}
    return groups, labels, metadata


def interaction_strata_table(
    x_test: pd.DataFrame,
    shap_values: np.ndarray,
    target: str,
    interaction: str,
    target_labels: list[str],
) -> tuple[pd.DataFrame, dict]:
    target_index = x_test.columns.get_loc(target)
    values = x_test[target].reset_index(drop=True)
    colors = x_test[interaction].reset_index(drop=True)
    target_codes, _ = target_bins(values, target)
    interaction_codes, interaction_labels, metadata = interaction_groups(colors)
    frame = pd.DataFrame(
        {
            "Target_Value": values,
            "Target_Bin_Code": target_codes,
            "Interaction_Value": colors,
            "Interaction_Group_Code": interaction_codes,
            "SHAP": shap_values[:, target_index],
        }
    )
    frame = frame[frame["Interaction_Group_Code"].isin([0, 1])]
    rows = []
    for (target_code, interaction_code), subset in frame.groupby(
        ["Target_Bin_Code", "Interaction_Group_Code"], observed=True
    ):
        rows.append(
            {
                "Target_Feature": target,
                "Interaction_Feature": interaction,
                "Target_Bin_Code": int(target_code),
                "Target_Bin_Label": target_labels[int(target_code)],
                "Target_Feature_Median": float(subset["Target_Value"].median()),
                "Interaction_Group": interaction_labels[int(interaction_code)],
                "Interaction_Value_Median": float(
                    subset["Interaction_Value"].median()
                ),
                "N": int(len(subset)),
                "Mean_SHAP": float(subset["SHAP"].mean()),
                "Median_SHAP": float(subset["SHAP"].median()),
            }
        )
    return pd.DataFrame(rows), metadata


def color_norm(values: np.ndarray) -> tuple[mcolors.Normalize, float, float]:
    finite = values[np.isfinite(values)]
    unique = np.unique(finite)
    if len(unique) <= 2:
        low, high = float(np.min(finite)), float(np.max(finite))
    else:
        low, high = np.quantile(finite, [0.05, 0.95])
    if low == high:
        high = low + 1.0
    return mcolors.Normalize(vmin=low, vmax=high, clip=True), low, high


def dependence_panel(
    *,
    fig: plt.Figure,
    axis: plt.Axes,
    x_test: pd.DataFrame,
    shap_values: np.ndarray,
    target: str,
    interaction: str,
    pipeline,
    soil,
    rng: np.random.Generator,
    panel_label: str | None = None,
) -> None:
    target_index = x_test.columns.get_loc(target)
    x = x_test[target].to_numpy(dtype=float)
    y = shap_values[:, target_index].astype(float)
    color = x_test[interaction].to_numpy(dtype=float)
    draw_order = rng.permutation(len(x))
    plot_x = x.copy()
    unique_x = np.unique(x[np.isfinite(x)])
    if len(unique_x) <= 2:
        plot_x = plot_x + rng.normal(0, 0.045, len(plot_x))
    elif len(unique_x) <= 20:
        positive_steps = np.diff(unique_x)
        positive_steps = positive_steps[positive_steps > 0]
        jitter_scale = 0.085 * float(np.min(positive_steps))
        plot_x = plot_x + rng.normal(0, jitter_scale, len(plot_x))
    norm, color_low, color_high = color_norm(color)
    scatter = axis.scatter(
        plot_x[draw_order],
        y[draw_order],
        c=color[draw_order],
        cmap=SHAP_CMAP,
        norm=norm,
        s=9,
        alpha=0.66,
        linewidths=0,
        rasterized=True,
    )
    axis.axhline(0, color="#707782", linewidth=0.85, linestyle="--")
    axis.set_xlabel(x_axis_label(target, pipeline, soil))
    axis.set_ylabel(f"SHAP value for {feature_label(target, pipeline, soil)}")
    target_binary_labels = binary_level_labels(target, pipeline, soil)
    if len(unique_x) <= 2 and target_binary_labels is not None:
        axis.set_xticks(unique_x, target_binary_labels)
    title = feature_label(target, pipeline, soil)
    if panel_label:
        title = f"{panel_label}  {title}"
    axis.set_title(title, fontsize=11.2, weight="bold", pad=8)
    axis.grid(axis="y", color="#E5E7EB", linewidth=0.55)
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(labelsize=8.5)
    colorbar = fig.colorbar(scatter, ax=axis, pad=0.025, fraction=0.05)
    colorbar.set_label(
        f"Interaction: {feature_label(interaction, pipeline, soil)}",
        fontsize=8.5,
    )
    colorbar.ax.tick_params(labelsize=7.8)
    if np.unique(color).size <= 2:
        color_unique = np.unique(color)
        colorbar.set_ticks(color_unique)
        color_binary_labels = binary_level_labels(interaction, pipeline, soil)
        if color_binary_labels is not None and len(color_unique) == 2:
            colorbar.set_ticklabels(color_binary_labels)
    else:
        colorbar.set_ticks([color_low, color_high])
        colorbar.set_ticklabels(["Low", "High"])


def main() -> None:
    args = parse_args()
    target_features = list(dict.fromkeys(args.target_features))
    if not target_features:
        raise ValueError("At least one target feature is required")
    if args.output.exists():
        raise FileExistsError(f"Output directory already exists: {args.output}")
    plots = args.output / "plots"
    tables = args.output / "tables"
    plots.mkdir(parents=True)
    tables.mkdir()

    pipeline = load_module(args.pipeline_script, "relative_growth_pipeline")
    soil = load_module(args.soil_script, "soil_pipeline")
    raw = pd.read_csv(args.input)
    raw, _ = soil.add_soil_indicators(raw)
    long_data, construction = pipeline.build_long_data(raw)
    data, encoding = pipeline.add_split_and_dummies(long_data)

    model = XGBRegressor(**pipeline.XGB_PARAMETERS)
    model.load_model(
        args.full_analysis / "models/xgb_period_controlled_soil_augmented.json"
    )
    feature_columns = list(model.get_booster().feature_names or [])
    if not feature_columns:
        raise AssertionError("The frozen XGBoost model does not contain feature names")
    _, x_test, _ = pipeline.prepare_matrices(data, feature_columns)
    x_test = x_test.reset_index(drop=True)
    explainer = shap.TreeExplainer(model)
    shap_values = np.asarray(explainer.shap_values(x_test), dtype=np.float32)

    test_data = data.loc[data["Split"].eq("Test")].reset_index(drop=True)
    saved = pd.read_csv(args.saved_observations)
    key_columns = ["OID_", "Period", "Species_Name_Model"]
    current_keys = test_data[key_columns].astype(str).agg("|".join, axis=1)
    saved_keys = saved[key_columns].astype(str).agg("|".join, axis=1)
    if current_keys.duplicated().any() or saved_keys.duplicated().any():
        raise AssertionError("Tree-period-species alignment keys are not unique")
    if not np.array_equal(current_keys.to_numpy(), saved_keys.to_numpy()):
        current_index = pd.Index(current_keys)
        indexer = current_index.get_indexer(saved_keys)
        if len(indexer) != len(saved) or np.any(indexer < 0):
            raise AssertionError(
                "Saved SHAP rows are not the same observations as the reconstructed test rows"
            )
        x_test = x_test.iloc[indexer].reset_index(drop=True)
        shap_values = shap_values[indexer]
        test_data = test_data.iloc[indexer].reset_index(drop=True)
        current_keys = test_data[key_columns].astype(str).agg("|".join, axis=1)
        if not np.array_equal(current_keys.to_numpy(), saved_keys.to_numpy()):
            raise AssertionError("Could not reorder reconstructed rows to saved SHAP rows")

    validation_rows = []
    ranking_rows = []
    selected_interactions: dict[str, str] = {}
    missing_targets = [target for target in target_features if target not in x_test]
    if missing_targets:
        raise KeyError(f"Target features are absent from the model matrix: {missing_targets}")
    interaction_candidates = (
        list(dict.fromkeys(args.interaction_candidates))
        if args.interaction_candidates is not None
        else feature_columns
    )
    missing_candidates = [
        feature for feature in interaction_candidates if feature not in feature_columns
    ]
    if missing_candidates:
        raise KeyError(
            f"Interaction candidates are absent from the model matrix: {missing_candidates}"
        )
    interaction_candidate_set = set(interaction_candidates)

    for target in target_features:
        target_index = x_test.columns.get_loc(target)
        saved_feature_values = saved[f"{target}__value"].to_numpy(dtype=float)
        matrix_feature_values = x_test[target].to_numpy(dtype=float)
        saved_values = saved[f"{target}__shap"].to_numpy(dtype=float)
        computed_values = shap_values[:, target_index].astype(float)
        validation_rows.append(
            {
                "Feature": target,
                "N": len(saved_values),
                "Maximum_Absolute_Feature_Value_Difference": float(
                    np.max(np.abs(saved_feature_values - matrix_feature_values))
                ),
                "Maximum_Absolute_SHAP_Difference": float(
                    np.max(np.abs(saved_values - computed_values))
                ),
                "Mean_Absolute_SHAP_Difference": float(
                    np.mean(np.abs(saved_values - computed_values))
                ),
            }
        )
        # The frozen observation-level SHAP table is the source of truth for
        # the published model analysis. Reuse those values exactly so plotting
        # remains stable across SHAP/XGBoost runtime versions.
        shap_values[:, target_index] = saved_values.astype(np.float32)
        order = shap.approximate_interactions(target_index, shap_values, x_test)
        ordered_nonself = [int(index) for index in order if int(index) != target_index]
        overall_rank = {
            feature_columns[index]: rank
            for rank, index in enumerate(ordered_nonself, start=1)
        }
        filtered = [
            index
            for index in ordered_nonself
            if feature_columns[index] in interaction_candidate_set
        ]
        if not filtered:
            raise AssertionError(f"No interaction feature found for {target}")
        selected_interactions[target] = feature_columns[filtered[0]]
        for rank, index in enumerate(filtered[:10], start=1):
            feature = feature_columns[index]
            ranking_rows.append(
                {
                    "Target_Feature": target,
                    "Target_Label": feature_label(target, pipeline, soil),
                    "Approximate_Interaction_Rank": rank,
                    "Approximate_Interaction_Overall_Rank": overall_rank[feature],
                    "Interaction_Feature": feature,
                    "Interaction_Label": feature_label(feature, pipeline, soil),
                    "Selected_For_Color": rank == 1,
                }
            )
    validation = pd.DataFrame(validation_rows)
    if validation["Maximum_Absolute_Feature_Value_Difference"].max() > 1e-5:
        raise AssertionError("Reconstructed feature values do not reproduce saved values")

    ranking = pd.DataFrame(ranking_rows)
    strata_tables = []
    strata_metadata = {}
    for target, interaction in selected_interactions.items():
        _, target_labels = target_bins(x_test[target], target)
        table, metadata = interaction_strata_table(
            x_test,
            shap_values,
            target,
            interaction,
            target_labels,
        )
        table["Target_Label"] = feature_label(target, pipeline, soil)
        table["Interaction_Label"] = feature_label(interaction, pipeline, soil)
        strata_tables.append(table)
        strata_metadata[target] = metadata
    strata = pd.concat(strata_tables, ignore_index=True)

    observation_output = test_data[
        ["OID_", "Period", "Species_Name_Model"]
    ].copy()
    for target, interaction in selected_interactions.items():
        target_index = x_test.columns.get_loc(target)
        observation_output[f"{target}__value"] = x_test[target].to_numpy()
        observation_output[f"{target}__shap"] = shap_values[:, target_index]
        observation_output[f"{target}__interaction_feature"] = interaction
        observation_output[f"{target}__interaction_value"] = x_test[
            interaction
        ].to_numpy()

    validation.to_csv(tables / "shap_recomputation_validation.csv", index=False)
    ranking.to_csv(tables / "approximate_interaction_ranking.csv", index=False)
    strata.to_csv(tables / "interaction_color_strata_summary.csv", index=False)
    observation_output.to_csv(
        tables / "shap_dependence_interaction_observations.csv", index=False
    )

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.3,
            "axes.labelcolor": "#20242A",
            "xtick.color": "#343A40",
            "ytick.color": "#343A40",
        }
    )
    rng = np.random.default_rng(args.random_state + 73000)
    target_count = len(target_features)
    if target_count <= 3:
        row_count, column_count = 1, target_count
        figure_size = (5.45 * target_count, 5.0)
    else:
        column_count = 3
        row_count = int(np.ceil(target_count / column_count))
        figure_size = (17.4, 4.25 * row_count)
    fig, axes = plt.subplots(row_count, column_count, figsize=figure_size, squeeze=False)
    flat_axes = list(axes.flat)
    if target_count == 7 and row_count == 3 and column_count == 3:
        plot_axes = flat_axes[:6] + [axes[2, 1]]
        unused_axes = [axes[2, 0], axes[2, 2]]
    else:
        plot_axes = flat_axes[:target_count]
        unused_axes = flat_axes[target_count:]
    for axis in unused_axes:
        axis.set_visible(False)
    panel_labels = [chr(ord("A") + index) for index in range(target_count)]
    for panel_label, axis, target in zip(panel_labels, plot_axes, target_features):
        dependence_panel(
            fig=fig,
            axis=axis,
            x_test=x_test,
            shap_values=shap_values,
            target=target,
            interaction=selected_interactions[target],
            pipeline=pipeline,
            soil=soil,
            rng=rng,
            panel_label=panel_label,
        )
    figure_title = args.figure_title or (
        "SHAP dependence and strongest approximate interactions · "
        "pooled XGBoost independent test (n = 6,845)"
    )
    fig.suptitle(
        figure_title,
        fontsize=13.8,
        weight="bold",
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.965], w_pad=1.4, h_pad=2.0)
    for extension in ["png", "pdf", "svg"]:
        fig.savefig(
            plots / f"{args.figure_stem}.{extension}",
            dpi=350 if extension == "png" else None,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)

    for target in target_features:
        fig, axis = plt.subplots(figsize=(6.7, 4.9))
        dependence_panel(
            fig=fig,
            axis=axis,
            x_test=x_test,
            shap_values=shap_values,
            target=target,
            interaction=selected_interactions[target],
            pipeline=pipeline,
            soil=soil,
            rng=rng,
        )
        fig.tight_layout()
        for extension in ["png", "pdf", "svg"]:
            fig.savefig(
                plots / f"{target}_SHAP_dependence_interaction.{extension}",
                dpi=350 if extension == "png" else None,
                bbox_inches="tight",
                facecolor="white",
            )
        plt.close(fig)

    metadata = {
        "input": str(args.input),
        "pipeline_script": str(args.pipeline_script),
        "soil_script": str(args.soil_script),
        "full_analysis": str(args.full_analysis),
        "saved_observations": str(args.saved_observations),
        "shap_value_source": (
            "Frozen observation-level SHAP values from saved_observations; "
            "fresh TreeExplainer values are used only for numerical validation."
        ),
        "model_scope": "Pooled period-controlled three-soil XGBoost",
        "sample": "Independent test",
        "observations": int(len(x_test)),
        "trees": int(test_data["OID_"].nunique()),
        "target_features": target_features,
        "interaction_candidates": interaction_candidates,
        "selected_interactions": selected_interactions,
        "interaction_group_metadata": strata_metadata,
        "interaction_selection": (
            "Highest-ranked non-self feature returned by "
            "shap.approximate_interactions after restricting color selection "
            "to interaction_candidates."
        ),
        "color_scaling": (
            "Continuous color features clipped to their 5th–95th percentiles; "
            "binary features use their observed 0/1 range."
        ),
        "construction": construction,
        "interpretation_warning": (
            "Color separation is an interaction/dependence diagnostic, not a "
            "causal interaction estimate. Correlated predictors can contribute."
        ),
    }
    (args.output / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    interaction_lines = [
        f"- {feature_label(target, pipeline, soil)}: "
        f"{feature_label(interaction, pipeline, soil)}"
        for target, interaction in selected_interactions.items()
    ]
    run_log = "\n".join(
        [
            "# SHAP dependence plots with interaction coloring",
            "",
            "## Scope",
            "",
            "- Model: pooled, one-hot, period-controlled three-soil XGBoost",
            f"- Sample: independent test ({len(x_test):,} observations; "
            f"{test_data['OID_'].nunique():,} trees)",
            "- Axes: raw feature value on x; observation-level SHAP value on y",
            "- Color: highest-ranked non-self feature from "
            "`shap.approximate_interactions` within the specified candidate set",
            f"- Eligible color features: {', '.join(interaction_candidates)}",
            "- SHAP scale: contribution to log annualized specific carbon growth",
            "- SHAP source: frozen observation-level values from the completed "
            "three-soil analysis (fresh recomputation used only for validation)",
            "",
            "## Selected interaction features",
            "",
            *interaction_lines,
            "",
            "## Interpretation constraint",
            "",
            "These plots diagnose fitted dependence and possible interaction "
            "structure; they do not identify causal effects. For binary targets "
            "such as park and moraine context, two x-axis clusters represent "
            "the two indicator levels, while vertical spread reflects conditional "
            "heterogeneity rather than a continuous nonlinear response.",
            "",
            "## Outputs",
            "",
            f"- Combined figure: `plots/{args.figure_stem}.png`, `.pdf`, `.svg`",
            "- Individual feature figures: `plots/*_SHAP_dependence_interaction.*`",
            "- Observation-level plotting data and approximate-interaction rankings: "
            "`tables/`",
            "",
        ]
    )
    (args.output / "RUN_LOG.md").write_text(run_log, encoding="utf-8")
    print("Selected interaction features")
    for target, interaction in selected_interactions.items():
        print(f"{target}: {interaction}")
    print("\nSHAP validation")
    print(validation.to_string(index=False))
    print("\nTop interaction rankings")
    print(ranking.groupby("Target_Feature", sort=False).head(5).to_string(index=False))


if __name__ == "__main__":
    main()
