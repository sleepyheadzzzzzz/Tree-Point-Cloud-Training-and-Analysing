#!/usr/bin/env python3
"""Quantify and plot nonlinear pooled-XGBoost SHAP dependence patterns.

The analysis uses the saved independent-test observation-level SHAP values from
the frozen three-soil, period-controlled pooled XGBoost model. Repeated periods
from the same tree are respected by resampling complete OID_ clusters when
estimating uncertainty for binned mean SHAP contributions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler


FEATURES = {
    "avg_LST": {
        "label": "Land-surface temperature",
        "x_label": "Land-surface temperature (°C)",
        "kind": "quantile",
        "seed": 4101,
    },
    "lightemiss": {
        "label": "Nighttime illumination",
        "x_label": "Nighttime illumination (relative units)",
        "kind": "illumination",
        "seed": 4102,
    },
    "type_Puisto": {
        "label": "Park context",
        "x_label": "Site context",
        "kind": "binary",
        "seed": 4103,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--observations", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bootstrap-repetitions", type=int, default=1500)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def assign_bins(values: pd.Series, kind: str) -> tuple[pd.Series, list[str]]:
    if kind == "quantile":
        bins = pd.qcut(values, q=10, duplicates="drop")
        categories = list(bins.cat.categories)
        labels = [f"Q{index}" for index in range(1, len(categories) + 1)]
        codes = bins.cat.codes
    elif kind == "illumination":
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
        codes = cut.cat.codes
    elif kind == "binary":
        labels = ["Street/non-park", "Park"]
        codes = values.round().astype(int)
    else:
        raise ValueError(kind)
    return pd.Series(codes, index=values.index, dtype=int), labels


def tree_cluster_weights(
    tree_ids: pd.Series,
    repetitions: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    unique_trees, tree_codes = np.unique(tree_ids.to_numpy(), return_inverse=True)
    rng = np.random.default_rng(random_state)
    probabilities = np.full(len(unique_trees), 1.0 / len(unique_trees))
    weights = rng.multinomial(len(unique_trees), probabilities, size=repetitions)
    return unique_trees, tree_codes, weights


def bin_summary(
    frame: pd.DataFrame,
    feature: str,
    labels: list[str],
    tree_codes: np.ndarray,
    weights: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray]:
    bin_codes = frame["Bin_Code"].to_numpy(dtype=int)
    shap_values = frame["SHAP"].to_numpy(dtype=float)
    feature_values = frame["Feature_Value"].to_numpy(dtype=float)
    n_trees = weights.shape[1]
    n_bins = len(labels)
    sums = np.zeros((n_trees, n_bins), dtype=float)
    counts = np.zeros((n_trees, n_bins), dtype=float)
    np.add.at(sums, (tree_codes, bin_codes), shap_values)
    np.add.at(counts, (tree_codes, bin_codes), 1.0)
    boot_sums = weights @ sums
    boot_counts = weights @ counts
    boot_means = np.divide(
        boot_sums,
        boot_counts,
        out=np.full_like(boot_sums, np.nan),
        where=boot_counts > 0,
    )

    rows = []
    for code, label in enumerate(labels):
        mask = bin_codes == code
        values = feature_values[mask]
        shap_subset = shap_values[mask]
        draws = boot_means[:, code]
        rows.append(
            {
                "Feature": feature,
                "Feature_Label": FEATURES[feature]["label"],
                "Bin_Code": code,
                "Bin_Label": label,
                "Feature_Min": float(np.min(values)),
                "Feature_Median": float(np.median(values)),
                "Feature_Max": float(np.max(values)),
                "N_Observations": int(mask.sum()),
                "N_Trees": int(frame.loc[mask, "OID_"].nunique()),
                "Mean_SHAP": float(np.mean(shap_subset)),
                "Median_SHAP": float(np.median(shap_subset)),
                "SHAP_P05": float(np.quantile(shap_subset, 0.05)),
                "SHAP_P95": float(np.quantile(shap_subset, 0.95)),
                "Cluster_Bootstrap_Mean_CI95_Lower": float(
                    np.nanquantile(draws, 0.025)
                ),
                "Cluster_Bootstrap_Mean_CI95_Upper": float(
                    np.nanquantile(draws, 0.975)
                ),
            }
        )
    return pd.DataFrame(rows), boot_means


def group_cv_nonlinearity(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
) -> tuple[float, float, float]:
    splitter = GroupKFold(n_splits=5)
    x_matrix = x.reshape(-1, 1)
    linear = LinearRegression()
    spline = make_pipeline(
        SplineTransformer(n_knots=6, degree=3, include_bias=False),
        StandardScaler(),
        Ridge(alpha=1.0),
    )
    linear_prediction = cross_val_predict(
        linear, x_matrix, y, groups=groups, cv=splitter, n_jobs=1
    )
    spline_prediction = cross_val_predict(
        spline, x_matrix, y, groups=groups, cv=splitter, n_jobs=1
    )
    linear_r2 = float(r2_score(y, linear_prediction))
    spline_r2 = float(r2_score(y, spline_prediction))
    return linear_r2, spline_r2, spline_r2 - linear_r2


def sign_changes(values: np.ndarray) -> int:
    signs = np.sign(values)
    signs = signs[signs != 0]
    if len(signs) < 2:
        return 0
    return int(np.sum(signs[1:] != signs[:-1]))


def plot_panel(
    axis: plt.Axes,
    frame: pd.DataFrame,
    bins: pd.DataFrame,
    feature: str,
    sample_indices: np.ndarray,
) -> None:
    config = FEATURES[feature]
    x = frame["Feature_Value"].to_numpy(dtype=float)
    y = frame["SHAP"].to_numpy(dtype=float)
    if config["kind"] == "binary":
        rng = np.random.default_rng(config["seed"])
        scatter_x = x[sample_indices] + rng.normal(0, 0.035, len(sample_indices))
        axis.set_xticks([0, 1], ["Street/non-park", "Park"])
        line_x = bins["Bin_Code"].to_numpy(dtype=float)
    else:
        scatter_x = x[sample_indices]
        if config["kind"] == "illumination":
            rng = np.random.default_rng(config["seed"])
            scatter_x = scatter_x + rng.normal(0, 1.1, len(sample_indices))
        line_x = bins["Feature_Median"].to_numpy(dtype=float)

    axis.scatter(
        scatter_x,
        y[sample_indices],
        s=8,
        alpha=0.16,
        color="#4477AA",
        linewidths=0,
        rasterized=True,
        label="Test observations",
    )
    lower = bins["Cluster_Bootstrap_Mean_CI95_Lower"].to_numpy(dtype=float)
    upper = bins["Cluster_Bootstrap_Mean_CI95_Upper"].to_numpy(dtype=float)
    mean = bins["Mean_SHAP"].to_numpy(dtype=float)
    axis.errorbar(
        line_x,
        mean,
        yerr=np.vstack([mean - lower, upper - mean]),
        color="#CC3311",
        marker="o",
        markersize=4.2,
        linewidth=2.0,
        elinewidth=1.15,
        capsize=2.4,
        label="Binned mean ± 95% tree-bootstrap CI",
        zorder=4,
    )
    axis.axhline(0, color="#6B7280", linewidth=0.9, linestyle="--")
    axis.set_title(config["label"], fontsize=11, weight="bold", pad=8)
    axis.set_xlabel(config["x_label"])
    axis.set_ylabel("SHAP contribution to log annualized specific growth")
    axis.grid(axis="y", color="#E5E7EB", linewidth=0.6)
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(labelsize=8.5)


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Output directory already exists: {args.output}")
    tables = args.output / "tables"
    plots = args.output / "plots"
    tables.mkdir(parents=True)
    plots.mkdir()

    source = pd.read_csv(args.observations)
    unique_trees, tree_codes, weights = tree_cluster_weights(
        source["OID_"], args.bootstrap_repetitions, args.random_state + 61000
    )
    rng = np.random.default_rng(args.random_state + 62000)
    sample_indices = np.sort(
        rng.choice(len(source), size=min(3500, len(source)), replace=False)
    )

    all_bins = []
    all_statistics = []
    all_boot_means: dict[str, np.ndarray] = {}
    feature_frames: dict[str, pd.DataFrame] = {}

    for feature, config in FEATURES.items():
        frame = source[["OID_", f"{feature}__value", f"{feature}__shap"]].copy()
        frame.columns = ["OID_", "Feature_Value", "SHAP"]
        frame = frame.dropna().reset_index(drop=True)
        if len(frame) != len(source):
            raise AssertionError(f"Missing values found for {feature}")
        codes, labels = assign_bins(frame["Feature_Value"], config["kind"])
        frame["Bin_Code"] = codes
        bins, boot_means = bin_summary(
            frame, feature, labels, tree_codes, weights
        )
        all_bins.append(bins)
        all_boot_means[feature] = boot_means
        feature_frames[feature] = frame

        x = frame["Feature_Value"].to_numpy(dtype=float)
        y = frame["SHAP"].to_numpy(dtype=float)
        correlation = float(spearmanr(x, y).statistic)
        if config["kind"] == "binary":
            linear_r2 = np.nan
            spline_r2 = np.nan
            spline_gain = np.nan
        else:
            linear_r2, spline_r2, spline_gain = group_cv_nonlinearity(
                x, y, frame["OID_"].to_numpy()
            )
        peak = bins.loc[bins["Mean_SHAP"].idxmax()]
        trough = bins.loc[bins["Mean_SHAP"].idxmin()]
        all_statistics.append(
            {
                "Feature": feature,
                "Feature_Label": config["label"],
                "N_Observations": int(len(frame)),
                "N_Trees": int(frame["OID_"].nunique()),
                "Unique_Feature_Values": int(frame["Feature_Value"].nunique()),
                "Mean_Absolute_SHAP": float(np.mean(np.abs(y))),
                "SHAP_P05": float(np.quantile(y, 0.05)),
                "SHAP_P95": float(np.quantile(y, 0.95)),
                "Feature_SHAP_Spearman": correlation,
                "Linear_GroupCV_R2": linear_r2,
                "Spline_GroupCV_R2": spline_r2,
                "Spline_Minus_Linear_GroupCV_R2": spline_gain,
                "Binned_Mean_Sign_Changes": sign_changes(
                    bins["Mean_SHAP"].to_numpy(dtype=float)
                ),
                "Peak_Bin_Label": peak["Bin_Label"],
                "Peak_Feature_Median": float(peak["Feature_Median"]),
                "Peak_Mean_SHAP": float(peak["Mean_SHAP"]),
                "Trough_Bin_Label": trough["Bin_Label"],
                "Trough_Feature_Median": float(trough["Feature_Median"]),
                "Trough_Mean_SHAP": float(trough["Mean_SHAP"]),
            }
        )

    bins_table = pd.concat(all_bins, ignore_index=True)
    statistics = pd.DataFrame(all_statistics)

    contrasts = []
    for feature in ["avg_LST", "lightemiss"]:
        bins = bins_table[bins_table["Feature"].eq(feature)].reset_index(drop=True)
        means = all_boot_means[feature]
        peak_index = int(bins["Mean_SHAP"].idxmax() - bins.index.min())
        trough_index = int(bins["Mean_SHAP"].idxmin() - bins.index.min())
        draws = means[:, peak_index] - means[:, trough_index]
        contrasts.append(
            {
                "Feature": feature,
                "Contrast": "Peak bin minus trough bin",
                "First_Level": bins.loc[peak_index, "Bin_Label"],
                "Second_Level": bins.loc[trough_index, "Bin_Label"],
                "Estimate": float(
                    bins.loc[peak_index, "Mean_SHAP"]
                    - bins.loc[trough_index, "Mean_SHAP"]
                ),
                "Cluster_Bootstrap_CI95_Lower": float(np.quantile(draws, 0.025)),
                "Cluster_Bootstrap_CI95_Upper": float(np.quantile(draws, 0.975)),
            }
        )

    park_bins = bins_table[bins_table["Feature"].eq("type_Puisto")].reset_index(
        drop=True
    )
    park_draws = all_boot_means["type_Puisto"][:, 1] - all_boot_means[
        "type_Puisto"
    ][:, 0]
    contrasts.append(
        {
            "Feature": "type_Puisto",
            "Contrast": "Park minus street/non-park mean SHAP",
            "First_Level": "Park",
            "Second_Level": "Street/non-park",
            "Estimate": float(
                park_bins.loc[1, "Mean_SHAP"] - park_bins.loc[0, "Mean_SHAP"]
            ),
            "Cluster_Bootstrap_CI95_Lower": float(
                np.quantile(park_draws, 0.025)
            ),
            "Cluster_Bootstrap_CI95_Upper": float(
                np.quantile(park_draws, 0.975)
            ),
        }
    )
    contrasts = pd.DataFrame(contrasts)

    park_observations = feature_frames["type_Puisto"].copy()
    park_distribution = (
        park_observations.groupby("Bin_Code", observed=True)
        .agg(
            N_Observations=("SHAP", "size"),
            N_Trees=("OID_", "nunique"),
            Mean_SHAP=("SHAP", "mean"),
            Median_SHAP=("SHAP", "median"),
            Mean_Absolute_SHAP=("SHAP", lambda values: np.mean(np.abs(values))),
            Proportion_Positive_SHAP=("SHAP", lambda values: np.mean(values > 0)),
            Proportion_Negative_SHAP=("SHAP", lambda values: np.mean(values < 0)),
            SHAP_P05=("SHAP", lambda values: np.quantile(values, 0.05)),
            SHAP_P95=("SHAP", lambda values: np.quantile(values, 0.95)),
        )
        .reset_index()
    )
    park_distribution["Context"] = park_distribution["Bin_Code"].map(
        {0: "Street/non-park", 1: "Park"}
    )
    park_distribution = park_distribution[
        ["Context"] + [column for column in park_distribution if column not in {"Context", "Bin_Code"}]
    ]

    bins_table.to_csv(tables / "nonlinear_dependency_bins.csv", index=False)
    statistics.to_csv(tables / "nonlinear_dependency_statistics.csv", index=False)
    contrasts.to_csv(tables / "nonlinear_dependency_contrasts.csv", index=False)
    park_distribution.to_csv(tables / "park_context_shap_distribution.csv", index=False)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.labelcolor": "#20242A",
            "xtick.color": "#343A40",
            "ytick.color": "#343A40",
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(15.3, 4.9))
    for axis, feature in zip(axes, FEATURES):
        plot_panel(
            axis,
            feature_frames[feature],
            bins_table[bins_table["Feature"].eq(feature)].reset_index(drop=True),
            feature,
            sample_indices,
        )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.suptitle(
        "Nonlinear environmental SHAP dependence · pooled XGBoost independent test (n = 6,845)",
        fontsize=14,
        weight="bold",
        y=1.01,
    )
    fig.tight_layout(rect=[0, 0.07, 1, 0.96], w_pad=2.2)
    for extension in ["png", "pdf", "svg"]:
        fig.savefig(
            plots / f"Figure4b_nonlinear_SHAP_dependency.{extension}",
            dpi=350 if extension == "png" else None,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)

    for feature in FEATURES:
        fig, axis = plt.subplots(figsize=(6.2, 4.7))
        plot_panel(
            axis,
            feature_frames[feature],
            bins_table[bins_table["Feature"].eq(feature)].reset_index(drop=True),
            feature,
            sample_indices,
        )
        handles, labels = axis.get_legend_handles_labels()
        axis.legend(handles, labels, loc="best", frameon=False, fontsize=8)
        fig.tight_layout()
        for extension in ["png", "pdf", "svg"]:
            fig.savefig(
                plots / f"{feature}_SHAP_dependency.{extension}",
                dpi=350 if extension == "png" else None,
                bbox_inches="tight",
                facecolor="white",
            )
        plt.close(fig)

    metadata = {
        "source": str(args.observations),
        "model_scope": "Pooled period-controlled three-soil XGBoost",
        "sample": "Independent test",
        "observations": int(len(source)),
        "trees": int(len(unique_trees)),
        "features": list(FEATURES),
        "bootstrap": {
            "cluster": "OID_",
            "repetitions": args.bootstrap_repetitions,
            "random_state": args.random_state + 61000,
        },
        "continuous_dependency": (
            "Raw observation-level SHAP scatter with binned mean and 95% "
            "OID_-cluster bootstrap confidence intervals. LST uses deciles; "
            "illumination uses pre-specified groups that retain its discrete scale."
        ),
        "park_dependency": (
            "Binary context comparison with jittered observations and category "
            "means with 95% OID_-cluster bootstrap confidence intervals."
        ),
        "interpretation_warning": (
            "SHAP dependence describes fitted conditional associations and may "
            "reflect correlated predictors and interactions. It is not causal."
        ),
    }
    (args.output / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(statistics.to_string(index=False))
    print("\nContrasts")
    print(contrasts.to_string(index=False))
    print("\nPark distribution")
    print(park_distribution.to_string(index=False))


if __name__ == "__main__":
    main()
