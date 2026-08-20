#!/usr/bin/env python3
"""Reproducible relative-growth model comparison and SHAP workflow.

The workflow reproduces the documented manuscript methodology:

* tree-period response:
    g = (ln(C_end) - ln(C_start)) / years
    y = ln(g)
* tree-grouped 85/15 development/test split using OID_
* pooled one-hot species representation
* period-controlled primary analysis and no-period deployment sensitivity
* OLS, Random Forest, XGBoost, and MLP on identical observations/features
* log-SGR, annual percentage-point, and stock-aware kg C/tree/year metrics
* pooled XGBoost SHAP values, summarized by functional group and genus

The input dataset is read-only. The output directory must not already exist.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import sklearn
import xgboost
from PIL import Image, ImageDraw, ImageFont
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


RANDOM_STATE = 42
TREE_ID = "OID_"

SPECIES_MAP = {
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

ENVIRONMENT_FEATURES = [
    "avg_noise_day",
    "Density25",
    "Mono_Rate",
    "avg_svf",
    "avg_radiation",
    "avg_LST",
    "lightemiss",
    "type_Puisto",
]

FEATURE_LABELS = {
    "avg_noise_day": "Daytime noise",
    "Density25": "Tree density (25 m)",
    "Mono_Rate": "Monoculture rate",
    "avg_svf": "Sky-view factor",
    "avg_radiation": "Solar radiation",
    "avg_LST": "Land-surface temperature",
    "lightemiss": "Night illumination",
    "type_Puisto": "Park context",
}

GROUPS = {
    "Overall": None,
    "Conifer": {"General_Conifer", "Pinus"},
    "Broadleaf": {
        "General_Broadleaf",
        "Acer",
        "Alnus",
        "Betula",
        "Prunus",
        "Quercus",
        "Sorbus",
        "Tilia",
        "Ulmus",
    },
    "Acer": {"Acer"},
    "Alnus": {"Alnus"},
    "Betula": {"Betula"},
    "Pinus": {"Pinus"},
    "Prunus": {"Prunus"},
    "Quercus": {"Quercus"},
    "Sorbus": {"Sorbus"},
    "Tilia": {"Tilia"},
    "Ulmus": {"Ulmus"},
}

XGB_PARAMETERS = {
    "objective": "reg:squarederror",
    "n_estimators": 550,
    "learning_rate": 0.03,
    "max_depth": 6,
    "min_child_weight": 10,
    "subsample": 0.85,
    "colsample_bytree": 0.90,
    "reg_lambda": 2.0,
    "reg_alpha": 0.0,
    "tree_method": "hist",
    "n_jobs": -1,
    "random_state": RANDOM_STATE,
}

RF_PARAMETERS = {
    "n_estimators": 500,
    "max_depth": 18,
    "min_samples_leaf": 5,
    "max_features": 0.80,
    "n_jobs": -1,
    "random_state": RANDOM_STATE,
}

MLP_PARAMETERS = {
    "hidden_layer_sizes": (96, 48),
    "activation": "relu",
    "solver": "adam",
    "alpha": 0.01,
    "batch_size": 512,
    "learning_rate_init": 0.001,
    "max_iter": 500,
    "early_stopping": True,
    "validation_fraction": 0.15,
    "n_iter_no_change": 20,
    "random_state": RANDOM_STATE,
}


def build_long_data(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Construct the three-period longitudinal table used in the handoff."""
    data = raw[raw["type"].isin(["Katu", "Puisto"])].copy()
    data["Species_Name_Model"] = data["Species"].map(SPECIES_MAP)

    for column in [
        "noise17d",
        "noise22d",
        "lightemiss",
        "Density25",
        "Mono_Rate",
    ]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data["avg_noise_day"] = data[["noise17d", "noise22d"]].mean(axis=1)

    configuration = {
        "15_17": ("15", "17", 2, "ra15_17", "LST_1516"),
        "17_21": ("17", "21", 4, "ra17_21", "LST_1720"),
        "21_23": ("21", "23", 2, "ra21_23", "LST_2122"),
    }

    frames = []
    for period, (start, end, years, radiation, lst) in configuration.items():
        frame = data.copy()
        frame["Period"] = period
        frame["Years"] = years
        frame["Growth"] = pd.to_numeric(frame[f"Ann_{period}"], errors="coerce")
        frame["Height"] = pd.to_numeric(frame[f"H{start}"], errors="coerce")
        frame["Initial_Carbon"] = pd.to_numeric(
            frame[f"CS_{start}"], errors="coerce"
        )
        frame["End_Carbon"] = pd.to_numeric(
            frame[f"CS_{end}"], errors="coerce"
        )

        frame = frame[
            frame["Species_Name_Model"].notna()
            & (frame["Growth"] > 0)
            & (frame["Height"] > 0)
            & (frame["Initial_Carbon"] > 0)
            & (frame["End_Carbon"] > frame["Initial_Carbon"])
        ].copy()

        frame["Log_Annual_Carbon_Growth"] = np.log(frame["Growth"])
        frame["Log_Height"] = np.log(frame["Height"])
        frame["Specific_Growth_Rate"] = (
            np.log(frame["End_Carbon"]) - np.log(frame["Initial_Carbon"])
        ) / years
        frame["Log_Specific_Growth_Rate"] = np.log(
            frame["Specific_Growth_Rate"].clip(lower=1e-9)
        )
        frame["avg_svf"] = frame[[f"svf{start}", f"svf{end}"]].mean(axis=1)
        frame["avg_radiation"] = pd.to_numeric(
            frame[radiation], errors="coerce"
        )
        frame["avg_LST"] = pd.to_numeric(frame[lst], errors="coerce")
        frame["type_Puisto"] = (frame["type"] == "Puisto").astype(np.float32)
        frames.append(frame)

    before_trim = pd.concat(frames, ignore_index=True)
    lower, upper = before_trim["Log_Annual_Carbon_Growth"].quantile([0.05, 0.95])
    long_data = before_trim[
        before_trim["Log_Annual_Carbon_Growth"].between(lower, upper)
    ].copy()
    long_data.reset_index(drop=True, inplace=True)
    metadata = {
        "valid_tree_period_rows_before_legacy_trim": int(len(before_trim)),
        "rows_after_legacy_trim": int(len(long_data)),
        "unique_trees_after_legacy_trim": int(long_data[TREE_ID].nunique()),
        "legacy_log_annual_growth_p05": float(lower),
        "legacy_log_annual_growth_p95": float(upper),
        "filter_note": (
            "For direct comparability with the handed-off analysis, the 5th-95th "
            "percentile filter was calculated on positive annual absolute-growth "
            "rows before the grouped split. The split itself is leakage-free by OID_."
        ),
    }
    return long_data, metadata


def add_split_and_dummies(data: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    tree_meta = data.groupby(TREE_ID, as_index=False).agg(
        Species_Name_Model=("Species_Name_Model", "first")
    )
    development_ids, test_ids = train_test_split(
        tree_meta[TREE_ID],
        test_size=0.15,
        random_state=RANDOM_STATE,
        stratify=tree_meta["Species_Name_Model"],
    )
    split_map = pd.concat(
        [
            pd.DataFrame({TREE_ID: development_ids, "Split": "Development"}),
            pd.DataFrame({TREE_ID: test_ids, "Split": "Test"}),
        ],
        ignore_index=True,
    )
    data = data.merge(split_map, on=TREE_ID, how="inner", validate="many_to_one")

    species_dummies = pd.get_dummies(
        data["Species_Name_Model"],
        prefix="Species",
        dtype=np.float32,
    )
    period_dummies = pd.get_dummies(
        data["Period"],
        prefix="Period",
        dtype=np.float32,
    )
    data = pd.concat(
        [
            data.reset_index(drop=True),
            species_dummies.reset_index(drop=True),
            period_dummies.reset_index(drop=True),
        ],
        axis=1,
    )
    return data, {
        "split_map": split_map,
        "species_columns": species_dummies.columns.tolist(),
        "period_columns": period_dummies.columns.tolist(),
    }


def group_mask(data: pd.DataFrame, group: str) -> np.ndarray:
    species = GROUPS[group]
    if species is None:
        return np.ones(len(data), dtype=bool)
    return data["Species_Name_Model"].isin(species).to_numpy()


def inverse_percentage(y: np.ndarray) -> np.ndarray:
    """Convert log-SGR to an annual compound percentage rate."""
    sgr = np.exp(np.clip(np.asarray(y, dtype=float), -30.0, 5.0))
    return 100.0 * np.expm1(sgr)


def inverse_kg(
    y: np.ndarray,
    initial_carbon: np.ndarray,
    years: np.ndarray,
) -> np.ndarray:
    """Convert log-SGR to interval-consistent mean kg C/tree/year."""
    sgr = np.exp(np.clip(np.asarray(y, dtype=float), -30.0, 5.0))
    carbon_ratio = np.exp(np.clip(sgr * years, -30.0, 30.0))
    return initial_carbon * (carbon_ratio - 1.0) / years


def metric_record(
    actual: np.ndarray,
    predicted: np.ndarray,
    initial_carbon: np.ndarray,
    years: np.ndarray,
) -> dict:
    actual_pct = inverse_percentage(actual)
    predicted_pct = inverse_percentage(predicted)
    actual_kg = inverse_kg(actual, initial_carbon, years)
    predicted_kg = inverse_kg(predicted, initial_carbon, years)
    return {
        "R2_LogSGR": r2_score(actual, predicted),
        "RMSE_LogSGR": mean_squared_error(actual, predicted) ** 0.5,
        "MAE_LogSGR": mean_absolute_error(actual, predicted),
        "RMSE_Annual_Percentage_Points": mean_squared_error(
            actual_pct, predicted_pct
        )
        ** 0.5,
        "MAE_Annual_Percentage_Points": mean_absolute_error(
            actual_pct, predicted_pct
        ),
        "Bias_Annual_Percentage_Points": float(
            np.mean(predicted_pct - actual_pct)
        ),
        "RMSE_kg_C_per_tree_per_year": mean_squared_error(
            actual_kg, predicted_kg
        )
        ** 0.5,
        "MAE_kg_C_per_tree_per_year": mean_absolute_error(
            actual_kg, predicted_kg
        ),
        "Bias_kg_C_per_tree_per_year": float(np.mean(predicted_kg - actual_kg)),
    }


def prepare_matrices(
    data: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    development = data["Split"].eq("Development")
    medians = data.loc[development, feature_columns].median(numeric_only=True)
    x = data[feature_columns].fillna(medians).astype(np.float32)
    return x.loc[development].copy(), x.loc[~development].copy(), medians


def fit_models(
    data: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[dict, dict, pd.Series, dict]:
    development = data["Split"].eq("Development")
    test = ~development
    target = data["Log_Specific_Growth_Rate"].astype(np.float32)
    x_development, x_test, medians = prepare_matrices(data, feature_columns)
    y_development = target.loc[development]

    scaler = StandardScaler()
    x_development_scaled = scaler.fit_transform(x_development)
    x_test_scaled = scaler.transform(x_test)

    fitted = {}
    predictions = {}
    timings = {}

    model_specs = [
        ("OLS", LinearRegression(), x_development_scaled, x_test_scaled),
        (
            "RF",
            RandomForestRegressor(**RF_PARAMETERS),
            x_development,
            x_test,
        ),
        ("XGB", XGBRegressor(**XGB_PARAMETERS), x_development, x_test),
        (
            "MLP",
            MLPRegressor(**MLP_PARAMETERS),
            x_development_scaled,
            x_test_scaled,
        ),
    ]
    for name, model, fit_x, test_x in model_specs:
        started = time.perf_counter()
        model.fit(fit_x, y_development)
        timings[name] = time.perf_counter() - started
        fitted[name] = model
        predictions[name] = {
            "Development": model.predict(fit_x).astype(np.float32),
            "Test": model.predict(test_x).astype(np.float32),
        }
    return fitted, predictions, medians, {
        "scaler": scaler,
        "timings_seconds": timings,
        "x_development": x_development,
        "x_test": x_test,
    }


def evaluate_models(
    data: pd.DataFrame,
    predictions: dict,
    specification: str,
) -> pd.DataFrame:
    rows = []
    for split_name in ["Development", "Test"]:
        split_data = data[data["Split"].eq(split_name)].copy()
        actual = split_data["Log_Specific_Growth_Rate"].to_numpy()
        initial_carbon = split_data["Initial_Carbon"].to_numpy()
        years = split_data["Years"].to_numpy()
        for group in GROUPS:
            mask = group_mask(split_data, group)
            for model_name, model_predictions in predictions.items():
                record = metric_record(
                    actual[mask],
                    model_predictions[split_name][mask],
                    initial_carbon[mask],
                    years[mask],
                )
                rows.append(
                    {
                        "Specification": specification,
                        "Evaluation_Strategy": (
                            "One pooled model; group rows are within-group "
                            "evaluations of the same pooled predictions"
                        ),
                        "Split": split_name,
                        "Group": group,
                        "Model": model_name,
                        "N_Rows": int(mask.sum()),
                        "N_Trees": int(
                            split_data.loc[mask, TREE_ID].nunique()
                        ),
                        **record,
                    }
                )
    return pd.DataFrame(rows)


def compact_performance_table(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    primary = metrics[metrics["Specification"].eq("Period-controlled")]
    for group in GROUPS:
        row = {"Group": group}
        for model in ["OLS", "RF", "XGB", "MLP"]:
            development = primary[
                primary["Split"].eq("Development")
                & primary["Group"].eq(group)
                & primary["Model"].eq(model)
            ].iloc[0]
            test = primary[
                primary["Split"].eq("Test")
                & primary["Group"].eq(group)
                & primary["Model"].eq(model)
            ].iloc[0]
            row[f"{model}_R2_Development_Test"] = (
                f"{development['R2_LogSGR']:.3f} / {test['R2_LogSGR']:.3f}"
            )
            row[f"{model}_MAE_pctpt"] = test[
                "MAE_Annual_Percentage_Points"
            ]
        rows.append(row)
    return pd.DataFrame(rows)


def save_performance_table_plot(compact: pd.DataFrame, output: Path) -> None:
    display = compact.copy()
    columns = ["Group"]
    for model in ["OLS", "RF", "XGB", "MLP"]:
        display[f"{model}\nR2 (Dev/Test)"] = display[
            f"{model}_R2_Development_Test"
        ]
        display[f"{model}\nMAE (pp/yr)"] = display[f"{model}_MAE_pctpt"].map(
            lambda value: f"{value:.2f}"
        )
        columns.extend([f"{model}\nR2 (Dev/Test)", f"{model}\nMAE (pp/yr)"])
    display = display[columns]

    fig, ax = plt.subplots(figsize=(15.5, 7.5))
    ax.axis("off")
    table = ax.table(
        cellText=display.values,
        colLabels=display.columns,
        loc="center",
        cellLoc="center",
        colLoc="center",
        colWidths=[0.11] + [0.111] * 8,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 1.55)
    model_colors = {
        1: "#E8F0FE",
        2: "#E8F0FE",
        3: "#E8F5E9",
        4: "#E8F5E9",
        5: "#FFF3E0",
        6: "#FFF3E0",
        7: "#F3E5F5",
        8: "#F3E5F5",
    }
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor("#B6BEC8")
        cell.set_linewidth(0.55)
        if row == 0:
            cell.set_facecolor("#243447")
            cell.set_text_props(color="white", weight="bold")
        elif column == 0:
            cell.set_facecolor("#EEF1F4")
            cell.set_text_props(weight="bold", ha="left")
        else:
            cell.set_facecolor(model_colors[column])
    ax.set_title(
        "Relative-growth model comparison (tree-grouped independent test)",
        fontsize=15,
        weight="bold",
        pad=18,
    )
    fig.text(
        0.5,
        0.045,
        "Target: log annualized specific carbon growth. "
        "MAE is back-transformed to annual percentage points; it is not kg C.",
        ha="center",
        fontsize=9,
        color="#354052",
    )
    fig.text(
        0.5,
        0.022,
        "Each algorithm was fitted once to the pooled data; group rows are "
        "within-group evaluations of those pooled predictions.",
        ha="center",
        fontsize=8.5,
        color="#354052",
    )
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def cluster_bootstrap_r2(
    test_data: pd.DataFrame,
    predictions: dict,
    repetitions: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(RANDOM_STATE + 1000)
    unique_trees = test_data[TREE_ID].unique()
    tree_rows = {
        tree: np.flatnonzero(test_data[TREE_ID].to_numpy() == tree)
        for tree in unique_trees
    }
    actual = test_data["Log_Specific_Growth_Rate"].to_numpy()
    records = []
    paired = []
    for iteration in range(repetitions):
        sampled = rng.choice(unique_trees, size=len(unique_trees), replace=True)
        index = np.concatenate([tree_rows[tree] for tree in sampled])
        values = {}
        for model in ["OLS", "RF", "XGB", "MLP"]:
            score = r2_score(actual[index], predictions[model]["Test"][index])
            values[model] = score
            records.append(
                {
                    "Iteration": iteration + 1,
                    "Model": model,
                    "Test_R2": score,
                }
            )
        paired.append(
            {
                "Iteration": iteration + 1,
                "XGB_minus_RF_Test_R2": values["XGB"] - values["RF"],
            }
        )
    return pd.DataFrame(records), pd.DataFrame(paired)


def cluster_bootstrap_environmental_attribution(
    test_data: pd.DataFrame,
    baseline_prediction: np.ndarray,
    full_prediction: np.ndarray,
    repetitions: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE + 1500)
    unique_trees = test_data[TREE_ID].unique()
    tree_rows = {
        tree: np.flatnonzero(test_data[TREE_ID].to_numpy() == tree)
        for tree in unique_trees
    }
    actual = test_data["Log_Specific_Growth_Rate"].to_numpy()
    rows = []
    for iteration in range(repetitions):
        sampled = rng.choice(unique_trees, size=len(unique_trees), replace=True)
        index = np.concatenate([tree_rows[tree] for tree in sampled])
        baseline_r2 = r2_score(actual[index], baseline_prediction[index])
        full_r2 = r2_score(actual[index], full_prediction[index])
        baseline_sse = np.sum(
            (actual[index] - baseline_prediction[index]) ** 2
        )
        full_sse = np.sum((actual[index] - full_prediction[index]) ** 2)
        rows.append(
            {
                "Iteration": iteration + 1,
                "Baseline_Test_R2": baseline_r2,
                "Full_Test_R2": full_r2,
                "Incremental_Delta_R2": full_r2 - baseline_r2,
                "Environmental_Partial_R2": 1.0 - full_sse / baseline_sse,
            }
        )
    return pd.DataFrame(rows)


def bootstrap_summary_plot(
    bootstrap: pd.DataFrame,
    point_estimates: dict,
    output: Path,
) -> pd.DataFrame:
    order = ["OLS", "RF", "XGB", "MLP"]
    rows = []
    for model in order:
        values = bootstrap.loc[bootstrap["Model"].eq(model), "Test_R2"]
        rows.append(
            {
                "Model": model,
                "Test_R2": point_estimates[model],
                "CI95_Lower": values.quantile(0.025),
                "CI95_Upper": values.quantile(0.975),
            }
        )
    summary = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    x = np.arange(len(summary))
    y = summary["Test_R2"].to_numpy()
    lower = y - summary["CI95_Lower"].to_numpy()
    upper = summary["CI95_Upper"].to_numpy() - y
    ax.errorbar(
        x,
        y,
        yerr=np.vstack([lower, upper]),
        fmt="o",
        color="#1F5A7A",
        ecolor="#5D7990",
        capsize=5,
        markersize=7,
    )
    ax.set_xticks(x, summary["Model"])
    ax.set_ylabel("Independent-test R2 (log-SGR)")
    ax.set_title("Tree-cluster bootstrap uncertainty")
    ax.grid(axis="y", color="#DCE2E8", linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return summary


def environmental_attribution(
    data: pd.DataFrame,
    full_model: XGBRegressor,
    full_test_matrix: pd.DataFrame,
    full_predictions: np.ndarray,
    baseline_columns: list[str],
    permutation_repetitions: int,
) -> tuple[pd.DataFrame, pd.DataFrame, XGBRegressor, np.ndarray]:
    development = data["Split"].eq("Development")
    test = ~development
    target = data["Log_Specific_Growth_Rate"].astype(np.float32)
    baseline_development, baseline_test, _ = prepare_matrices(
        data, baseline_columns
    )
    baseline_model = XGBRegressor(**XGB_PARAMETERS)
    baseline_model.fit(baseline_development, target.loc[development])
    baseline_prediction = baseline_model.predict(baseline_test)
    y_test = target.loc[test].to_numpy()
    baseline_r2 = r2_score(y_test, baseline_prediction)
    full_r2 = r2_score(y_test, full_predictions)
    delta_r2 = full_r2 - baseline_r2
    partial_r2 = 1.0 - np.sum((y_test - full_predictions) ** 2) / np.sum(
        (y_test - baseline_prediction) ** 2
    )

    rng = np.random.default_rng(RANDOM_STATE + 2000)
    test_data = data.loc[test].reset_index(drop=True)
    base_matrix = full_test_matrix.reset_index(drop=True)
    strata = test_data.groupby(["Species_Name_Model", "Period"]).indices
    permutation_rows = []
    for repetition in range(permutation_repetitions):
        permuted = base_matrix.copy()
        for row_indices in strata.values():
            shuffled = rng.permutation(row_indices)
            permuted.loc[row_indices, ENVIRONMENT_FEATURES] = base_matrix.loc[
                shuffled, ENVIRONMENT_FEATURES
            ].to_numpy()
        prediction = full_model.predict(permuted)
        permuted_r2 = r2_score(y_test, prediction)
        permutation_rows.append(
            {
                "Repetition": repetition + 1,
                "Permuted_Test_R2": permuted_r2,
                "Environmental_Block_R2_Loss": full_r2 - permuted_r2,
            }
        )
    permutation = pd.DataFrame(permutation_rows)
    summary = pd.DataFrame(
        [
            {
                "Baseline_Test_R2": baseline_r2,
                "Full_Test_R2": full_r2,
                "Incremental_Delta_R2": delta_r2,
                "Environmental_Partial_R2": partial_r2,
                "Permutation_Mean_R2_Loss": permutation[
                    "Environmental_Block_R2_Loss"
                ].mean(),
                "Permutation_CI95_Lower": permutation[
                    "Environmental_Block_R2_Loss"
                ].quantile(0.025),
                "Permutation_CI95_Upper": permutation[
                    "Environmental_Block_R2_Loss"
                ].quantile(0.975),
                "Permutation_Repetitions": permutation_repetitions,
            }
        ]
    )
    return summary, permutation, baseline_model, baseline_prediction


def compute_shap(
    model: XGBRegressor,
    x_test: pd.DataFrame,
) -> np.ndarray:
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(x_test)
    return np.asarray(values, dtype=np.float32)


def shap_group_statistics(
    test_data: pd.DataFrame,
    x_test: pd.DataFrame,
    shap_values: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    environment_indices = [x_test.columns.get_loc(f) for f in ENVIRONMENT_FEATURES]
    environment_shap = shap_values[:, environment_indices]
    statistics = []
    dependence = []

    observation = test_data[
        [TREE_ID, "Period", "Species_Name_Model", "Years", "Initial_Carbon"]
    ].reset_index(drop=True)
    for feature_index, feature in enumerate(ENVIRONMENT_FEATURES):
        observation[f"{feature}__value"] = x_test[feature].to_numpy()
        observation[f"{feature}__shap"] = environment_shap[:, feature_index]

    for group in GROUPS:
        mask = group_mask(test_data, group)
        for feature_index, feature in enumerate(ENVIRONMENT_FEATURES):
            x = x_test.loc[mask, feature].to_numpy(dtype=float)
            values = environment_shap[mask, feature_index].astype(float)
            if np.unique(x[~np.isnan(x)]).size <= 2:
                contrast = np.nanmean(values[x == 1]) - np.nanmean(values[x == 0])
                correlation = np.nan
            else:
                q25, q75 = np.nanquantile(x, [0.25, 0.75])
                contrast = np.nanmean(values[x >= q75]) - np.nanmean(
                    values[x <= q25]
                )
                correlation = spearmanr(x, values, nan_policy="omit").statistic
            p05, p95 = np.nanquantile(values, [0.05, 0.95])
            statistics.append(
                {
                    "Group": group,
                    "Feature": feature,
                    "Feature_Label": FEATURE_LABELS[feature],
                    "N_SHAP": int(len(values)),
                    "SHAP_Min": np.nanmin(values),
                    "SHAP_Max": np.nanmax(values),
                    "SHAP_Range": np.nanmax(values) - np.nanmin(values),
                    "SHAP_P05": p05,
                    "SHAP_P95": p95,
                    "SHAP_Robust_Range_P05_P95": p95 - p05,
                    "Mean_Absolute_SHAP": np.nanmean(np.abs(values)),
                    "High_minus_Low_SHAP": contrast,
                    "SHAP_Feature_Spearman": correlation,
                }
            )

            if np.unique(x[~np.isnan(x)]).size <= 2:
                bins = pd.Series(x).fillna(-1)
            else:
                try:
                    bins = pd.qcut(x, q=10, duplicates="drop")
                except ValueError:
                    bins = pd.cut(x, bins=10, duplicates="drop")
            binned = pd.DataFrame({"Feature_Value": x, "SHAP": values, "Bin": bins})
            for decile, (_, subset) in enumerate(
                binned.groupby("Bin", observed=True), start=1
            ):
                dependence.append(
                    {
                        "Group": group,
                        "Feature": feature,
                        "Feature_Label": FEATURE_LABELS[feature],
                        "Decile": decile,
                        "Feature_Median": subset["Feature_Value"].median(),
                        "Mean_SHAP": subset["SHAP"].mean(),
                        "N": int(len(subset)),
                    }
                )
    return (
        pd.DataFrame(statistics),
        pd.DataFrame(dependence),
        observation,
    )


def save_beeswarm(
    x_environment: pd.DataFrame,
    shap_environment: np.ndarray,
    title: str,
    output: Path,
    x_limits: tuple[float, float] | None = None,
) -> None:
    plt.figure(figsize=(8.2, 5.6))
    shap.summary_plot(
        shap_environment,
        x_environment,
        feature_names=[FEATURE_LABELS[f] for f in ENVIRONMENT_FEATURES],
        show=False,
        max_display=len(ENVIRONMENT_FEATURES),
        plot_size=None,
    )
    axis = plt.gca()
    if x_limits:
        axis.set_xlim(*x_limits)
    axis.set_xlabel("SHAP contribution to log annualized specific growth")
    axis.set_title(title, fontsize=12, weight="bold", pad=10)
    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()


def save_dependency_grid(
    x_environment: pd.DataFrame,
    shap_environment: np.ndarray,
    dependence: pd.DataFrame,
    output: Path,
) -> None:
    rng = np.random.default_rng(RANDOM_STATE + 3000)
    sample_size = min(3000, len(x_environment))
    sample = np.sort(
        rng.choice(len(x_environment), size=sample_size, replace=False)
    )
    fig, axes = plt.subplots(2, 4, figsize=(14.5, 6.8))
    for index, (feature, axis) in enumerate(
        zip(ENVIRONMENT_FEATURES, axes.flat)
    ):
        x = x_environment[feature].to_numpy(dtype=float)
        y = shap_environment[:, index]
        if feature == "type_Puisto":
            jitter = rng.normal(0, 0.025, size=sample_size)
            scatter_x = x[sample] + jitter
            axis.set_xticks([0, 1], ["Street", "Park"])
        else:
            scatter_x = x[sample]
        axis.scatter(
            scatter_x,
            y[sample],
            s=7,
            alpha=0.18,
            color="#587A91",
            linewidths=0,
        )
        line = dependence[
            dependence["Group"].eq("Overall")
            & dependence["Feature"].eq(feature)
        ].sort_values("Feature_Median")
        axis.plot(
            line["Feature_Median"],
            line["Mean_SHAP"],
            color="#D04A35",
            linewidth=2.1,
            marker="o",
            markersize=3,
        )
        axis.axhline(0, color="#7A7A7A", linewidth=0.8, linestyle="--")
        axis.set_title(FEATURE_LABELS[feature], fontsize=10, weight="bold")
        axis.set_xlabel("")
        axis.set_ylabel("SHAP" if index % 4 == 0 else "")
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(color="#E5E8EB", linewidth=0.45, alpha=0.8)
    fig.suptitle(
        "Environmental dependence patterns (decile mean in red)",
        fontsize=14,
        weight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_dependency_heatmap(dependence: pd.DataFrame, output: Path) -> None:
    overall = dependence[dependence["Group"].eq("Overall")].copy()
    pivot = overall.pivot(
        index="Feature",
        columns="Decile",
        values="Mean_SHAP",
    ).reindex(ENVIRONMENT_FEATURES)
    values = pivot.to_numpy(dtype=float)
    limit = float(np.nanmax(np.abs(values)))
    fig, ax = plt.subplots(figsize=(8.4, 5.6))
    image = ax.imshow(
        values,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
        interpolation="nearest",
    )
    ax.set_yticks(
        np.arange(len(ENVIRONMENT_FEATURES)),
        [FEATURE_LABELS[feature] for feature in ENVIRONMENT_FEATURES],
    )
    ax.set_xticks(
        np.arange(values.shape[1]),
        [f"D{index}" for index in range(1, values.shape[1] + 1)],
    )
    ax.set_xlabel("Feature-value decile (low to high)")
    ax.set_title(
        "Environmental dependence map",
        fontsize=12,
        weight="bold",
        pad=10,
    )
    colorbar = fig.colorbar(image, ax=ax, pad=0.025)
    colorbar.set_label("Mean SHAP contribution to log-SGR")
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            if np.isfinite(values[row, column]):
                color = (
                    "white"
                    if abs(values[row, column]) > 0.60 * limit
                    else "#1F2933"
                )
                ax.text(
                    column,
                    row,
                    f"{values[row, column]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=6.6,
                    color=color,
                )
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def combine_images(
    left_path: Path,
    right_path: Path,
    output: Path,
    title: str,
) -> None:
    left = Image.open(left_path).convert("RGB")
    right = Image.open(right_path).convert("RGB")
    target_height = max(left.height, right.height)
    left = left.resize(
        (round(left.width * target_height / left.height), target_height),
        Image.Resampling.LANCZOS,
    )
    right = right.resize(
        (round(right.width * target_height / right.height), target_height),
        Image.Resampling.LANCZOS,
    )
    title_height = 100
    canvas = Image.new(
        "RGB",
        (left.width + right.width + 30, target_height + title_height),
        "white",
    )
    canvas.paste(left, (0, title_height))
    canvas.paste(right, (left.width + 30, title_height))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arialbd.ttf", 42)
    except OSError:
        font = ImageFont.load_default()
    bounds = draw.textbbox((0, 0), title, font=font)
    x = (canvas.width - (bounds[2] - bounds[0])) // 2
    draw.text((x, 24), title, fill="#1F2D3D", font=font)
    canvas.save(output, quality=95, dpi=(300, 300))


def species_beeswarm_plots(
    test_data: pd.DataFrame,
    x_environment: pd.DataFrame,
    shap_environment: np.ndarray,
    output_dir: Path,
) -> list[Path]:
    global_limit = float(
        np.nanquantile(np.abs(shap_environment), 0.999)
    )
    x_limits = (-global_limit, global_limit)
    outputs = []
    for order, group in enumerate(GROUPS, start=1):
        mask = group_mask(test_data, group)
        path = output_dir / f"{order:02d}_{group.lower()}_beeswarm.png"
        save_beeswarm(
            x_environment.loc[mask].reset_index(drop=True),
            shap_environment[mask],
            f"{group} (n = {int(mask.sum()):,})",
            path,
            x_limits=x_limits,
        )
        outputs.append(path)
    return outputs


def contact_sheet(
    images: list[Path],
    output: Path,
    title: str,
    columns: int = 3,
) -> None:
    opened = [Image.open(path).convert("RGB") for path in images]
    width = 1450
    resized = []
    for image in opened:
        resized.append(
            image.resize(
                (width, round(image.height * width / image.width)),
                Image.Resampling.LANCZOS,
            )
        )
    cell_height = max(image.height for image in resized)
    rows = int(np.ceil(len(resized) / columns))
    title_height = 120
    canvas = Image.new(
        "RGB",
        (columns * width, rows * cell_height + title_height),
        "white",
    )
    for index, image in enumerate(resized):
        x = (index % columns) * width
        y = title_height + (index // columns) * cell_height
        canvas.paste(image, (x, y))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arialbd.ttf", 48)
    except OSError:
        font = ImageFont.load_default()
    bounds = draw.textbbox((0, 0), title, font=font)
    x = (canvas.width - (bounds[2] - bounds[0])) // 2
    draw.text((x, 28), title, fill="#1F2D3D", font=font)
    canvas.save(output, quality=94, dpi=(300, 300))


def shap_stability(
    period_stats: pd.DataFrame,
    no_period_stats: pd.DataFrame,
) -> pd.DataFrame:
    period = period_stats[period_stats["Group"].eq("Overall")][
        [
            "Feature",
            "Mean_Absolute_SHAP",
            "High_minus_Low_SHAP",
            "SHAP_Feature_Spearman",
        ]
    ].rename(
        columns={
            "Mean_Absolute_SHAP": "Period_Mean_Absolute_SHAP",
            "High_minus_Low_SHAP": "Period_High_minus_Low_SHAP",
            "SHAP_Feature_Spearman": "Period_SHAP_Feature_Spearman",
        }
    )
    no_period = no_period_stats[no_period_stats["Group"].eq("Overall")][
        [
            "Feature",
            "Mean_Absolute_SHAP",
            "High_minus_Low_SHAP",
            "SHAP_Feature_Spearman",
        ]
    ].rename(
        columns={
            "Mean_Absolute_SHAP": "NoPeriod_Mean_Absolute_SHAP",
            "High_minus_Low_SHAP": "NoPeriod_High_minus_Low_SHAP",
            "SHAP_Feature_Spearman": "NoPeriod_SHAP_Feature_Spearman",
        }
    )
    result = period.merge(no_period, on="Feature", validate="one_to_one")
    result["MeanAbsSHAP_Rank_Period"] = result[
        "Period_Mean_Absolute_SHAP"
    ].rank(ascending=False, method="min")
    result["MeanAbsSHAP_Rank_NoPeriod"] = result[
        "NoPeriod_Mean_Absolute_SHAP"
    ].rank(ascending=False, method="min")
    return result.sort_values("MeanAbsSHAP_Rank_Period")


def write_run_log(
    output: Path,
    metadata: dict,
    attribution: pd.DataFrame,
    metrics: pd.DataFrame,
    timings: dict,
    bootstrap_pairs: pd.DataFrame,
) -> None:
    overall = metrics[
        metrics["Specification"].eq("Period-controlled")
        & metrics["Split"].eq("Test")
        & metrics["Group"].eq("Overall")
    ].set_index("Model")
    attribution_row = attribution.iloc[0]
    paired_ci = bootstrap_pairs["XGB_minus_RF_Test_R2"].quantile(
        [0.025, 0.975]
    )
    text = f"""# Relative-growth pipeline v2 run log

## Data and target

- Input rows: {metadata['raw_rows']:,}
- Longitudinal model rows: {metadata['rows_after_legacy_trim']:,}
- Unique modelled trees: {metadata['unique_trees_after_legacy_trim']:,}
- Development rows: {metadata['development_rows']:,}
- Test rows: {metadata['test_rows']:,}
- Split identifier: `OID_` (all repeated periods from one tree remain together)
- Target: `ln(([ln(C_end) - ln(C_start)] / years))`
- Primary species representation: pooled one-hot encoding
- Primary scientific specification: monitoring-period controlled
- Deployment sensitivity: categorical monitoring period removed

## Overall independent-test performance

| Model | Test R2 | RMSE log-SGR | MAE percentage points | RMSE kg C/tree/year | MAE kg C/tree/year |
|---|---:|---:|---:|---:|---:|
"""
    for model in ["OLS", "RF", "XGB", "MLP"]:
        row = overall.loc[model]
        text += (
            f"| {model} | {row['R2_LogSGR']:.3f} | "
            f"{row['RMSE_LogSGR']:.3f} | "
            f"{row['MAE_Annual_Percentage_Points']:.2f} | "
            f"{row['RMSE_kg_C_per_tree_per_year']:.2f} | "
            f"{row['MAE_kg_C_per_tree_per_year']:.2f} |\n"
        )
    text += f"""
## Environmental attribution

- Biological-temporal baseline test R2: {attribution_row['Baseline_Test_R2']:.3f}
- Full period-controlled XGBoost test R2: {attribution_row['Full_Test_R2']:.3f}
- Incremental environmental delta R2: {attribution_row['Incremental_Delta_R2']:.3f}
- Environmental partial R2: {attribution_row['Environmental_Partial_R2']:.3f}
- Environmental partial R2 95% tree-bootstrap interval: {attribution_row['Environmental_Partial_R2_CI95_Lower']:.3f} to {attribution_row['Environmental_Partial_R2_CI95_Upper']:.3f}
- Environmental-block permutation mean R2 loss: {attribution_row['Permutation_Mean_R2_Loss']:.3f}
- Paired bootstrap XGBoost-minus-RF R2 95% interval: {paired_ci.loc[0.025]:.3f} to {paired_ci.loc[0.975]:.3f}

## Reproducibility notes

- The 5th-95th percentile absolute-growth filter is retained from the handed-off
  analysis for direct numerical comparability and is calculated before splitting.
- All subsequent model fitting, imputation, and scaling use development data only.
- Percentage-point error is computed after `y -> g -> 100 * (exp(g) - 1)`.
- kg C/tree/year is interval-consistent:
  `C_start * (exp(years * g) - 1) / years`.
- SHAP values are associations on the log-SGR target scale, not causal effects.
- Full min-max SHAP ranges are sensitive to rare observations; P05-P95 ranges are
  the main robust summaries.
- Models were fitted to the pooled data. Functional-group and genus rows in the
  comparison table are within-group evaluations of those same pooled predictions.

## Runtime

```json
{json.dumps(timings, indent=2)}
```
"""
    (output / "RUN_LOG.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bootstrap-repetitions", type=int, default=1500)
    parser.add_argument("--permutation-repetitions", type=int, default=300)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(
            f"Output directory already exists; choose a versioned path: {args.output}"
        )
    tables = args.output / "tables"
    plots = args.output / "plots"
    species_plots = plots / "species_beeswarm"
    models = args.output / "models"
    for directory in [tables, plots, species_plots, models]:
        directory.mkdir(parents=True, exist_ok=False)

    started = time.perf_counter()
    raw = pd.read_csv(args.input)
    data, metadata = build_long_data(raw)
    data, encoding = add_split_and_dummies(data)
    encoding["split_map"].to_csv(tables / "tree_split.csv", index=False)

    species_columns = encoding["species_columns"]
    period_columns = encoding["period_columns"]
    period_features = (
        ["Log_Height"]
        + species_columns
        + period_columns
        + ENVIRONMENT_FEATURES
    )
    no_period_features = ["Log_Height"] + species_columns + ENVIRONMENT_FEATURES

    all_metrics = []
    fitted_models = {}
    prediction_sets = {}
    run_context = {}
    medians = {}
    for specification, columns in [
        ("Period-controlled", period_features),
        ("No-period", no_period_features),
    ]:
        fitted, predictions, feature_medians, context = fit_models(data, columns)
        fitted_models[specification] = fitted
        prediction_sets[specification] = predictions
        run_context[specification] = context
        medians[specification] = feature_medians
        all_metrics.append(evaluate_models(data, predictions, specification))

    metrics = pd.concat(all_metrics, ignore_index=True)
    metrics.to_csv(tables / "model_performance_all_groups.csv", index=False)
    compact = compact_performance_table(metrics)
    compact.to_csv(tables / "manuscript_table4_relative_growth.csv", index=False)
    save_performance_table_plot(
        compact, plots / "performance_table_relative_growth.png"
    )

    test_data = data[data["Split"].eq("Test")].copy().reset_index(drop=True)
    period_predictions = prediction_sets["Period-controlled"]
    bootstrap, paired = cluster_bootstrap_r2(
        test_data,
        period_predictions,
        repetitions=args.bootstrap_repetitions,
    )
    bootstrap.to_csv(tables / "test_r2_cluster_bootstrap_draws.csv", index=False)
    paired.to_csv(tables / "xgb_vs_rf_paired_bootstrap.csv", index=False)
    point_estimates = {
        model: r2_score(
            test_data["Log_Specific_Growth_Rate"],
            period_predictions[model]["Test"],
        )
        for model in ["OLS", "RF", "XGB", "MLP"]
    }
    bootstrap_summary = bootstrap_summary_plot(
        bootstrap,
        point_estimates,
        plots / "model_comparison_test_r2_ci.png",
    )
    bootstrap_summary.to_csv(
        tables / "test_r2_cluster_bootstrap_summary.csv", index=False
    )

    attribution, permutation, baseline_model, baseline_prediction = environmental_attribution(
        data,
        fitted_models["Period-controlled"]["XGB"],
        run_context["Period-controlled"]["x_test"],
        period_predictions["XGB"]["Test"],
        ["Log_Height"] + species_columns + period_columns,
        args.permutation_repetitions,
    )
    attribution_bootstrap = cluster_bootstrap_environmental_attribution(
        test_data,
        baseline_prediction,
        period_predictions["XGB"]["Test"],
        repetitions=args.bootstrap_repetitions,
    )
    for metric in [
        "Baseline_Test_R2",
        "Full_Test_R2",
        "Incremental_Delta_R2",
        "Environmental_Partial_R2",
    ]:
        attribution[f"{metric}_CI95_Lower"] = attribution_bootstrap[
            metric
        ].quantile(0.025)
        attribution[f"{metric}_CI95_Upper"] = attribution_bootstrap[
            metric
        ].quantile(0.975)
    attribution.to_csv(tables / "environmental_attribution.csv", index=False)
    attribution_bootstrap.to_csv(
        tables / "environmental_attribution_tree_bootstrap_draws.csv",
        index=False,
    )
    permutation.to_csv(
        tables / "environmental_block_permutation_draws.csv", index=False
    )

    primary_model = fitted_models["Period-controlled"]["XGB"]
    primary_x_test = run_context["Period-controlled"]["x_test"].reset_index(
        drop=True
    )
    primary_shap = compute_shap(primary_model, primary_x_test)
    environment_indices = [
        primary_x_test.columns.get_loc(feature)
        for feature in ENVIRONMENT_FEATURES
    ]
    primary_environment_shap = primary_shap[:, environment_indices]
    primary_environment_x = primary_x_test[ENVIRONMENT_FEATURES]
    shap_stats, dependence, observations = shap_group_statistics(
        test_data,
        primary_x_test,
        primary_shap,
    )
    shap_stats.to_csv(
        tables / "pooled_onehot_shap_impact_values_by_species.csv",
        index=False,
    )
    dependence.to_csv(
        tables / "pooled_onehot_dependency_data_by_species.csv",
        index=False,
    )
    observations.to_csv(
        tables / "pooled_onehot_environment_shap_observations.csv",
        index=False,
    )

    overall_beeswarm = plots / "xgb_environment_beeswarm.png"
    overall_dependency = plots / "xgb_environment_dependency_grid.png"
    overall_dependency_heatmap = (
        plots / "xgb_environment_dependency_heatmap.png"
    )
    save_beeswarm(
        primary_environment_x,
        primary_environment_shap,
        "Pooled period-controlled XGBoost",
        overall_beeswarm,
    )
    save_dependency_grid(
        primary_environment_x,
        primary_environment_shap,
        dependence,
        overall_dependency,
    )
    save_dependency_heatmap(dependence, overall_dependency_heatmap)
    combine_images(
        overall_beeswarm,
        overall_dependency_heatmap,
        plots / "Figure4_SHAP_beeswarm_dependence.png",
        "Environmental associations in the pooled relative-growth model",
    )

    species_paths = species_beeswarm_plots(
        test_data,
        primary_environment_x,
        primary_environment_shap,
        species_plots,
    )
    contact_sheet(
        species_paths,
        plots / "Figure5_species_SHAP_contact_sheet.png",
        "Pooled one-hot XGBoost SHAP profiles by functional group and genus",
        columns=3,
    )

    no_period_model = fitted_models["No-period"]["XGB"]
    no_period_x_test = run_context["No-period"]["x_test"].reset_index(drop=True)
    no_period_shap = compute_shap(no_period_model, no_period_x_test)
    no_period_stats, _, _ = shap_group_statistics(
        test_data,
        no_period_x_test,
        no_period_shap,
    )
    no_period_stats.to_csv(
        tables / "no_period_shap_impact_values_by_species.csv",
        index=False,
    )
    stability = shap_stability(shap_stats, no_period_stats)
    stability.to_csv(
        tables / "shap_stability_period_vs_no_period.csv", index=False
    )
    no_period_environment_shap = no_period_shap[
        :,
        [no_period_x_test.columns.get_loc(f) for f in ENVIRONMENT_FEATURES],
    ]
    save_beeswarm(
        no_period_x_test[ENVIRONMENT_FEATURES],
        no_period_environment_shap,
        "Pooled XGBoost without monitoring-period indicators",
        plots / "xgb_environment_beeswarm_no_period.png",
    )

    primary_model.save_model(models / "xgb_period_controlled.json")
    no_period_model.save_model(models / "xgb_no_period.json")
    baseline_model.save_model(models / "xgb_biological_temporal_baseline.json")
    joblib.dump(
        {
            "period_controlled_scaler": run_context["Period-controlled"][
                "scaler"
            ],
            "no_period_scaler": run_context["No-period"]["scaler"],
            "period_controlled_medians": medians["Period-controlled"],
            "no_period_medians": medians["No-period"],
            "species_columns": species_columns,
            "period_columns": period_columns,
            "environment_features": ENVIRONMENT_FEATURES,
        },
        models / "preprocessing.joblib",
    )

    with pd.ExcelWriter(tables / "relative_growth_results.xlsx") as writer:
        metrics.to_excel(writer, sheet_name="Model metrics", index=False)
        compact.to_excel(writer, sheet_name="Manuscript Table 4", index=False)
        attribution.to_excel(writer, sheet_name="Env attribution", index=False)
        attribution_bootstrap.to_excel(
            writer, sheet_name="Env bootstrap", index=False
        )
        bootstrap_summary.to_excel(
            writer, sheet_name="Bootstrap R2", index=False
        )
        shap_stats.to_excel(writer, sheet_name="SHAP statistics", index=False)
        dependence.to_excel(writer, sheet_name="SHAP dependence", index=False)
        stability.to_excel(writer, sheet_name="Period stability", index=False)

    metadata.update(
        {
            "raw_rows": int(len(raw)),
            "raw_unique_trees": int(raw[TREE_ID].nunique()),
            "development_rows": int(data["Split"].eq("Development").sum()),
            "test_rows": int(data["Split"].eq("Test").sum()),
            "development_trees": int(
                data.loc[data["Split"].eq("Development"), TREE_ID].nunique()
            ),
            "test_trees": int(
                data.loc[data["Split"].eq("Test"), TREE_ID].nunique()
            ),
            "period_counts": data["Period"].value_counts().to_dict(),
            "target": "ln(([ln(C_end)-ln(C_start)]/years))",
            "feature_sets": {
                "period_controlled": period_features,
                "no_period": no_period_features,
            },
            "parameters": {
                "XGBoost": XGB_PARAMETERS,
                "RandomForest": RF_PARAMETERS,
                "MLP": {
                    **MLP_PARAMETERS,
                    "hidden_layer_sizes": list(
                        MLP_PARAMETERS["hidden_layer_sizes"]
                    ),
                },
            },
            "software": {
                "python": sys.version,
                "platform": platform.platform(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "scikit_learn": sklearn.__version__,
                "xgboost": xgboost.__version__,
                "shap": shap.__version__,
            },
            "elapsed_seconds": time.perf_counter() - started,
        }
    )
    (args.output / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    timings = {
        specification: run_context[specification]["timings_seconds"]
        for specification in run_context
    }
    write_run_log(
        args.output,
        metadata,
        attribution,
        metrics,
        timings,
        paired,
    )
    print(compact.to_string(index=False))
    print()
    print(attribution.to_string(index=False))
    print(f"\nOutputs: {args.output.resolve()}")


if __name__ == "__main__":
    main()
