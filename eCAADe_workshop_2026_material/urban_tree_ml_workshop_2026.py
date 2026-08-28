"""eCAADe 2026 workshop: pooled urban-tree growth modelling workflow.

The workflow is deliberately split into five teaching stages:

1. environment and data preparation;
2. leakage-safe data processing;
3. train/validation/locked-test XGBoost modelling;
4. pooled SHAP explanation (beeswarm, dependence, waterfall, spatial map);
5. ONNX export and numerical parity testing.

The response follows the manuscript definition:

    g = (ln(C_end) - ln(C_start)) / years
    y = ln(g)

where g is annualized specific carbon growth (year^-1). The model predicts y.
Percentage and kg C outputs are explicitly back-transformed after prediction.
All periods from one tree remain in the same split.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib

# Command-line runs use a non-interactive backend; notebooks keep their inline backend.
if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from matplotlib.colors import TwoSlopeNorm
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor


TREE_ID = "Original_Tree_RowID"
TARGET = "Log_Annualized_Specific_Growth"

NUMERIC_FEATURES = [
    "Log_Height",
    "avg_noise_day",
    "Density25",
    "Mono_Rate",
    "avg_svf",
    "avg_radiation",
    "avg_LST",
    "lightemiss",
]
CATEGORICAL_FEATURES = ["Species_Name", "type", "Period"]
RAW_MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
ENVIRONMENTAL_FEATURES = [
    "avg_noise_day",
    "Density25",
    "Mono_Rate",
    "avg_svf",
    "avg_radiation",
    "avg_LST",
    "lightemiss",
]

PERIODS = {
    "2015-2017": {
        "years": 2.0,
        "height_start": "H15",
        "carbon_start": "CS_15",
        "carbon_end": "CS_17",
        "svf_start": "svf15",
        "svf_end": "svf17",
        "radiation": "ra15_17",
        "lst": "LST_1516",
        "noise": ["noise17d"],
    },
    "2017-2021": {
        "years": 4.0,
        "height_start": "H17",
        "carbon_start": "CS_17",
        "carbon_end": "CS_21",
        "svf_start": "svf17",
        "svf_end": "svf21",
        "radiation": "ra17_21",
        "lst": "LST_1720",
        "noise": ["noise17d", "noise22d"],
    },
    "2021-2023": {
        "years": 2.0,
        "height_start": "H21",
        "carbon_start": "CS_21",
        "carbon_end": "CS_23",
        "svf_start": "svf21",
        "svf_end": "svf23",
        "radiation": "ra21_23",
        "lst": "LST_2122",
        "noise": ["noise22d"],
    },
}

REQUIRED_COLUMNS = sorted(
    {
        TREE_ID,
        "Species_Name",
        "type",
        "X",
        "Y",
        "Density25",
        "Mono_Rate",
        "lightemiss",
        *[
            column
            for spec in PERIODS.values()
            for key, value in spec.items()
            if key != "years"
            for column in ([value] if isinstance(value, str) else value)
        ],
    }
)


@dataclass
class WorkflowConfig:
    input_csv: Path
    output_dir: Path
    random_state: int = 2026
    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    test_fraction: float = 0.15
    shap_sample: int = 1200
    example_rows: int = 12
    onnx_opset: int = 15


def make_output_dirs(output_dir: Path) -> dict[str, Path]:
    paths = {
        "root": output_dir,
        "data": output_dir / "data",
        "tables": output_dir / "tables",
        "figures": output_dir / "figures",
        "models": output_dir / "models",
        "onnx": output_dir / "onnx",
        "examples": output_dir / "examples",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def load_tree_level_data(input_csv: Path) -> pd.DataFrame:
    """Load the teaching CSV and validate the source columns."""
    frame = pd.read_csv(input_csv)
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {missing}")
    if frame[TREE_ID].isna().any():
        raise ValueError(f"{TREE_ID} contains missing identifiers.")
    frame[TREE_ID] = frame[TREE_ID].astype(str)
    return frame


def assign_tree_splits(frame: pd.DataFrame, config: WorkflowConfig) -> pd.DataFrame:
    """Assign whole trees to 70/15/15 splits, stratified by species label."""
    total = config.train_fraction + config.validation_fraction + config.test_fraction
    if not math.isclose(total, 1.0, abs_tol=1e-9):
        raise ValueError("Train, validation, and test fractions must sum to 1.")

    per_tree_species = frame.groupby(TREE_ID)["Species_Name"].nunique(dropna=False)
    if int(per_tree_species.max()) > 1:
        raise ValueError("At least one tree has multiple Species_Name values.")

    tree_table = frame[[TREE_ID, "Species_Name"]].drop_duplicates(TREE_ID).copy()
    train_ids, temporary_ids = train_test_split(
        tree_table[TREE_ID],
        test_size=1.0 - config.train_fraction,
        random_state=config.random_state,
        stratify=tree_table["Species_Name"],
    )
    temporary = tree_table[tree_table[TREE_ID].isin(temporary_ids)].copy()
    relative_test_fraction = config.test_fraction / (
        config.validation_fraction + config.test_fraction
    )
    validation_ids, test_ids = train_test_split(
        temporary[TREE_ID],
        test_size=relative_test_fraction,
        random_state=config.random_state + 1,
        stratify=temporary["Species_Name"],
    )

    split_map = {tree_id: "train" for tree_id in train_ids}
    split_map.update({tree_id: "validation" for tree_id in validation_ids})
    split_map.update({tree_id: "test" for tree_id in test_ids})
    result = frame.copy()
    result["Split"] = result[TREE_ID].map(split_map)
    if result["Split"].isna().any():
        raise RuntimeError("Split assignment failed for one or more trees.")
    return result


def build_long_format(tree_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert one-row-per-tree data into one-row-per-tree-period data."""
    records: list[pd.DataFrame] = []
    audit_rows: list[dict[str, Any]] = []

    for period, spec in PERIODS.items():
        part = tree_frame.copy()
        start_carbon = pd.to_numeric(part[spec["carbon_start"]], errors="coerce")
        end_carbon = pd.to_numeric(part[spec["carbon_end"]], errors="coerce")
        start_height = pd.to_numeric(part[spec["height_start"]], errors="coerce")
        years = float(spec["years"])

        positive_carbon = (start_carbon > 0) & (end_carbon > 0)
        specific_growth = pd.Series(np.nan, index=part.index, dtype=float)
        specific_growth.loc[positive_carbon] = (
            np.log(end_carbon.loc[positive_carbon])
            - np.log(start_carbon.loc[positive_carbon])
        ) / years
        valid = (
            np.isfinite(specific_growth)
            & (specific_growth > 0)
            & np.isfinite(start_height)
            & (start_height > 0)
            & np.isfinite(start_carbon)
            & (start_carbon > 0)
        )

        before = len(part)
        part = part.loc[valid].copy()
        specific_growth = specific_growth.loc[valid]
        start_carbon = start_carbon.loc[valid]

        part["Period"] = period
        part["Years"] = years
        part["Initial_Carbon_kg"] = start_carbon
        part["Log_Height"] = np.log(start_height.loc[valid])
        part["avg_svf"] = part[[spec["svf_start"], spec["svf_end"]]].mean(axis=1)
        part["avg_radiation"] = pd.to_numeric(part[spec["radiation"]], errors="coerce")
        part["avg_LST"] = pd.to_numeric(part[spec["lst"]], errors="coerce")
        part["avg_noise_day"] = part[spec["noise"]].mean(axis=1)
        part["Annualized_Specific_Growth"] = specific_growth
        part[TARGET] = np.log(specific_growth)
        part["Observed_Annual_Growth_Percent"] = 100.0 * np.expm1(specific_growth)
        part["Observed_Annual_Carbon_Gain_kg"] = (
            start_carbon * np.expm1(specific_growth)
        )

        numeric_check = NUMERIC_FEATURES + [TARGET, "X", "Y"]
        for column in numeric_check:
            part[column] = pd.to_numeric(part[column], errors="coerce")
        before_feature_filter = len(part)
        part = part.replace([np.inf, -np.inf], np.nan).dropna(
            subset=RAW_MODEL_FEATURES
            + [TARGET, "Initial_Carbon_kg", "X", "Y", TREE_ID, "Split"]
        )

        audit_rows.append(
            {
                "period": period,
                "tree_rows_available": before,
                "rows_after_positive_growth_filter": before_feature_filter,
                "rows_after_complete_feature_filter": len(part),
                "rows_removed_total": before - len(part),
            }
        )
        records.append(part)

    long_frame = pd.concat(records, ignore_index=True)
    leakage_check = long_frame.groupby(TREE_ID)["Split"].nunique()
    if int(leakage_check.max()) != 1:
        raise RuntimeError("Tree-level leakage detected after long-format conversion.")

    columns = [
        TREE_ID,
        "id",
        "Species_Name",
        "type",
        "Period",
        "Split",
        "X",
        "Y",
        "Initial_Carbon_kg",
        *NUMERIC_FEATURES,
        "Annualized_Specific_Growth",
        TARGET,
        "Observed_Annual_Growth_Percent",
        "Observed_Annual_Carbon_Gain_kg",
    ]
    columns = list(dict.fromkeys(column for column in columns if column in long_frame))
    return long_frame[columns].copy(), pd.DataFrame(audit_rows)


def prepare_data(config: WorkflowConfig) -> dict[str, Any]:
    """Run stages 1-2 and save an auditable prepared dataset."""
    paths = make_output_dirs(config.output_dir)
    raw = load_tree_level_data(config.input_csv)
    split_tree = assign_tree_splits(raw, config)
    long_frame, processing_audit = build_long_format(split_tree)

    split_summary = (
        long_frame.groupby("Split")
        .agg(rows=(TREE_ID, "size"), trees=(TREE_ID, "nunique"))
        .reindex(["train", "validation", "test"])
        .reset_index()
    )
    species_summary = (
        long_frame.groupby(["Split", "Species_Name"], observed=True)
        .size()
        .rename("rows")
        .reset_index()
    )
    target_summary = (
        long_frame.groupby("Split")[
            [
                "Annualized_Specific_Growth",
                TARGET,
                "Observed_Annual_Growth_Percent",
                "Observed_Annual_Carbon_Gain_kg",
            ]
        ]
        .describe()
    )

    long_frame.to_csv(paths["data"] / "prepared_long_format.csv", index=False)
    processing_audit.to_csv(paths["tables"] / "processing_audit.csv", index=False)
    split_summary.to_csv(paths["tables"] / "split_summary.csv", index=False)
    species_summary.to_csv(paths["tables"] / "split_species_summary.csv", index=False)
    target_summary.to_csv(paths["tables"] / "target_summary.csv")

    print("Prepared pooled long-format data")
    print(split_summary.to_string(index=False))
    return {
        "raw": raw,
        "long": long_frame,
        "processing_audit": processing_audit,
        "split_summary": split_summary,
        "paths": paths,
    }


def build_preprocessor() -> ColumnTransformer:
    numeric_pipe = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipe, NUMERIC_FEATURES),
            ("categorical", categorical_pipe, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def readable_feature_name(name: str) -> str:
    result = name.replace("numeric__", "").replace("categorical__", "")
    result = result.replace("Species_Name_", "Species: ")
    result = result.replace("Period_", "Period: ")
    result = result.replace("type_", "Site: ")
    labels = {
        "Log_Height": "Log height",
        "avg_noise_day": "Daytime noise",
        "Density25": "Tree density (25 m)",
        "Mono_Rate": "Monoculture rate",
        "avg_svf": "Sky-view factor",
        "avg_radiation": "Solar radiation",
        "avg_LST": "Land-surface temperature",
        "lightemiss": "Night illumination",
    }
    return labels.get(result, result)


def as_engineered_frame(
    preprocessor: ColumnTransformer, raw_features: pd.DataFrame
) -> pd.DataFrame:
    values = preprocessor.transform(raw_features)
    names = [readable_feature_name(name) for name in preprocessor.get_feature_names_out()]
    return pd.DataFrame(values, columns=names, index=raw_features.index).astype(np.float32)


def back_transform(log_sgr: np.ndarray, initial_carbon_kg: np.ndarray) -> dict[str, np.ndarray]:
    log_sgr = np.asarray(log_sgr, dtype=float).reshape(-1)
    initial = np.asarray(initial_carbon_kg, dtype=float).reshape(-1)
    sgr = np.exp(np.clip(log_sgr, -20.0, 5.0))
    fraction = np.expm1(sgr)
    return {
        "log_sgr": log_sgr,
        "specific_growth_rate": sgr,
        "growth_percent": 100.0 * fraction,
        "carbon_gain_kg": initial * fraction,
    }


def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    initial_carbon_kg: np.ndarray,
) -> dict[str, float]:
    true_outputs = back_transform(y_true, initial_carbon_kg)
    pred_outputs = back_transform(y_pred, initial_carbon_kg)
    return {
        "R2_log_SGR": float(r2_score(y_true, y_pred)),
        "RMSE_log_SGR": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE_log_SGR": float(mean_absolute_error(y_true, y_pred)),
        "RMSE_annual_percentage_points": float(
            math.sqrt(
                mean_squared_error(
                    true_outputs["growth_percent"], pred_outputs["growth_percent"]
                )
            )
        ),
        "MAE_annual_percentage_points": float(
            mean_absolute_error(
                true_outputs["growth_percent"], pred_outputs["growth_percent"]
            )
        ),
        "RMSE_kg_C_tree_year": float(
            math.sqrt(
                mean_squared_error(
                    true_outputs["carbon_gain_kg"], pred_outputs["carbon_gain_kg"]
                )
            )
        ),
        "MAE_kg_C_tree_year": float(
            mean_absolute_error(
                true_outputs["carbon_gain_kg"], pred_outputs["carbon_gain_kg"]
            )
        ),
    }


def xgb_parameters(random_state: int, n_estimators: int) -> dict[str, Any]:
    return {
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "n_estimators": n_estimators,
        "max_depth": 4,
        "learning_rate": 0.03,
        "min_child_weight": 8,
        "subsample": 0.80,
        "colsample_bytree": 0.85,
        "reg_lambda": 10.0,
        "reg_alpha": 0.30,
        "gamma": 0.02,
        "random_state": random_state,
        "n_jobs": -1,
        "verbosity": 0,
    }


def train_validate_refit_test(
    prepared: dict[str, Any], config: WorkflowConfig
) -> dict[str, Any]:
    """Select boosting rounds on validation, refit on 85%, test once."""
    frame = prepared["long"]
    paths = prepared["paths"]
    split = {name: frame[frame["Split"] == name].copy() for name in ["train", "validation", "test"]}

    selection_preprocessor = build_preprocessor()
    selection_preprocessor.fit(split["train"][RAW_MODEL_FEATURES])
    x_train = as_engineered_frame(selection_preprocessor, split["train"][RAW_MODEL_FEATURES])
    x_validation = as_engineered_frame(selection_preprocessor, split["validation"][RAW_MODEL_FEATURES])
    y_train = split["train"][TARGET].to_numpy(dtype=np.float32)
    y_validation = split["validation"][TARGET].to_numpy(dtype=np.float32)

    selection_model = XGBRegressor(
        **xgb_parameters(config.random_state, n_estimators=1500),
        early_stopping_rounds=60,
    )
    selection_model.fit(
        x_train,
        y_train,
        eval_set=[(x_validation, y_validation)],
        verbose=False,
    )
    best_iteration = int(getattr(selection_model, "best_iteration", 1499))
    selected_trees = best_iteration + 1

    validation_prediction = selection_model.predict(x_validation)
    validation_metrics = regression_metrics(
        y_validation,
        validation_prediction,
        split["validation"]["Initial_Carbon_kg"].to_numpy(),
    )

    development = pd.concat([split["train"], split["validation"]], ignore_index=True)
    final_preprocessor = build_preprocessor()
    final_preprocessor.fit(development[RAW_MODEL_FEATURES])
    x_development = as_engineered_frame(final_preprocessor, development[RAW_MODEL_FEATURES])
    x_test = as_engineered_frame(final_preprocessor, split["test"][RAW_MODEL_FEATURES])

    final_model = XGBRegressor(
        **xgb_parameters(config.random_state, n_estimators=selected_trees)
    )
    final_model.fit(x_development, development[TARGET].to_numpy(dtype=np.float32))
    test_prediction = final_model.predict(x_test)
    test_metrics = regression_metrics(
        split["test"][TARGET].to_numpy(),
        test_prediction,
        split["test"]["Initial_Carbon_kg"].to_numpy(),
    )

    metric_rows = [
        {"stage": "validation_model_selection", "n": len(split["validation"]), **validation_metrics},
        {"stage": "locked_test_after_85pct_refit", "n": len(split["test"]), **test_metrics},
    ]
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(paths["tables"] / "model_metrics.csv", index=False)

    predictions = split["test"][[
        TREE_ID,
        "id",
        "Species_Name",
        "type",
        "Period",
        "X",
        "Y",
        "Initial_Carbon_kg",
        TARGET,
        "Observed_Annual_Growth_Percent",
        "Observed_Annual_Carbon_Gain_kg",
    ]].copy()
    transformed = back_transform(test_prediction, predictions["Initial_Carbon_kg"].to_numpy())
    predictions["Predicted_Log_SGR"] = transformed["log_sgr"]
    predictions["Predicted_Annual_Growth_Percent"] = transformed["growth_percent"]
    predictions["Predicted_Annual_Carbon_Gain_kg"] = transformed["carbon_gain_kg"]
    predictions.to_csv(paths["tables"] / "locked_test_predictions.csv", index=False)

    joblib.dump(final_preprocessor, paths["models"] / "pooled_preprocessor.joblib")
    joblib.dump(final_model, paths["models"] / "pooled_xgboost.joblib")
    final_model.save_model(paths["models"] / "pooled_xgboost.json")
    pd.DataFrame({"engineered_feature": x_development.columns}).to_csv(
        paths["tables"] / "engineered_feature_order.csv", index=False
    )

    print(f"Selected boosting rounds from validation: {selected_trees}")
    print(metrics.round(4).to_string(index=False))
    return {
        "split": split,
        "development": development,
        "preprocessor": final_preprocessor,
        "model": final_model,
        "x_development": x_development,
        "x_test": x_test,
        "test_prediction": test_prediction,
        "metrics": metrics,
        "selected_trees": selected_trees,
        "paths": paths,
    }


def explain_with_shap(
    trained: dict[str, Any], config: WorkflowConfig
) -> dict[str, Any]:
    """Create pooled test-set beeswarm, dependence, waterfall, and spatial SHAP."""
    paths = trained["paths"]
    x_test = trained["x_test"]
    test_raw = trained["split"]["test"]
    sample_n = min(config.shap_sample, len(x_test))
    rng = np.random.default_rng(config.random_state)
    sample_positions = np.sort(rng.choice(len(x_test), size=sample_n, replace=False))
    x_sample = x_test.iloc[sample_positions].copy()
    raw_sample = test_raw.iloc[sample_positions].copy()

    explainer = shap.TreeExplainer(trained["model"])
    shap_values = np.asarray(explainer.shap_values(x_sample))
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 0]
    expected_value = float(np.asarray(explainer.expected_value).reshape(-1)[0])
    explanation = shap.Explanation(
        values=shap_values,
        base_values=np.full(sample_n, expected_value),
        data=x_sample.to_numpy(),
        feature_names=list(x_sample.columns),
    )

    plt.figure(figsize=(10, 7))
    shap.plots.beeswarm(explanation, max_display=15, show=False)
    plt.title("Pooled XGBoost SHAP beeswarm · locked test sample")
    plt.tight_layout()
    beeswarm_path = paths["figures"] / "shap_beeswarm_pooled.png"
    plt.savefig(beeswarm_path, dpi=220, bbox_inches="tight")
    plt.close()

    mean_abs = np.abs(shap_values).mean(axis=0)
    summary = pd.DataFrame(
        {"feature": x_sample.columns, "mean_absolute_SHAP": mean_abs}
    ).sort_values("mean_absolute_SHAP", ascending=False)
    summary.to_csv(paths["tables"] / "shap_mean_absolute_summary.csv", index=False)

    environmental_columns = [
        name for name in x_sample.columns if name in {readable_feature_name(x) for x in ENVIRONMENTAL_FEATURES}
    ]
    top_environment = (
        summary[summary["feature"].isin(environmental_columns)]
        .head(3)["feature"]
        .tolist()
    )
    fig, axes = plt.subplots(1, len(top_environment), figsize=(6 * len(top_environment), 5))
    axes = np.atleast_1d(axes)
    for axis, feature in zip(axes, top_environment):
        feature_index = x_sample.columns.get_loc(feature)
        interaction_ranking = shap.approximate_interactions(feature_index, shap_values, x_sample)
        interaction_index = next(
            (int(index) for index in interaction_ranking if int(index) != feature_index),
            feature_index,
        )
        shap.dependence_plot(
            feature_index,
            shap_values,
            x_sample,
            interaction_index=interaction_index,
            ax=axis,
            show=False,
        )
        axis.set_title(feature)
        axis.axhline(0, color="0.45", linestyle="--", linewidth=0.9)
    fig.suptitle("SHAP dependence · strongest approximate interactions", fontsize=15)
    fig.tight_layout()
    dependence_path = paths["figures"] / "shap_dependence_top3_environment.png"
    fig.savefig(dependence_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    sample_predictions = trained["model"].predict(x_sample)
    representative = int(np.argmin(np.abs(sample_predictions - np.median(sample_predictions))))
    plt.figure(figsize=(9, 6))
    shap.plots.waterfall(explanation[representative], max_display=12, show=False)
    plt.title("SHAP waterfall · representative locked-test observation")
    plt.tight_layout()
    waterfall_path = paths["figures"] / "shap_waterfall_representative.png"
    plt.savefig(waterfall_path, dpi=220, bbox_inches="tight")
    plt.close()

    top_spatial_feature = top_environment[0]
    spatial_index = x_sample.columns.get_loc(top_spatial_feature)
    spatial_values = shap_values[:, spatial_index]
    limit = float(np.quantile(np.abs(spatial_values), 0.99))
    limit = max(limit, 1e-9)
    x_km = (raw_sample["X"].to_numpy() - test_raw["X"].min()) / 1000.0
    y_km = (raw_sample["Y"].to_numpy() - test_raw["Y"].min()) / 1000.0
    fig, axis = plt.subplots(figsize=(7.5, 8.5))
    points = axis.scatter(
        x_km,
        y_km,
        c=spatial_values,
        s=13,
        cmap="RdYlGn",
        norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
        alpha=0.82,
        linewidths=0,
    )
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("Easting from study minimum (km)")
    axis.set_ylabel("Northing from study minimum (km)")
    axis.set_title(f"Spatial SHAP map · {top_spatial_feature}")
    axis.annotate("N", xy=(0.06, 0.96), xytext=(0.06, 0.87), xycoords="axes fraction", ha="center", arrowprops={"arrowstyle": "-|>", "color": "black"})
    colorbar = fig.colorbar(points, ax=axis, shrink=0.75)
    colorbar.set_label("SHAP contribution to log-SGR")
    fig.tight_layout()
    spatial_path = paths["figures"] / "spatial_shap_top_environmental_feature.png"
    fig.savefig(spatial_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    spatial_table = raw_sample[[TREE_ID, "Species_Name", "Period", "X", "Y"]].copy()
    spatial_table["mapped_feature"] = top_spatial_feature
    spatial_table["feature_value"] = x_sample[top_spatial_feature].to_numpy()
    spatial_table["SHAP_value"] = spatial_values
    spatial_table.to_csv(paths["tables"] / "spatial_shap_test_points.csv", index=False)

    print("SHAP figures saved:")
    for path in [beeswarm_path, dependence_path, waterfall_path, spatial_path]:
        print(" -", path)
    return {
        "explanation": explanation,
        "summary": summary,
        "top_environment": top_environment,
        "beeswarm": beeswarm_path,
        "dependence": dependence_path,
        "waterfall": waterfall_path,
        "spatial": spatial_path,
    }


def add_onnx_output_transforms(model: Any, feature_names: list[str]) -> Any:
    """Add percent and kg C back-transformations to an XGBoost ONNX graph."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    raw_output = model.graph.output[0].name
    model.graph.input.extend(
        [helper.make_tensor_value_info("initial_carbon_kg", TensorProto.FLOAT, [None, 1])]
    )
    model.graph.node.extend(
        [
            helper.make_node("Identity", [raw_output], ["log_sgr"], name="ExposeLogSGR"),
            helper.make_node("Exp", [raw_output], ["specific_growth_rate"], name="BackTransformLogSGR"),
            helper.make_node("Exp", ["specific_growth_rate"], ["growth_factor"], name="OneYearGrowthFactor"),
            helper.make_node("Sub", ["growth_factor", "constant_one"], ["growth_fraction"], name="GrowthFraction"),
            helper.make_node("Mul", ["growth_fraction", "constant_hundred"], ["annual_growth_percent"], name="GrowthPercent"),
            helper.make_node("Mul", ["growth_fraction", "initial_carbon_kg"], ["carbon_gain_kg_tree_year"], name="CarbonGain"),
        ]
    )
    model.graph.initializer.extend(
        [
            numpy_helper.from_array(np.asarray(1.0, dtype=np.float32), "constant_one"),
            numpy_helper.from_array(np.asarray(100.0, dtype=np.float32), "constant_hundred"),
        ]
    )
    del model.graph.output[:]
    model.graph.output.extend(
        [
            helper.make_tensor_value_info("annual_growth_percent", TensorProto.FLOAT, [None, 1]),
            helper.make_tensor_value_info("carbon_gain_kg_tree_year", TensorProto.FLOAT, [None, 1]),
            helper.make_tensor_value_info("log_sgr", TensorProto.FLOAT, [None, 1]),
            helper.make_tensor_value_info("specific_growth_rate", TensorProto.FLOAT, [None, 1]),
        ]
    )
    model.graph.name = "eCAADe2026PooledUrbanTreeGrowth"
    model.doc_string = (
        "Pooled teaching model. SHAP and predictions are fitted associations, not causal effects."
    )
    metadata = {
        "engineered_feature_order": json.dumps(feature_names),
        "target": "log_sgr = ln(annualized_specific_carbon_growth_rate)",
        "percent_output": "100 * (exp(exp(log_sgr)) - 1)",
        "kg_output": "initial_carbon_kg * (exp(exp(log_sgr)) - 1)",
        "model_scope": "single pooled model; species and period are one-hot inputs",
    }
    for key, value in metadata.items():
        entry = model.metadata_props.add()
        entry.key = key
        entry.value = value
    if not any(item.domain == "" for item in model.opset_import):
        model.opset_import.append(helper.make_opsetid("", 15))
    onnx.checker.check_model(model)
    return model


def export_onnx_and_examples(
    trained: dict[str, Any], config: WorkflowConfig
) -> dict[str, Any]:
    """Export ONNX, raw/engineered examples, and verify ONNX Runtime parity."""
    import onnx
    import onnxruntime as ort
    from onnxmltools.convert import convert_xgboost
    from onnxmltools.convert.common.data_types import FloatTensorType

    paths = trained["paths"]
    feature_names = list(trained["x_test"].columns)
    source_model_path = paths["models"] / "pooled_xgboost.json"
    booster = xgb.Booster()
    booster.load_model(source_model_path)
    converter_features = [f"f{index}" for index in range(len(feature_names))]
    booster.feature_names = converter_features

    onnx_model = convert_xgboost(
        booster,
        initial_types=[("features", FloatTensorType([None, len(feature_names)]))],
        target_opset=config.onnx_opset,
    )
    onnx_model = add_onnx_output_transforms(onnx_model, feature_names)
    onnx_path = paths["onnx"] / "pooled_urban_tree_growth.onnx"
    onnx.save_model(onnx_model, onnx_path)

    n_examples = min(config.example_rows, len(trained["x_test"]))
    example_positions = np.linspace(0, len(trained["x_test"]) - 1, n_examples, dtype=int)
    raw_examples = trained["split"]["test"].iloc[example_positions].copy()
    engineered_examples = trained["x_test"].iloc[example_positions].copy()
    initial_carbon = raw_examples["Initial_Carbon_kg"].to_numpy(dtype=np.float32).reshape(-1, 1)

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_values = session.run(
        None,
        {
            "features": engineered_examples.to_numpy(dtype=np.float32),
            "initial_carbon_kg": initial_carbon,
        },
    )
    output_names = [item.name for item in session.get_outputs()]
    output_map = dict(zip(output_names, ort_values))
    python_log = trained["model"].predict(engineered_examples).astype(np.float32)
    python_outputs = back_transform(python_log, initial_carbon.reshape(-1))
    onnx_log = np.asarray(output_map["log_sgr"]).reshape(-1)
    onnx_percent = np.asarray(output_map["annual_growth_percent"]).reshape(-1)
    onnx_kg = np.asarray(output_map["carbon_gain_kg_tree_year"]).reshape(-1)

    parity = {
        "maximum_absolute_log_sgr_error": float(np.max(np.abs(python_log - onnx_log))),
        "maximum_absolute_annual_growth_percent_error": float(
            np.max(np.abs(python_outputs["growth_percent"] - onnx_percent))
        ),
        "maximum_absolute_carbon_gain_kg_error": float(
            np.max(np.abs(python_outputs["carbon_gain_kg"] - onnx_kg))
        ),
    }
    if parity["maximum_absolute_log_sgr_error"] > 1e-5:
        raise RuntimeError(f"ONNX log-SGR parity check failed: {parity}")
    if parity["maximum_absolute_annual_growth_percent_error"] > 2e-4:
        raise RuntimeError(f"ONNX percentage parity check failed: {parity}")
    if parity["maximum_absolute_carbon_gain_kg_error"] > 2e-3:
        raise RuntimeError(f"ONNX kg C parity check failed: {parity}")

    raw_export_columns = [
        TREE_ID,
        "Species_Name",
        "type",
        "Period",
        "X",
        "Y",
        "Initial_Carbon_kg",
        *NUMERIC_FEATURES,
        TARGET,
        "Observed_Annual_Growth_Percent",
        "Observed_Annual_Carbon_Gain_kg",
    ]
    raw_examples[raw_export_columns].to_csv(
        paths["examples"] / "example_test_set_raw.csv", index=False
    )
    engineered_export = engineered_examples.reset_index(drop=True).copy()
    engineered_export.insert(0, "initial_carbon_kg", initial_carbon.reshape(-1))
    engineered_export.to_csv(
        paths["examples"] / "example_test_set_engineered.csv", index=False
    )
    prediction_export = pd.DataFrame(
        {
            TREE_ID: raw_examples[TREE_ID].to_numpy(),
            "python_log_sgr": python_log,
            "onnx_log_sgr": onnx_log,
            "onnx_annual_growth_percent": onnx_percent,
            "onnx_carbon_gain_kg_tree_year": onnx_kg,
        }
    )
    prediction_export.to_csv(
        paths["examples"] / "example_onnx_predictions.csv", index=False
    )

    schema = {
        "raw_model_features": RAW_MODEL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "engineered_feature_order": feature_names,
        "onnx_inputs": [
            {"name": "features", "dtype": "float32", "shape": [None, len(feature_names)]},
            {"name": "initial_carbon_kg", "dtype": "float32", "shape": [None, 1]},
        ],
        "onnx_outputs": [
            "annual_growth_percent",
            "carbon_gain_kg_tree_year",
            "log_sgr",
            "specific_growth_rate",
        ],
        "parity": parity,
    }
    (paths["onnx"] / "feature_schema.json").write_text(
        json.dumps(schema, indent=2), encoding="utf-8"
    )
    (paths["onnx"] / "onnx_parity_report.json").write_text(
        json.dumps(parity, indent=2), encoding="utf-8"
    )
    print("ONNX export passed parity checks")
    print(json.dumps(parity, indent=2))
    return {"onnx_path": onnx_path, "schema": schema, "parity": parity}


def environment_report() -> dict[str, str]:
    versions = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": __import__("sklearn").__version__,
        "xgboost": xgb.__version__,
        "shap": shap.__version__,
    }
    try:
        versions["onnx"] = __import__("onnx").__version__
        versions["onnxruntime"] = __import__("onnxruntime").__version__
        versions["onnxmltools"] = __import__("onnxmltools").__version__
    except Exception:
        pass
    return versions


def run_workflow(config: WorkflowConfig) -> dict[str, Any]:
    paths = make_output_dirs(config.output_dir)
    (paths["root"] / "environment_versions.json").write_text(
        json.dumps(environment_report(), indent=2), encoding="utf-8"
    )
    (paths["root"] / "workflow_config.json").write_text(
        json.dumps({**asdict(config), "input_csv": str(config.input_csv), "output_dir": str(config.output_dir)}, indent=2),
        encoding="utf-8",
    )

    prepared = prepare_data(config)
    trained = train_validate_refit_test(prepared, config)
    shap_outputs = explain_with_shap(trained, config)
    onnx_outputs = export_onnx_and_examples(trained, config)
    return {
        "prepared": prepared,
        "trained": trained,
        "shap": shap_outputs,
        "onnx": onnx_outputs,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    default_input = Path(__file__).resolve().parent / "data" / "tree_carbon_ml_teaching_sample.csv"
    default_output = Path(__file__).resolve().parent / "outputs"
    parser.add_argument("--input", type=Path, default=default_input)
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument("--random-state", type=int, default=2026)
    parser.add_argument("--shap-sample", type=int, default=1200)
    parser.add_argument("--example-rows", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = WorkflowConfig(
        input_csv=args.input.resolve(),
        output_dir=args.output.resolve(),
        random_state=args.random_state,
        shap_sample=args.shap_sample,
        example_rows=args.example_rows,
    )
    print("Environment")
    print(json.dumps(environment_report(), indent=2))
    run_workflow(config)
    print("Completed. Outputs:", config.output_dir)


if __name__ == "__main__":
    main()
