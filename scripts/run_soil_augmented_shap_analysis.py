#!/usr/bin/env python3
"""Fit and explain a soil-augmented pooled XGBoost sensitivity model.

The primary manuscript specification is preserved. This script reconstructs
the identical leakage-free tree split, fits a period-controlled base XGBoost
and a period-controlled XGBoost with a selectable soil-composition block,
and produces independent-test metrics plus pooled/species-subset SHAP outputs.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import shap
from sklearn.metrics import r2_score
from xgboost import XGBRegressor


ALL_SOIL_FEATURES = [
    "soil_infill",
    "soil_bedrock",
    "soil_moraine",
    "soil_clay",
    "soil_silt_sand",
]

REDUCED_SOIL_FEATURES = [
    "soil_infill",
    "soil_bedrock",
    "soil_moraine",
]

SOIL_LABELS = {
    "soil_infill": "Soil: fill",
    "soil_bedrock": "Soil: bedrock",
    "soil_moraine": "Soil: moraine",
    "soil_clay": "Soil: clay",
    "soil_silt_sand": "Soil: silt–sand",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--pipeline-script", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument(
        "--soil-feature-set",
        choices=["five", "three"],
        default="five",
        help="Use all five parsed indicators or the reduced infill/bedrock/moraine block.",
    )
    return parser.parse_args()


def load_pipeline(path: Path):
    spec = importlib.util.spec_from_file_location("relative_growth_pipeline", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load pipeline module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def add_soil_indicators(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "soil" not in raw:
        raise KeyError("Input dataset has no 'soil' column")
    result = raw.copy()
    normalized = (
        result["soil"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.casefold()
        .str.replace(" ", "", regex=False)
    )
    result["soil_infill"] = normalized.str.startswith("t").astype(np.float32)
    result["soil_bedrock"] = normalized.str.contains("ka", regex=False).astype(
        np.float32
    )
    result["soil_moraine"] = normalized.str.contains("mr", regex=False).astype(
        np.float32
    )
    result["soil_clay"] = normalized.str.contains("sa", regex=False).astype(
        np.float32
    )
    result["soil_silt_sand"] = normalized.str.contains(
        "s+h", regex=False
    ).astype(np.float32)

    counts = (
        result.assign(
            soil_raw=result["soil"].fillna("<missing>").astype(str),
            soil_normalized=normalized.replace("", "<missing>"),
        )
        .groupby(["soil_raw", "soil_normalized", *ALL_SOIL_FEATURES], dropna=False)
        .size()
        .rename("N_Raw_Trees")
        .reset_index()
        .sort_values("N_Raw_Trees", ascending=False)
    )
    return result, counts


def prediction_metrics(
    pipeline,
    data: pd.DataFrame,
    predictions: dict[str, dict[str, np.ndarray]],
) -> pd.DataFrame:
    rows = []
    for split in ["Development", "Test"]:
        mask = data["Split"].eq(split)
        actual = data.loc[mask, "Log_Specific_Growth_Rate"].to_numpy(dtype=float)
        carbon = data.loc[mask, "Initial_Carbon"].to_numpy(dtype=float)
        years = data.loc[mask, "Years"].to_numpy(dtype=float)
        for specification, split_predictions in predictions.items():
            record = pipeline.metric_record(
                actual,
                split_predictions[split],
                carbon,
                years,
            )
            record.update(
                {
                    "Specification": specification,
                    "Split": split,
                    "N": int(mask.sum()),
                }
            )
            rows.append(record)
    return pd.DataFrame(rows)[
        ["Specification", "Split", "N"]
        + [
            column
            for column in rows[0]
            if column not in {"Specification", "Split", "N"}
        ]
    ]


def clustered_delta_bootstrap(
    test_data: pd.DataFrame,
    base_prediction: np.ndarray,
    soil_prediction: np.ndarray,
    replicates: int,
    random_state: int,
) -> pd.DataFrame:
    tree_ids = test_data["OID_"].drop_duplicates().to_numpy()
    row_lookup = {
        tree_id: np.flatnonzero(test_data["OID_"].to_numpy() == tree_id)
        for tree_id in tree_ids
    }
    actual = test_data["Log_Specific_Growth_Rate"].to_numpy(dtype=float)
    rng = np.random.default_rng(random_state)
    records = []
    for replicate in range(1, replicates + 1):
        sampled_ids = rng.choice(tree_ids, size=len(tree_ids), replace=True)
        sampled_rows = np.concatenate([row_lookup[tree_id] for tree_id in sampled_ids])
        base_r2 = r2_score(actual[sampled_rows], base_prediction[sampled_rows])
        soil_r2 = r2_score(actual[sampled_rows], soil_prediction[sampled_rows])
        records.append(
            {
                "Replicate": replicate,
                "Base_Test_R2": base_r2,
                "Soil_Augmented_Test_R2": soil_r2,
                "Delta_R2_Soil_minus_Base": soil_r2 - base_r2,
            }
        )
    return pd.DataFrame(records)


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True)
    tables = args.output / "tables"
    plots = args.output / "plots"
    species_plots = plots / "species_beeswarm"
    models = args.output / "models"
    for directory in [tables, plots, species_plots, models]:
        directory.mkdir()

    pipeline = load_pipeline(args.pipeline_script)
    selected_soil_features = (
        REDUCED_SOIL_FEATURES
        if args.soil_feature_set == "three"
        else ALL_SOIL_FEATURES
    )
    soil_model_name = (
        "Three_soil_period_controlled_XGB"
        if args.soil_feature_set == "three"
        else "Soil_augmented_period_controlled_XGB"
    )
    soil_descriptor = (
        "three-soil (infill, bedrock, moraine)"
        if args.soil_feature_set == "three"
        else "five-soil"
    )
    raw = pd.read_csv(args.input)
    raw_with_soil, soil_counts = add_soil_indicators(raw)
    long_data, construction_metadata = pipeline.build_long_data(raw_with_soil)
    data, encoding = pipeline.add_split_and_dummies(long_data)

    base_environment = list(pipeline.ENVIRONMENT_FEATURES)
    base_features = (
        ["Log_Height"]
        + list(encoding["species_columns"])
        + list(encoding["period_columns"])
        + base_environment
    )
    soil_features = base_features + selected_soil_features
    development = data["Split"].eq("Development")
    test = ~development
    y = data["Log_Specific_Growth_Rate"].astype(np.float32)

    fitted = {}
    matrices = {}
    medians = {}
    predictions: dict[str, dict[str, np.ndarray]] = {}
    for name, features in [
        ("Base_period_controlled_XGB", base_features),
        (soil_model_name, soil_features),
    ]:
        x_development, x_test, feature_medians = pipeline.prepare_matrices(
            data, features
        )
        model = XGBRegressor(**pipeline.XGB_PARAMETERS)
        model.fit(x_development, y.loc[development])
        fitted[name] = model
        matrices[name] = (x_development, x_test)
        medians[name] = feature_medians
        predictions[name] = {
            "Development": model.predict(x_development),
            "Test": model.predict(x_test),
        }

    metrics = prediction_metrics(pipeline, data, predictions)
    base_test = metrics[
        metrics["Specification"].eq("Base_period_controlled_XGB")
        & metrics["Split"].eq("Test")
    ].iloc[0]
    soil_test = metrics[
        metrics["Specification"].eq(soil_model_name)
        & metrics["Split"].eq("Test")
    ].iloc[0]
    delta_r2 = float(soil_test["R2_LogSGR"] - base_test["R2_LogSGR"])
    partial_r2 = delta_r2 / (1.0 - float(base_test["R2_LogSGR"]))

    test_data = data.loc[test].reset_index(drop=True)
    bootstrap = clustered_delta_bootstrap(
        test_data,
        predictions["Base_period_controlled_XGB"]["Test"],
        predictions[soil_model_name]["Test"],
        args.bootstrap_replicates,
        pipeline.RANDOM_STATE + 7000,
    )
    bootstrap_summary = pd.DataFrame(
        [
            {
                "Observed_Delta_R2": delta_r2,
                "Environmental_Soil_Partial_R2_vs_Base": partial_r2,
                "Bootstrap_Replicates": args.bootstrap_replicates,
                "Cluster_Unit": "OID_",
                "Delta_R2_CI_Lower_2_5pct": bootstrap[
                    "Delta_R2_Soil_minus_Base"
                ].quantile(0.025),
                "Delta_R2_CI_Upper_97_5pct": bootstrap[
                    "Delta_R2_Soil_minus_Base"
                ].quantile(0.975),
                "Probability_Delta_R2_Positive": (
                    bootstrap["Delta_R2_Soil_minus_Base"] > 0
                ).mean(),
            }
        ]
    )

    soil_model = fitted[soil_model_name]
    x_development, x_test = matrices[soil_model_name]
    shap_values = np.asarray(
        shap.TreeExplainer(soil_model).shap_values(x_test), dtype=np.float32
    )

    # Reuse the manuscript pipeline's summary and subset plotting routines,
    # with soil indicators added to the environmental block.
    pipeline.ENVIRONMENT_FEATURES = base_environment + selected_soil_features
    pipeline.FEATURE_LABELS.update(SOIL_LABELS)
    shap_statistics, shap_dependence, shap_observations = (
        pipeline.shap_group_statistics(
            test_data,
            x_test.reset_index(drop=True),
            shap_values,
        )
    )
    environment_indices = [
        x_test.columns.get_loc(feature) for feature in pipeline.ENVIRONMENT_FEATURES
    ]
    shap_environment = shap_values[:, environment_indices]
    x_environment = x_test[pipeline.ENVIRONMENT_FEATURES].reset_index(drop=True)

    pipeline.save_beeswarm(
        x_environment,
        shap_environment,
        f"{soil_descriptor.title()} pooled XGBoost (independent test, n = {len(test_data):,})",
        plots / "Figure4_soil_augmented_SHAP_beeswarm.png",
    )
    group_paths = pipeline.species_beeswarm_plots(
        test_data,
        x_environment,
        shap_environment,
        species_plots,
    )
    pipeline.contact_sheet(
        group_paths,
        plots / "Figure5_soil_augmented_species_SHAP_contact_sheet.png",
        f"{soil_descriptor.title()} environmental SHAP by species/group",
    )

    soil_model.save_model(models / "xgb_period_controlled_soil_augmented.json")
    joblib.dump(
        {
            "feature_columns": soil_features,
            "feature_medians": medians[soil_model_name],
            "species_columns": encoding["species_columns"],
            "period_columns": encoding["period_columns"],
            "environment_features": pipeline.ENVIRONMENT_FEATURES,
            "soil_features": selected_soil_features,
            "soil_feature_set": args.soil_feature_set,
            "soil_parsing": {
                "infill": "normalized code starts with t (including T, T0, t0, tä)",
                "bedrock": "normalized code contains Ka",
                "moraine": "normalized code contains Mr",
                "clay": "normalized code contains Sa",
                "silt_sand": "normalized code contains S+H",
            },
        },
        models / "preprocessing_soil_augmented.joblib",
    )

    soil_counts.to_csv(tables / "soil_code_mapping_and_counts.csv", index=False)
    metrics.to_csv(tables / "xgb_base_vs_soil_performance.csv", index=False)
    bootstrap.to_csv(tables / "soil_delta_r2_cluster_bootstrap.csv", index=False)
    bootstrap_summary.to_csv(
        tables / "soil_incremental_performance_summary.csv", index=False
    )
    shap_statistics.to_csv(
        tables / "soil_augmented_shap_group_statistics.csv", index=False
    )
    shap_statistics[
        shap_statistics["Feature"].isin(selected_soil_features)
    ].to_csv(tables / "soil_only_shap_statistics.csv", index=False)
    shap_dependence.to_csv(
        tables / "soil_augmented_shap_dependence_deciles.csv", index=False
    )
    shap_observations.to_csv(
        tables / "soil_augmented_test_observation_shap.csv", index=False
    )

    actual_test = test_data["Log_Specific_Growth_Rate"].to_numpy(dtype=float)
    prediction_table = test_data[
        ["OID_", "Period", "Species_Name_Model", "Initial_Carbon", "Years"]
    ].copy()
    prediction_table["Actual_LogSGR"] = actual_test
    prediction_table["Base_Predicted_LogSGR"] = predictions[
        "Base_period_controlled_XGB"
    ]["Test"]
    prediction_table["Soil_Predicted_LogSGR"] = predictions[soil_model_name]["Test"]
    prediction_table["Soil_Residual_LogSGR_Observed_minus_Predicted"] = (
        actual_test - prediction_table["Soil_Predicted_LogSGR"]
    )
    prediction_table["Actual_Annual_Growth_Percent"] = pipeline.inverse_percentage(
        actual_test
    )
    prediction_table["Soil_Predicted_Annual_Growth_Percent"] = (
        pipeline.inverse_percentage(prediction_table["Soil_Predicted_LogSGR"])
    )
    prediction_table.to_csv(
        tables / "independent_test_predictions_and_residuals.csv", index=False
    )

    metadata = {
        "input": str(args.input),
        "pipeline_script": str(args.pipeline_script),
        "construction": construction_metadata,
        "development_rows": int(development.sum()),
        "test_rows": int(test.sum()),
        "development_trees": int(data.loc[development, "OID_"].nunique()),
        "test_trees": int(data.loc[test, "OID_"].nunique()),
        "soil_features": selected_soil_features,
        "soil_feature_set": args.soil_feature_set,
        "base_test_r2": float(base_test["R2_LogSGR"]),
        "soil_augmented_test_r2": float(soil_test["R2_LogSGR"]),
        "soil_incremental_delta_r2": delta_r2,
        "soil_partial_r2_vs_base": partial_r2,
    }
    (args.output / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    (args.output / "RUN_LOG.md").write_text(
        "\n".join(
            [
                f"# {soil_descriptor.title()} XGBoost and SHAP run",
                "",
                "- The original period-controlled pooled XGBoost specification was retained as the base comparator.",
                f"- The fitted soil block contains {len(selected_soil_features)} non-exclusive binary indicators: {', '.join(selected_soil_features)}.",
                "- Clay and silt-sand are still parsed for the code audit table but are excluded from the reduced three-soil model.",
                "- The split is identical and leakage-free at tree level (`OID_`).",
                "- SHAP was recomputed after refitting; soil values were not appended to SHAP from the older model.",
                "- Soil SHAP values are predictive associations, not causal effects.",
                f"- Base independent-test R2 (log-SGR): {float(base_test['R2_LogSGR']):.4f}",
                f"- Soil-augmented independent-test R2 (log-SGR): {float(soil_test['R2_LogSGR']):.4f}",
                f"- Soil incremental Delta R2: {delta_r2:.4f}",
                f"- Soil partial R2 versus base: {partial_r2:.4f}",
                (
                    "- Tree-cluster bootstrap 95% CI for Delta R2: "
                    f"[{float(bootstrap_summary.iloc[0]['Delta_R2_CI_Lower_2_5pct']):.4f}, "
                    f"{float(bootstrap_summary.iloc[0]['Delta_R2_CI_Upper_97_5pct']):.4f}]"
                ),
                (
                    "- The increment is metric-specific: test log-SGR RMSE "
                    f"{float(base_test['RMSE_LogSGR']):.4f} to "
                    f"{float(soil_test['RMSE_LogSGR']):.4f}; annual percentage-point "
                    f"MAE {float(base_test['MAE_Annual_Percentage_Points']):.4f} to "
                    f"{float(soil_test['MAE_Annual_Percentage_Points']):.4f}; "
                    f"kg C MAE {float(base_test['MAE_kg_C_per_tree_per_year']):.4f} "
                    f"to {float(soil_test['MAE_kg_C_per_tree_per_year']):.4f}."
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
