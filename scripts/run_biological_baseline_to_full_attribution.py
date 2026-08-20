#!/usr/bin/env python3
"""Compare a height-plus-species XGBoost baseline with the full model.

The full model is the frozen period-controlled pooled XGBoost containing
height, one-hot species, one-hot monitoring period, and the measured
environmental predictor block. The contrast therefore quantifies the combined
incremental predictive information from period plus environment.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from xgboost import XGBRegressor


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def clustered_bootstrap(
    test_data: pd.DataFrame,
    baseline_prediction: np.ndarray,
    full_prediction: np.ndarray,
    repetitions: int,
    random_state: int,
) -> pd.DataFrame:
    tree_ids = test_data["OID_"].drop_duplicates().to_numpy()
    tree_rows = {
        tree_id: np.flatnonzero(test_data["OID_"].to_numpy() == tree_id)
        for tree_id in tree_ids
    }
    actual = test_data["Log_Specific_Growth_Rate"].to_numpy(dtype=float)
    rng = np.random.default_rng(random_state)
    rows = []
    for iteration in range(1, repetitions + 1):
        sampled_ids = rng.choice(tree_ids, size=len(tree_ids), replace=True)
        sampled_rows = np.concatenate([tree_rows[tree_id] for tree_id in sampled_ids])
        y = actual[sampled_rows]
        baseline = baseline_prediction[sampled_rows]
        full = full_prediction[sampled_rows]
        baseline_r2 = r2_score(y, baseline)
        full_r2 = r2_score(y, full)
        baseline_sse = float(np.sum((y - baseline) ** 2))
        full_sse = float(np.sum((y - full) ** 2))
        rows.append(
            {
                "Iteration": iteration,
                "Biological_Baseline_Test_R2": baseline_r2,
                "Full_Model_Test_R2": full_r2,
                "Combined_Period_Environment_Delta_R2": full_r2 - baseline_r2,
                "Combined_Period_Environment_Partial_R2": 1.0
                - full_sse / baseline_sse,
            }
        )
    return pd.DataFrame(rows)


def ci(bootstrap: pd.DataFrame, column: str) -> tuple[float, float]:
    return (
        float(bootstrap[column].quantile(0.025)),
        float(bootstrap[column].quantile(0.975)),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--pipeline-script", required=True, type=Path)
    parser.add_argument("--soil-script", required=True, type=Path)
    parser.add_argument("--full-analysis", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bootstrap-repetitions", type=int, default=1500)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"Output directory already exists: {args.output}")
    tables = args.output / "tables"
    models = args.output / "models"
    tables.mkdir(parents=True)
    models.mkdir()

    pipeline = load_module(args.pipeline_script, "relative_growth_pipeline")
    soil = load_module(args.soil_script, "soil_pipeline")
    raw = pd.read_csv(args.input)
    raw, _ = soil.add_soil_indicators(raw)
    long_data, construction = pipeline.build_long_data(raw)
    data, encoding = pipeline.add_split_and_dummies(long_data)

    development = data["Split"].eq("Development")
    test = data["Split"].eq("Test")
    target = data["Log_Specific_Growth_Rate"].astype(np.float32)

    biological_features = ["Log_Height"] + list(encoding["species_columns"])
    if set(biological_features) & set(encoding["period_columns"]):
        raise AssertionError("Monitoring-period columns entered the biological baseline")
    baseline_development, baseline_test, baseline_medians = pipeline.prepare_matrices(
        data, biological_features
    )
    baseline_model = XGBRegressor(**pipeline.XGB_PARAMETERS)
    baseline_model.fit(baseline_development, target.loc[development])
    baseline_prediction = baseline_model.predict(baseline_test)

    full_preprocessing = joblib.load(
        args.full_analysis / "models/preprocessing_soil_augmented.joblib"
    )
    full_features = list(full_preprocessing["feature_columns"])
    _, full_test, _ = pipeline.prepare_matrices(data, full_features)
    full_model = XGBRegressor(**pipeline.XGB_PARAMETERS)
    full_model.load_model(
        args.full_analysis / "models/xgb_period_controlled_soil_augmented.json"
    )
    full_prediction = full_model.predict(full_test)

    test_data = data.loc[test].reset_index(drop=True)
    actual = test_data["Log_Specific_Growth_Rate"].to_numpy(dtype=float)
    baseline_r2 = float(r2_score(actual, baseline_prediction))
    full_r2 = float(r2_score(actual, full_prediction))
    delta_r2 = full_r2 - baseline_r2
    baseline_sse = float(np.sum((actual - baseline_prediction) ** 2))
    full_sse = float(np.sum((actual - full_prediction) ** 2))
    partial_r2 = 1.0 - full_sse / baseline_sse

    bootstrap = clustered_bootstrap(
        test_data,
        baseline_prediction,
        full_prediction,
        args.bootstrap_repetitions,
        pipeline.RANDOM_STATE + 46000,
    )
    baseline_ci = ci(bootstrap, "Biological_Baseline_Test_R2")
    full_ci = ci(bootstrap, "Full_Model_Test_R2")
    delta_ci = ci(bootstrap, "Combined_Period_Environment_Delta_R2")
    partial_ci = ci(bootstrap, "Combined_Period_Environment_Partial_R2")

    summary = pd.DataFrame(
        [
            {
                "Contrast": "Height + species biological baseline vs full period-controlled environmental model",
                "Development_Observations": int(development.sum()),
                "Test_Observations": int(test.sum()),
                "Development_Trees": int(data.loc[development, "OID_"].nunique()),
                "Test_Trees": int(data.loc[test, "OID_"].nunique()),
                "Biological_Baseline_Feature_Count": len(biological_features),
                "Full_Model_Feature_Count": len(full_features),
                "Biological_Baseline_Test_R2": baseline_r2,
                "Biological_Baseline_Test_R2_CI95_Lower": baseline_ci[0],
                "Biological_Baseline_Test_R2_CI95_Upper": baseline_ci[1],
                "Full_Model_Test_R2": full_r2,
                "Full_Model_Test_R2_CI95_Lower": full_ci[0],
                "Full_Model_Test_R2_CI95_Upper": full_ci[1],
                "Combined_Period_Environment_Delta_R2": delta_r2,
                "Combined_Period_Environment_Delta_R2_CI95_Lower": delta_ci[0],
                "Combined_Period_Environment_Delta_R2_CI95_Upper": delta_ci[1],
                "Combined_Period_Environment_Partial_R2": partial_r2,
                "Combined_Period_Environment_Partial_R2_CI95_Lower": partial_ci[0],
                "Combined_Period_Environment_Partial_R2_CI95_Upper": partial_ci[1],
                "Bootstrap_Repetitions": args.bootstrap_repetitions,
                "Bootstrap_Cluster": "OID_",
                "Probability_Delta_R2_Positive": float(
                    (bootstrap["Combined_Period_Environment_Delta_R2"] > 0).mean()
                ),
            }
        ]
    )

    predictions = test_data[
        ["OID_", "Period", "Species_Name_Model", "Log_Height", "Log_Specific_Growth_Rate"]
    ].copy()
    predictions = predictions.rename(
        columns={"Log_Specific_Growth_Rate": "Actual_LogSGR"}
    )
    predictions["Biological_Baseline_Predicted_LogSGR"] = baseline_prediction
    predictions["Full_Model_Predicted_LogSGR"] = full_prediction
    predictions["Biological_Baseline_Residual"] = (
        actual - baseline_prediction
    )
    predictions["Full_Model_Residual"] = actual - full_prediction

    summary.to_csv(tables / "biological_baseline_vs_full_summary.csv", index=False)
    bootstrap.to_csv(
        tables / "biological_baseline_vs_full_tree_bootstrap_draws.csv", index=False
    )
    predictions.to_csv(
        tables / "biological_baseline_vs_full_test_predictions.csv", index=False
    )
    baseline_model.save_model(models / "xgb_height_species_biological_baseline.json")
    joblib.dump(
        {
            "feature_columns": biological_features,
            "feature_medians": baseline_medians,
            "species_columns": encoding["species_columns"],
            "period_columns_excluded": encoding["period_columns"],
            "split_group": "OID_",
        },
        models / "biological_baseline_preprocessing.joblib",
    )
    metadata = {
        "input": str(args.input),
        "pipeline_script": str(args.pipeline_script),
        "full_analysis": str(args.full_analysis),
        "target": "Log_Specific_Growth_Rate",
        "biological_baseline": ["Log_Height", "one-hot species"],
        "full_model": [
            "Log_Height",
            "one-hot species",
            "one-hot monitoring period",
            "measured environmental predictors",
        ],
        "split_group": "OID_",
        "construction": construction,
        "xgb_parameters": pipeline.XGB_PARAMETERS,
    }
    (args.output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str), encoding="utf-8"
    )

    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
