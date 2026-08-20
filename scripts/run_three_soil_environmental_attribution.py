#!/usr/bin/env python3
"""Recompute nested environmental attribution for the three-soil XGBoost."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import joblib
import pandas as pd
from xgboost import XGBRegressor


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--pipeline-script", required=True, type=Path)
    parser.add_argument("--soil-script", required=True, type=Path)
    parser.add_argument("--analysis-dir", required=True, type=Path)
    parser.add_argument("--permutation-repetitions", type=int, default=300)
    parser.add_argument("--bootstrap-repetitions", type=int, default=1500)
    args = parser.parse_args()

    pipeline = load_module(args.pipeline_script, "relative_growth_pipeline")
    soil = load_module(args.soil_script, "soil_pipeline")
    raw = pd.read_csv(args.input)
    raw, _ = soil.add_soil_indicators(raw)
    long_data, _ = pipeline.build_long_data(raw)
    data, encoding = pipeline.add_split_and_dummies(long_data)

    preprocessing = joblib.load(
        args.analysis_dir / "models/preprocessing_soil_augmented.joblib"
    )
    features = list(preprocessing["feature_columns"])
    _, x_test, _ = pipeline.prepare_matrices(data, features)
    full_model = XGBRegressor(**pipeline.XGB_PARAMETERS)
    full_model.load_model(
        args.analysis_dir / "models/xgb_period_controlled_soil_augmented.json"
    )
    full_prediction = full_model.predict(x_test)

    base_environment = list(pipeline.ENVIRONMENT_FEATURES)
    pipeline.ENVIRONMENT_FEATURES = base_environment + list(
        soil.REDUCED_SOIL_FEATURES
    )
    baseline_columns = (
        ["Log_Height"]
        + list(encoding["species_columns"])
        + list(encoding["period_columns"])
    )
    summary, permutation, baseline_model, baseline_prediction = (
        pipeline.environmental_attribution(
            data,
            full_model,
            x_test,
            full_prediction,
            baseline_columns,
            args.permutation_repetitions,
        )
    )
    test_data = data.loc[data["Split"].eq("Test")].reset_index(drop=True)
    bootstrap = pipeline.cluster_bootstrap_environmental_attribution(
        test_data,
        baseline_prediction,
        full_prediction,
        args.bootstrap_repetitions,
    )
    for metric in [
        "Baseline_Test_R2",
        "Full_Test_R2",
        "Incremental_Delta_R2",
        "Environmental_Partial_R2",
    ]:
        summary[f"{metric}_CI95_Lower"] = bootstrap[metric].quantile(0.025)
        summary[f"{metric}_CI95_Upper"] = bootstrap[metric].quantile(0.975)

    tables = args.analysis_dir / "tables"
    summary.to_csv(tables / "three_soil_environmental_attribution.csv", index=False)
    permutation.to_csv(
        tables / "three_soil_environmental_block_permutation_draws.csv", index=False
    )
    bootstrap.to_csv(
        tables / "three_soil_environmental_attribution_tree_bootstrap_draws.csv",
        index=False,
    )
    baseline_model.save_model(
        args.analysis_dir / "models/xgb_biological_temporal_baseline.json"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
