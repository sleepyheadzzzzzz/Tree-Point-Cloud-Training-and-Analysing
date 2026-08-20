#!/usr/bin/env python3
"""Compare OLS, RF, XGBoost, and MLP with the reduced three-soil block."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import joblib
import pandas as pd


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--pipeline-script", required=True, type=Path)
    parser.add_argument("--soil-script", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bootstrap-repetitions", type=int, default=1000)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"Output directory already exists: {args.output}")
    tables = args.output / "tables"
    plots = args.output / "plots"
    models = args.output / "models"
    for directory in (tables, plots, models):
        directory.mkdir(parents=True, exist_ok=False)

    pipeline = load_module(args.pipeline_script, "relative_growth_pipeline")
    soil = load_module(args.soil_script, "soil_augmented_pipeline")

    raw = pd.read_csv(args.input)
    raw, soil_counts = soil.add_soil_indicators(raw)
    long_data, construction = pipeline.build_long_data(raw)
    data, encoding = pipeline.add_split_and_dummies(long_data)
    features = (
        ["Log_Height"]
        + list(encoding["species_columns"])
        + list(encoding["period_columns"])
        + list(pipeline.ENVIRONMENT_FEATURES)
        + list(soil.REDUCED_SOIL_FEATURES)
    )

    fitted, predictions, medians, context = pipeline.fit_models(data, features)
    metrics = pipeline.evaluate_models(data, predictions, "Period-controlled")
    compact = pipeline.compact_performance_table(metrics)
    pipeline.save_performance_table_plot(
        compact,
        plots / "three_soil_model_comparison_relative_growth.png",
    )

    test_data = data.loc[data["Split"].eq("Test")].reset_index(drop=True)
    bootstrap, paired = pipeline.cluster_bootstrap_r2(
        test_data,
        predictions,
        repetitions=args.bootstrap_repetitions,
    )
    point_estimates = {
        model: float(
            metrics.loc[
                metrics["Split"].eq("Test")
                & metrics["Group"].eq("Overall")
                & metrics["Model"].eq(model),
                "R2_LogSGR",
            ].iloc[0]
        )
        for model in ["OLS", "RF", "XGB", "MLP"]
    }
    bootstrap_summary = pipeline.bootstrap_summary_plot(
        bootstrap,
        point_estimates,
        plots / "three_soil_model_comparison_test_r2_ci.png",
    )

    encoding["split_map"].to_csv(tables / "tree_split.csv", index=False)
    soil_counts.to_csv(tables / "soil_code_mapping_and_counts.csv", index=False)
    metrics.to_csv(tables / "model_performance_all_groups_three_soil.csv", index=False)
    compact.to_csv(tables / "manuscript_model_comparison_three_soil.csv", index=False)
    metrics.loc[metrics["Group"].eq("Overall")].to_csv(
        tables / "overall_model_performance_three_soil.csv", index=False
    )
    bootstrap.to_csv(tables / "test_r2_cluster_bootstrap_draws.csv", index=False)
    paired.to_csv(tables / "xgb_vs_rf_paired_bootstrap.csv", index=False)
    bootstrap_summary.to_csv(
        tables / "test_r2_cluster_bootstrap_summary.csv", index=False
    )

    fitted["XGB"].save_model(models / "xgb_period_controlled_three_soil.json")
    joblib.dump(
        {
            "feature_columns": features,
            "feature_medians": medians,
            "species_columns": encoding["species_columns"],
            "period_columns": encoding["period_columns"],
            "environment_features": list(pipeline.ENVIRONMENT_FEATURES)
            + list(soil.REDUCED_SOIL_FEATURES),
            "soil_features": list(soil.REDUCED_SOIL_FEATURES),
            "scaler": context["scaler"],
        },
        models / "preprocessing_period_controlled_three_soil.joblib",
    )

    metadata = {
        "input": str(args.input),
        "target": "log annualized specific carbon growth (log-SGR)",
        "split": "85% development / 15% independent test, grouped by OID_",
        "construction": construction,
        "soil_features": list(soil.REDUCED_SOIL_FEATURES),
        "clay_and_silt_sand_excluded": True,
        "development_rows": int(data["Split"].eq("Development").sum()),
        "test_rows": int(data["Split"].eq("Test").sum()),
        "development_trees": int(
            data.loc[data["Split"].eq("Development"), "OID_"].nunique()
        ),
        "test_trees": int(data.loc[data["Split"].eq("Test"), "OID_"].nunique()),
        "test_r2": point_estimates,
        "fit_seconds": context["timings_seconds"],
    }
    (args.output / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    (args.output / "RUN_LOG.md").write_text(
        "\n".join(
            [
                "# Three-soil model comparison",
                "",
                "- OLS, RF, XGBoost and MLP were refitted on identical observations and predictors.",
                "- The split is leakage-free at tree level (`OID_`) and matches the primary pipeline.",
                "- Period controls and one-hot species indicators were retained.",
                "- The soil block contains only infill, bedrock and moraine; clay and silt-sand were excluded.",
                "- R2, RMSE and MAE are reported on log-SGR; percentage-point metrics are explicitly back-transformed.",
                "- kg C/tree/year metrics use each observation's initial carbon stock.",
                "",
                *[f"- {model} independent-test R2: {score:.4f}" for model, score in point_estimates.items()],
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
