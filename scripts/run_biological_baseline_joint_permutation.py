#!/usr/bin/env python3
"""Joint-permutation tests aligned with the biological-baseline comparison.

Two complementary tests are reported:

1. Environmental block only, jointly permuted within species-period strata.
   This preserves species and monitoring period and matches the established
   environmental-block analysis.
2. Monitoring-period plus environmental block, jointly permuted within
   species strata. This matches the added block in the comparison between the
   height-plus-species biological baseline and the retrospective full model.

All columns in a tested block are moved together from the same donor row, so
the observed relationships among the permuted predictors are retained.
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


def joint_permutation(
    *,
    model: XGBRegressor,
    test_matrix: pd.DataFrame,
    test_data: pd.DataFrame,
    actual: np.ndarray,
    full_r2: float,
    block_columns: list[str],
    strata_columns: list[str],
    repetitions: int,
    random_state: int,
) -> tuple[pd.DataFrame, dict]:
    """Jointly move an entire predictor block among rows inside strata."""
    base_matrix = test_matrix.reset_index(drop=True)
    strata = test_data.groupby(strata_columns, observed=True, sort=True).indices
    rng = np.random.default_rng(random_state)
    rows: list[dict] = []
    for repetition in range(1, repetitions + 1):
        permuted = base_matrix.copy()
        for row_indices in strata.values():
            source_indices = rng.permutation(row_indices)
            permuted.loc[row_indices, block_columns] = base_matrix.loc[
                source_indices, block_columns
            ].to_numpy()
        prediction = model.predict(permuted)
        permuted_r2 = float(r2_score(actual, prediction))
        rows.append(
            {
                "Repetition": repetition,
                "Full_Test_R2": full_r2,
                "Permuted_Test_R2": permuted_r2,
                "R2_Loss": full_r2 - permuted_r2,
            }
        )
    draws = pd.DataFrame(rows)
    sizes = np.asarray([len(indices) for indices in strata.values()], dtype=int)
    diagnostics = {
        "strata_count": int(len(strata)),
        "minimum_stratum_rows": int(sizes.min()),
        "median_stratum_rows": float(np.median(sizes)),
        "maximum_stratum_rows": int(sizes.max()),
        "singleton_strata": int((sizes == 1).sum()),
    }
    return draws, diagnostics


def summarize(
    *,
    contrast: str,
    permuted_block: str,
    stratification: str,
    feature_count: int,
    full_r2: float,
    direct_delta_r2: float,
    direct_partial_r2: float,
    draws: pd.DataFrame,
    diagnostics: dict,
) -> dict:
    loss = draws["R2_Loss"]
    return {
        "Contrast": contrast,
        "Permuted_Block": permuted_block,
        "Stratification": stratification,
        "Permuted_Feature_Count": feature_count,
        "Full_Test_R2": full_r2,
        "Direct_Nested_Delta_R2": direct_delta_r2,
        "Direct_Nested_Partial_R2": direct_partial_r2,
        "Permutation_Mean_R2_Loss": float(loss.mean()),
        "Permutation_Median_R2_Loss": float(loss.median()),
        "Permutation_SD_R2_Loss": float(loss.std(ddof=1)),
        "Permutation_P02_5_R2_Loss": float(loss.quantile(0.025)),
        "Permutation_P97_5_R2_Loss": float(loss.quantile(0.975)),
        "Permutation_Min_R2_Loss": float(loss.min()),
        "Permutation_Max_R2_Loss": float(loss.max()),
        "Permutation_Repetitions": int(len(draws)),
        "Probability_R2_Loss_Positive": float((loss > 0).mean()),
        "Strata_Count": diagnostics["strata_count"],
        "Minimum_Stratum_Rows": diagnostics["minimum_stratum_rows"],
        "Median_Stratum_Rows": diagnostics["median_stratum_rows"],
        "Maximum_Stratum_Rows": diagnostics["maximum_stratum_rows"],
        "Singleton_Strata": diagnostics["singleton_strata"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--pipeline-script", required=True, type=Path)
    parser.add_argument("--soil-script", required=True, type=Path)
    parser.add_argument("--full-analysis", required=True, type=Path)
    parser.add_argument("--biological-analysis", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--permutation-repetitions", type=int, default=300)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"Output directory already exists: {args.output}")
    tables = args.output / "tables"
    tables.mkdir(parents=True)

    pipeline = load_module(args.pipeline_script, "relative_growth_pipeline")
    soil = load_module(args.soil_script, "soil_pipeline")
    raw = pd.read_csv(args.input)
    raw, _ = soil.add_soil_indicators(raw)
    long_data, construction = pipeline.build_long_data(raw)
    data, encoding = pipeline.add_split_and_dummies(long_data)

    preprocessing = joblib.load(
        args.full_analysis / "models/preprocessing_soil_augmented.joblib"
    )
    full_features = list(preprocessing["feature_columns"])
    _, full_test, _ = pipeline.prepare_matrices(data, full_features)
    full_model = XGBRegressor(**pipeline.XGB_PARAMETERS)
    full_model.load_model(
        args.full_analysis / "models/xgb_period_controlled_soil_augmented.json"
    )

    test_data = data.loc[data["Split"].eq("Test")].reset_index(drop=True)
    actual = test_data["Log_Specific_Growth_Rate"].to_numpy(dtype=float)
    full_prediction = full_model.predict(full_test)
    full_r2 = float(r2_score(actual, full_prediction))

    species_columns = [
        column for column in encoding["species_columns"] if column in full_features
    ]
    period_columns = [
        column for column in encoding["period_columns"] if column in full_features
    ]
    environmental_columns = [
        column
        for column in list(pipeline.ENVIRONMENT_FEATURES)
        + list(soil.REDUCED_SOIL_FEATURES)
        if column in full_features
    ]
    biological_columns = ["Log_Height"] + species_columns
    combined_columns = period_columns + environmental_columns
    expected = biological_columns + combined_columns
    if expected != full_features:
        raise AssertionError(
            "Full feature specification does not equal biological plus added blocks"
        )

    environmental_nested = pd.read_csv(
        args.full_analysis / "tables/three_soil_environmental_attribution.csv"
    ).iloc[0]
    biological_nested = pd.read_csv(
        args.biological_analysis
        / "tables/biological_baseline_vs_full_summary.csv"
    ).iloc[0]

    environmental_draws, environmental_diagnostics = joint_permutation(
        model=full_model,
        test_matrix=full_test,
        test_data=test_data,
        actual=actual,
        full_r2=full_r2,
        block_columns=environmental_columns,
        strata_columns=["Species_Name_Model", "Period"],
        repetitions=args.permutation_repetitions,
        random_state=pipeline.RANDOM_STATE + 2000,
    )
    combined_draws, combined_diagnostics = joint_permutation(
        model=full_model,
        test_matrix=full_test,
        test_data=test_data,
        actual=actual,
        full_r2=full_r2,
        block_columns=combined_columns,
        strata_columns=["Species_Name_Model"],
        repetitions=args.permutation_repetitions,
        random_state=pipeline.RANDOM_STATE + 52000,
    )

    summary = pd.DataFrame(
        [
            summarize(
                contrast="Environment conditional on species and monitoring period",
                permuted_block="Measured environment",
                stratification="Species_Name_Model + Period",
                feature_count=len(environmental_columns),
                full_r2=full_r2,
                direct_delta_r2=float(environmental_nested["Incremental_Delta_R2"]),
                direct_partial_r2=float(
                    environmental_nested["Environmental_Partial_R2"]
                ),
                draws=environmental_draws,
                diagnostics=environmental_diagnostics,
            ),
            summarize(
                contrast="Added block relative to height-species biological baseline",
                permuted_block="Monitoring period + measured environment",
                stratification="Species_Name_Model",
                feature_count=len(combined_columns),
                full_r2=full_r2,
                direct_delta_r2=float(
                    biological_nested["Combined_Period_Environment_Delta_R2"]
                ),
                direct_partial_r2=float(
                    biological_nested["Combined_Period_Environment_Partial_R2"]
                ),
                draws=combined_draws,
                diagnostics=combined_diagnostics,
            ),
        ]
    )

    environmental_draws.to_csv(
        tables / "environmental_block_joint_permutation_draws.csv", index=False
    )
    combined_draws.to_csv(
        tables / "period_environment_block_joint_permutation_draws.csv", index=False
    )
    summary.to_csv(tables / "joint_permutation_summary.csv", index=False)

    metadata = {
        "input": str(args.input),
        "pipeline_script": str(args.pipeline_script),
        "soil_script": str(args.soil_script),
        "full_analysis": str(args.full_analysis),
        "biological_analysis": str(args.biological_analysis),
        "target": "Log_Specific_Growth_Rate",
        "full_test_r2": full_r2,
        "test_observations": int(len(test_data)),
        "test_trees": int(test_data["OID_"].nunique()),
        "environmental_columns": environmental_columns,
        "period_columns": period_columns,
        "biological_columns_retained": biological_columns,
        "environmental_test": {
            "permuted_block": environmental_columns,
            "strata": ["Species_Name_Model", "Period"],
            "random_state": pipeline.RANDOM_STATE + 2000,
        },
        "combined_test": {
            "permuted_block": combined_columns,
            "strata": ["Species_Name_Model"],
            "random_state": pipeline.RANDOM_STATE + 52000,
        },
        "permutation_repetitions": args.permutation_repetitions,
        "construction": construction,
        "joint_permutation_note": (
            "All columns in each tested block were moved together from the same "
            "donor row within the stated strata, preserving within-block predictor "
            "relationships. The reported 2.5th-97.5th percentile interval is the "
            "permutation distribution, not a sampling confidence interval."
        ),
    }
    (args.output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str), encoding="utf-8"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
