#!/usr/bin/env python3
"""Audit the research project without importing the scientific stack."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTED = json.loads((ROOT / "tests/expected_results.json").read_text(encoding="utf-8"))

REQUIRED_SCRIPTS = {
    "run_relative_growth_pipeline_v2.py",
    "run_three_soil_model_comparison.py",
    "run_soil_augmented_shap_analysis.py",
    "run_three_soil_environmental_attribution.py",
    "run_biological_baseline_to_full_attribution.py",
    "run_biological_baseline_joint_permutation.py",
    "run_shap_dependence_interaction_plots.py",
    "run_nonlinear_shap_dependency_analysis.py",
    "create_radial_shap_diagram.py",
    "create_species_categorized_shap_profiles.py",
    "run_spatial_block_validation_three_soil.py",
    "evaluate_suitability_diagnostic_schemes.py",
    "run_spatial_relative_growth_diagnosis_v2.py",
    "validate_spatial_relative_growth_outputs.py",
    "create_noise40_input_raster.py",
    "create_spatial_figure6.py",
    "create_spatial_change_detection.py",
    "validate_spatial_change_detection.py",
    "create_environment_change_figure_with_noise.py",
    "render_environment_change_figure_with_noise.py",
    "export_suitable_genus_count.py",
    "create_species_densification_plan.py",
    "geojson_to_shapefile_libpysal.py",
    "create_clean_unobscured_figures.py",
}

REQUIRED_MODELS = {
    "xgb_retrospective_period_controlled_three_soil.json",
    "preprocessing_retrospective_three_soil.joblib",
    "xgb_spatial_deployment_no_period_three_soil.json",
    "preprocessing_spatial_deployment.joblib",
    "xgb_height_species_biological_baseline.json",
    "preprocessing_biological_baseline.joblib",
}


def read_rows(relative: str) -> list[dict[str, str]]:
    with (ROOT / relative).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def close(actual: float, expected: float, tolerance: float = 1e-9) -> None:
    if abs(actual - expected) > tolerance:
        raise AssertionError(f"Expected {expected:.12g}; found {actual:.12g}")


def audit_files() -> None:
    scripts = {path.name for path in (ROOT / "scripts").glob("*.py")}
    missing_scripts = REQUIRED_SCRIPTS - scripts
    if missing_scripts:
        raise AssertionError(f"Missing scripts: {sorted(missing_scripts)}")

    models = {path.name for path in (ROOT / "models").iterdir() if path.is_file()}
    missing_models = REQUIRED_MODELS - models
    if missing_models:
        raise AssertionError(f"Missing model artifacts: {sorted(missing_models)}")

    for script in sorted((ROOT / "scripts").glob("*.py")):
        source = script.read_text(encoding="utf-8")
        compile(source, str(script), "exec")
        if re.search(r"(?i)(?:^|[\"'])[a-z]:[\\/]", source):
            raise AssertionError(f"Hard-coded drive path in {script.name}")

    for model in (ROOT / "models").glob("*.json"):
        payload = json.loads(model.read_text(encoding="utf-8"))
        feature_names = payload["learner"]["feature_names"]
        if model.name == "xgb_spatial_deployment_no_period_three_soil.json":
            if any(name.startswith("Period_") for name in feature_names):
                raise AssertionError("Deployment model unexpectedly contains period features")
            required = {"soil_infill", "soil_bedrock", "soil_moraine"}
            if not required.issubset(feature_names):
                raise AssertionError("Deployment model is missing a required soil indicator")


def audit_results() -> None:
    comparison = read_rows("results/model_comparison/manuscript_model_comparison_three_soil.csv")
    overall = next(row for row in comparison if row["Group"] == "Overall")
    for model, expected in EXPECTED["model_test_r2"].items():
        actual = float(overall[f"{model}_R2_Development_Test"].split("/")[1].strip())
        close(actual, expected, tolerance=5e-4)

    attribution = read_rows("results/attribution/biological_baseline_vs_full_summary.csv")[0]
    close(float(attribution["Biological_Baseline_Test_R2"]), EXPECTED["biological_baseline_test_r2"])
    close(float(attribution["Full_Model_Test_R2"]), EXPECTED["full_model_test_r2"])
    close(float(attribution["Combined_Period_Environment_Delta_R2"]), EXPECTED["combined_period_environment_delta_r2"])
    close(float(attribution["Combined_Period_Environment_Partial_R2"]), EXPECTED["combined_period_environment_partial_r2"])
    if attribution["Bootstrap_Cluster"] != "OID_":
        raise AssertionError("Attribution bootstrap is not clustered by OID_")

    permutation = read_rows("results/attribution/joint_permutation_summary.csv")
    environment = next(row for row in permutation if row["Permuted_Block"] == "Measured environment")
    combined = next(row for row in permutation if row["Permuted_Block"] == "Monitoring period + measured environment")
    close(float(environment["Direct_Nested_Delta_R2"]), EXPECTED["environment_only_delta_r2"])
    close(float(environment["Direct_Nested_Partial_R2"]), EXPECTED["environment_only_partial_r2"])
    close(float(environment["Permutation_Mean_R2_Loss"]), EXPECTED["environment_only_permutation_loss"])
    close(float(combined["Permutation_Mean_R2_Loss"]), EXPECTED["combined_period_environment_permutation_loss"])

    spatial = read_rows("results/spatial_validation/locked_test_selected_model_metrics.csv")[0]
    close(float(spatial["R2_LogSGR"]), EXPECTED["spatial_locked_test_r2"])
    close(float(spatial["RMSE_LogSGR"]), EXPECTED["spatial_locked_test_rmse_log_sgr"])
    close(float(spatial["MAE_LogSGR"]), EXPECTED["spatial_locked_test_mae_log_sgr"])

    agreement = read_rows("results/suitability/locked_test_diagnostic_agreement.csv")
    seven = next(row for row in agreement if row["Output"] == "Seven-level detail")
    zones = next(row for row in agreement if row["Output"] == "Three-zone diagnosis")
    close(float(seven["Exact_Accuracy"]), EXPECTED["seven_level_exact_accuracy"])
    close(float(seven["Within_One_Accuracy"]), EXPECTED["seven_level_within_one_accuracy"])
    close(float(zones["Exact_Accuracy"]), EXPECTED["three_zone_exact_accuracy"])

    noise = read_rows("results/change_detection/daytime_noise_change_summary.csv")[0]
    close(float(noise["Mean"]), EXPECTED["noise_change_mean_db"])


def write_manifest() -> Path:
    target = ROOT / "tests/MANIFEST.sha256"
    paths = sorted(
        path for path in ROOT.rglob("*")
        if path.is_file()
        and path != target
        and "__pycache__" not in path.parts
        and ".git" not in path.parts
    )
    lines = []
    for path in paths:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.relative_to(ROOT).as_posix()}")
    target.write_bytes(("\n".join(lines) + "\n").encode("utf-8"))
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-manifest", action="store_true")
    args = parser.parse_args()
    audit_files()
    audit_results()
    if args.write_manifest:
        manifest = write_manifest()
        print(f"Wrote {manifest.relative_to(ROOT)}")
    print("Project audit passed: scripts compile, model JSON parses, and archived results match expected values.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
