#!/usr/bin/env python3
"""Redesign suitability levels for spatial diagnosis without test-set tuning.

Candidate level schemes are compared on the original spatial validation blocks.
The best seven-level scheme for a location/tree diagnostic is selected there,
then applied once to the existing locked spatial-test predictions. A three-zone
action label is obtained by collapsing the selected seven levels (1-2, 3-5,
6-7); the seven-level score remains available as the detailed sublevel.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix, r2_score
from xgboost import XGBRegressor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--pipeline-script", required=True, type=Path)
    parser.add_argument("--soil-script", required=True, type=Path)
    parser.add_argument("--split-table", required=True, type=Path)
    parser.add_argument("--locked-test-predictions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bootstrap-repetitions", type=int, default=1000)
    return parser.parse_args()


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def add_species_dummies(data: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    dummies = pd.get_dummies(data["Species_Name_Model"], prefix="Species", dtype=np.float32)
    result = pd.concat([data.reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)
    return result, dummies.columns.tolist()


def prepare_unit(data: pd.DataFrame, unit: str) -> pd.DataFrame:
    columns = ["OID_", "Species_Name_Model", "Actual_Annual_Growth_Percent"]
    if "Predicted_Annual_Growth_Percent" in data.columns:
        columns.append("Predicted_Annual_Growth_Percent")
    working = data[columns].copy()
    if unit == "row":
        return working.reset_index(drop=True)
    if unit != "tree":
        raise ValueError(unit)
    aggregation: dict[str, tuple[str, str]] = {
        "Species_Name_Model": ("Species_Name_Model", "first"),
        "Actual_Annual_Growth_Percent": ("Actual_Annual_Growth_Percent", "mean"),
    }
    if "Predicted_Annual_Growth_Percent" in working.columns:
        aggregation["Predicted_Annual_Growth_Percent"] = (
            "Predicted_Annual_Growth_Percent", "mean"
        )
    return working.groupby("OID_", as_index=False).agg(**aggregation)


def derive_thresholds(reference: pd.DataFrame, levels: int, scope: str) -> pd.DataFrame:
    quantiles = np.arange(1, levels) / levels
    rows: list[dict[str, object]] = []
    if scope == "global":
        values = reference["Actual_Annual_Growth_Percent"].to_numpy(float)
        for number, value in enumerate(np.quantile(values, quantiles), start=1):
            rows.append({"Threshold_Group": "Overall", "Threshold_Number": number, "Threshold": float(value)})
    elif scope == "species":
        for species, subset in reference.groupby("Species_Name_Model", sort=True):
            values = subset["Actual_Annual_Growth_Percent"].to_numpy(float)
            if len(values) < levels * 10:
                raise ValueError(f"Too few reference units for {species}: {len(values)}")
            for number, value in enumerate(np.quantile(values, quantiles), start=1):
                rows.append({"Threshold_Group": species, "Threshold_Number": number, "Threshold": float(value)})
    else:
        raise ValueError(scope)
    return pd.DataFrame(rows)


def assign_levels(values: np.ndarray, species: np.ndarray, thresholds: pd.DataFrame, scope: str) -> np.ndarray:
    result = np.empty(len(values), dtype=int)
    if scope == "global":
        cuts = thresholds.loc[thresholds["Threshold_Group"].eq("Overall"), "Threshold"].to_numpy(float)
        return np.digitize(values, cuts, right=False) + 1
    for name in np.unique(species):
        mask = species == name
        cuts = thresholds.loc[thresholds["Threshold_Group"].eq(name), "Threshold"].to_numpy(float)
        if len(cuts) == 0:
            raise KeyError(f"No thresholds for species {name}")
        result[mask] = np.digitize(values[mask], cuts, right=False) + 1
    return result


def agreement(actual: np.ndarray, predicted: np.ndarray, levels: int) -> dict[str, float]:
    difference = np.abs(actual - predicted)
    return {
        "N_Units": int(len(actual)),
        "Exact_Accuracy": float(np.mean(difference == 0)),
        "Within_One_Accuracy": float(np.mean(difference <= 1)),
        "MAE_Levels": float(np.mean(difference)),
        "Quadratic_Weighted_Kappa": float(cohen_kappa_score(actual, predicted, weights="quadratic", labels=np.arange(1, levels + 1))),
    }


def zone_from_seven(level: np.ndarray) -> np.ndarray:
    # Action-oriented hierarchy: detailed levels remain; zones are the primary diagnostic.
    return np.where(level <= 2, 1, np.where(level <= 5, 2, 3)).astype(int)


def validation_candidates(training: pd.DataFrame, validation: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for levels in [7, 5, 3]:
        for unit in ["row", "tree"]:
            train_unit = prepare_unit(training, unit)
            valid_unit = prepare_unit(validation, unit)
            for scope in ["global", "species"]:
                thresholds = derive_thresholds(train_unit, levels, scope)
                actual_level = assign_levels(
                    valid_unit["Actual_Annual_Growth_Percent"].to_numpy(float),
                    valid_unit["Species_Name_Model"].to_numpy(str), thresholds, scope,
                )
                predicted_level = assign_levels(
                    valid_unit["Predicted_Annual_Growth_Percent"].to_numpy(float),
                    valid_unit["Species_Name_Model"].to_numpy(str), thresholds, scope,
                )
                rows.append({
                    "Levels": levels,
                    "Evaluation_Unit": unit,
                    "Threshold_Scope": scope,
                    **agreement(actual_level, predicted_level, levels),
                })
    return pd.DataFrame(rows)


def bootstrap_agreement(actual: np.ndarray, predicted: np.ndarray, repetitions: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for iteration in range(repetitions):
        index = rng.integers(0, len(actual), size=len(actual))
        rows.append({"Iteration": iteration + 1, **agreement(actual[index], predicted[index], int(max(actual.max(), predicted.max())))})
    return pd.DataFrame(rows)


def bootstrap_summary(draws: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in ["Exact_Accuracy", "Within_One_Accuracy", "MAE_Levels", "Quadratic_Weighted_Kappa"]:
        rows.append({
            "Metric": column,
            "Median": float(draws[column].median()),
            "CI95_Lower": float(draws[column].quantile(0.025)),
            "CI95_Upper": float(draws[column].quantile(0.975)),
        })
    return pd.DataFrame(rows)


def save_figure(validation: pd.DataFrame, seven_confusion: np.ndarray, zone_confusion: np.ndarray, test_metrics: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16.2, 5.0), constrained_layout=True)
    candidates = validation.loc[validation["Evaluation_Unit"].eq("tree")].copy()
    candidates["Label"] = candidates.apply(lambda row: f"{int(row['Levels'])} levels\n{row['Threshold_Scope']}", axis=1)
    x = np.arange(len(candidates))
    axes[0].bar(x - 0.18, candidates["Exact_Accuracy"] * 100, width=0.36, color="#2C7FB8", label="Exact")
    axes[0].bar(x + 0.18, candidates["Within_One_Accuracy"] * 100, width=0.36, color="#7FCDBB", label="Within ±1")
    axes[0].set_xticks(x, candidates["Label"], fontsize=8)
    axes[0].set_ylim(0, 100)
    axes[0].set_ylabel("Validation agreement (%)")
    axes[0].set_title("A  Validation-only scheme comparison", loc="left", fontweight="bold")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(axis="y", color="#D9DEE3", linewidth=0.6)

    def heatmap(ax: plt.Axes, matrix: np.ndarray, title: str, labels: list[str]) -> None:
        normalized = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
        image = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=max(0.01, normalized.max()))
        for row in range(len(labels)):
            for column in range(len(labels)):
                value = normalized[row, column]
                if value >= 0.06:
                    ax.text(column, row, f"{value:.2f}", ha="center", va="center", fontsize=7,
                            color="white" if value > 0.55 * normalized.max() else "#1E2A35")
        ax.set_xticks(range(len(labels)), labels)
        ax.set_yticks(range(len(labels)), labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Observed")
        ax.set_title(title, loc="left", fontweight="bold")
        fig.colorbar(image, ax=ax, shrink=0.78, label="Row proportion")

    heatmap(axes[1], seven_confusion, "B  Selected seven-level test agreement", [str(i) for i in range(1, 8)])
    heatmap(axes[2], zone_confusion, "C  Three-zone diagnostic agreement", ["Constrained", "Typical", "Favorable"])
    selected = test_metrics.set_index("Output")
    axes[1].text(0.03, 0.97,
                 f"exact = {selected.loc['Seven-level detail', 'Exact_Accuracy']:.1%}\n"
                 f"within ±1 = {selected.loc['Seven-level detail', 'Within_One_Accuracy']:.1%}\n"
                 f"κw = {selected.loc['Seven-level detail', 'Quadratic_Weighted_Kappa']:.3f}",
                 transform=axes[1].transAxes, va="top", fontsize=8,
                 bbox={"facecolor": "white", "edgecolor": "#C7CDD2", "alpha": 0.9})
    axes[2].text(0.03, 0.97,
                 f"exact zone = {selected.loc['Three-zone diagnosis', 'Exact_Accuracy']:.1%}\n"
                 f"κw = {selected.loc['Three-zone diagnosis', 'Quadratic_Weighted_Kappa']:.3f}",
                 transform=axes[2].transAxes, va="top", fontsize=8,
                 bbox={"facecolor": "white", "edgecolor": "#C7CDD2", "alpha": 0.9})
    fig.suptitle("Validation-selected suitability hierarchy for spatial diagnosis", fontsize=15, fontweight="bold")
    fig.savefig(output / "Figure_suitability_diagnostic_redesign.png", dpi=400, bbox_inches="tight")
    fig.savefig(output / "Figure_suitability_diagnostic_redesign.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    tables = args.output / "tables"
    plots = args.output / "plots"
    tables.mkdir(parents=True)
    plots.mkdir(parents=True)

    pipeline = load_module(args.pipeline_script, "relative_growth_pipeline")
    soil = load_module(args.soil_script, "soil_pipeline")
    raw = pd.read_csv(args.input)
    raw, _ = soil.add_soil_indicators(raw)
    long_data, construction = pipeline.build_long_data(raw)
    split = pd.read_csv(args.split_table)[["OID_", "Spatial_Split"]].drop_duplicates("OID_")
    long_data = long_data.merge(split, on="OID_", how="inner", validate="many_to_one")
    long_data, species_columns = add_species_dummies(long_data)
    environment = list(pipeline.ENVIRONMENT_FEATURES) + list(soil.REDUCED_SOIL_FEATURES)
    features = ["Log_Height"] + species_columns + environment

    training = long_data["Spatial_Split"].eq("Training").to_numpy()
    validation = long_data["Spatial_Split"].eq("Validation").to_numpy()
    medians = long_data.loc[training, features].median(numeric_only=True)
    x = long_data[features].fillna(medians).astype(np.float32)
    y = long_data["Log_Specific_Growth_Rate"].to_numpy(np.float32)
    model = XGBRegressor(**pipeline.XGB_PARAMETERS)
    model.fit(x.loc[training], y[training])
    validation_prediction = np.asarray(model.predict(x.loc[validation]), dtype=float)

    training_data = long_data.loc[training, ["OID_", "Species_Name_Model", "Log_Specific_Growth_Rate"]].copy()
    training_data["Actual_Annual_Growth_Percent"] = pipeline.inverse_percentage(training_data["Log_Specific_Growth_Rate"].to_numpy(float))
    validation_data = long_data.loc[validation, ["OID_", "Species_Name_Model", "Log_Specific_Growth_Rate"]].copy()
    validation_data["Actual_Annual_Growth_Percent"] = pipeline.inverse_percentage(validation_data["Log_Specific_Growth_Rate"].to_numpy(float))
    validation_data["Predicted_Annual_Growth_Percent"] = pipeline.inverse_percentage(validation_prediction)
    candidates = validation_candidates(training_data, validation_data)

    best_seven = (
        candidates.loc[(candidates["Levels"].eq(7)) & (candidates["Evaluation_Unit"].eq("tree"))]
        .sort_values(["Exact_Accuracy", "Quadratic_Weighted_Kappa", "Within_One_Accuracy"], ascending=False)
        .iloc[0]
    )
    selected_scope = str(best_seven["Threshold_Scope"])

    training_tree = prepare_unit(training_data, "tree")
    validation_tree = prepare_unit(validation_data, "tree")
    validation_thresholds = derive_thresholds(training_tree, 7, selected_scope)
    validation_observed_level = assign_levels(
        validation_tree["Actual_Annual_Growth_Percent"].to_numpy(float),
        validation_tree["Species_Name_Model"].to_numpy(str), validation_thresholds, selected_scope,
    )
    validation_predicted_level = assign_levels(
        validation_tree["Predicted_Annual_Growth_Percent"].to_numpy(float),
        validation_tree["Species_Name_Model"].to_numpy(str), validation_thresholds, selected_scope,
    )
    validation_hierarchy = pd.DataFrame([
        {"Output": "Seven-level detail", **agreement(validation_observed_level, validation_predicted_level, 7)},
        {"Output": "Three-zone diagnosis", **agreement(zone_from_seven(validation_observed_level), zone_from_seven(validation_predicted_level), 3)},
    ])

    development = long_data["Spatial_Split"].isin(["Training", "Validation"])
    development_data = long_data.loc[development, ["OID_", "Species_Name_Model", "Log_Specific_Growth_Rate"]].copy()
    development_data["Actual_Annual_Growth_Percent"] = pipeline.inverse_percentage(development_data["Log_Specific_Growth_Rate"].to_numpy(float))
    development_tree = prepare_unit(development_data, "tree")
    final_thresholds = derive_thresholds(development_tree, 7, selected_scope)

    test_rows = pd.read_csv(args.locked_test_predictions)
    test_tree = prepare_unit(test_rows, "tree")
    observed_level = assign_levels(
        test_tree["Actual_Annual_Growth_Percent"].to_numpy(float),
        test_tree["Species_Name_Model"].to_numpy(str), final_thresholds, selected_scope,
    )
    predicted_level = assign_levels(
        test_tree["Predicted_Annual_Growth_Percent"].to_numpy(float),
        test_tree["Species_Name_Model"].to_numpy(str), final_thresholds, selected_scope,
    )
    observed_zone = zone_from_seven(observed_level)
    predicted_zone = zone_from_seven(predicted_level)
    seven_metrics = agreement(observed_level, predicted_level, 7)
    zone_metrics = agreement(observed_zone, predicted_zone, 3)
    test_metrics = pd.DataFrame([
        {"Output": "Seven-level detail", "Threshold_Scope": selected_scope, "Evaluation_Unit": "tree", **seven_metrics},
        {"Output": "Three-zone diagnosis", "Threshold_Scope": selected_scope, "Evaluation_Unit": "tree", **zone_metrics},
    ])

    test_tree["Observed_Seven_Level"] = observed_level
    test_tree["Predicted_Seven_Level"] = predicted_level
    test_tree["Observed_Diagnostic_Zone"] = observed_zone
    test_tree["Predicted_Diagnostic_Zone"] = predicted_zone
    test_tree["Observed_Diagnostic_Label"] = pd.Series(observed_zone).map({1: "Constrained", 2: "Typical", 3: "Favorable"}).to_numpy()
    test_tree["Predicted_Diagnostic_Label"] = pd.Series(predicted_zone).map({1: "Constrained", 2: "Typical", 3: "Favorable"}).to_numpy()

    seven_confusion = confusion_matrix(observed_level, predicted_level, labels=np.arange(1, 8))
    zone_confusion = confusion_matrix(observed_zone, predicted_zone, labels=np.arange(1, 4))
    seven_draws = bootstrap_agreement(observed_level, predicted_level, args.bootstrap_repetitions, pipeline.RANDOM_STATE + 41000)
    zone_draws = bootstrap_agreement(observed_zone, predicted_zone, args.bootstrap_repetitions, pipeline.RANDOM_STATE + 42000)
    seven_summary = bootstrap_summary(seven_draws).assign(Output="Seven-level detail")
    zone_summary = bootstrap_summary(zone_draws).assign(Output="Three-zone diagnosis")

    candidates.to_csv(tables / "validation_candidate_suitability_schemes.csv", index=False)
    pd.DataFrame([best_seven]).to_csv(tables / "validation_selected_seven_level_scheme.csv", index=False)
    validation_thresholds.to_csv(tables / "validation_selected_seven_level_thresholds.csv", index=False)
    validation_hierarchy.to_csv(tables / "validation_selected_diagnostic_hierarchy.csv", index=False)
    final_thresholds.to_csv(tables / "fixed_selected_seven_level_thresholds.csv", index=False)
    test_metrics.to_csv(tables / "locked_test_diagnostic_agreement.csv", index=False)
    test_tree.to_csv(tables / "locked_test_tree_diagnostic_levels.csv", index=False)
    pd.DataFrame(seven_confusion, index=[f"Observed_{i}" for i in range(1, 8)], columns=[f"Predicted_{i}" for i in range(1, 8)]).to_csv(tables / "locked_test_seven_level_confusion.csv")
    pd.DataFrame(zone_confusion, index=["Observed_Constrained", "Observed_Typical", "Observed_Favorable"], columns=["Predicted_Constrained", "Predicted_Typical", "Predicted_Favorable"]).to_csv(tables / "locked_test_three_zone_confusion.csv")
    pd.concat([seven_summary, zone_summary], ignore_index=True).to_csv(tables / "locked_test_agreement_bootstrap_summary.csv", index=False)
    save_figure(candidates, seven_confusion, zone_confusion, test_metrics, plots)

    metadata = {
        "input": str(args.input),
        "split_table": str(args.split_table),
        "locked_test_predictions": str(args.locked_test_predictions),
        "selection_rule": "Among seven-level tree/location schemes, maximize validation exact accuracy; break ties by quadratic kappa and within-one accuracy.",
        "selected_threshold_scope": selected_scope,
        "diagnostic_hierarchy": {"primary": "Three zones: constrained=levels 1-2, typical=3-5, favorable=6-7", "secondary": "Seven fixed sublevels"},
        "validation_selected_hierarchy_metrics": validation_hierarchy.to_dict(orient="records"),
        "construction": construction,
        "validation_xgb_r2": float(r2_score(validation_data["Log_Specific_Growth_Rate"], validation_prediction)),
        "test_metrics": test_metrics.to_dict(orient="records"),
        "test_label_note": "The new scheme was selected on validation blocks; locked-test labels were not used in selection.",
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(candidates.to_string(index=False))
    print("\nSelected seven-level scope:", selected_scope)
    print(validation_hierarchy.to_string(index=False))
    print(test_metrics.to_string(index=False))


if __name__ == "__main__":
    main()
