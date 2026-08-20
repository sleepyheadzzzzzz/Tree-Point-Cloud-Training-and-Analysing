#!/usr/bin/env python3
"""Spatially blocked validation of the frozen three-soil deployment workflow.

Candidate algorithms are trained on 70% spatial blocks and compared on 15%
validation blocks. The selected algorithm is refitted on the combined 85%
development blocks and evaluated once on a locked 15% spatial test. All periods
from the same OID_ remain together. The deployment specification omits period
dummies and retains infill, bedrock, and moraine soil indicators.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import time
import warnings

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression
from sklearn.metrics import cohen_kappa_score, confusion_matrix, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


MODELS = ["OLS", "RF", "XGB", "MLP"]
TARGET_SHARES = {"Training": 0.70, "Validation": 0.15, "Test": 0.15}
SPLIT_COLORS = {"Training": "#7A8792", "Validation": "#E09F3E", "Test": "#2C7FB8"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--pipeline-script", required=True, type=Path)
    parser.add_argument("--soil-script", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--block-size-m", type=float, default=500.0)
    parser.add_argument("--split-search-iterations", type=int, default=20000)
    parser.add_argument("--bootstrap-repetitions", type=int, default=1000)
    parser.add_argument("--moran-permutations", type=int, default=999)
    return parser.parse_args()


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_model(name: str, pipeline):
    if name == "OLS":
        return LinearRegression()
    if name == "RF":
        return RandomForestRegressor(**pipeline.RF_PARAMETERS)
    if name == "XGB":
        return XGBRegressor(**pipeline.XGB_PARAMETERS)
    if name == "MLP":
        parameters = dict(pipeline.MLP_PARAMETERS)
        # Avoid a hidden row-level early-stopping split that could separate
        # repeated periods from the same training tree.
        parameters["early_stopping"] = False
        parameters.pop("validation_fraction", None)
        return MLPRegressor(**parameters)
    raise KeyError(name)


def add_spatial_blocks(data: pd.DataFrame, block_size_m: float) -> pd.DataFrame:
    result = data.copy()
    if result[["X", "Y"]].isna().any(axis=None):
        raise ValueError("Spatial validation requires non-missing X and Y")
    result["Block_X"] = np.floor(result["X"] / block_size_m).astype(np.int64)
    result["Block_Y"] = np.floor(result["Y"] / block_size_m).astype(np.int64)
    result["Spatial_Block"] = (
        result["Block_X"].astype(str) + "_" + result["Block_Y"].astype(str)
    )
    return result


def choose_block_split(
    tree_meta: pd.DataFrame,
    row_counts: pd.Series,
    iterations: int,
    random_state: int,
) -> tuple[dict[str, str], dict]:
    species = sorted(tree_meta["Species_Name_Model"].unique())
    blocks = sorted(tree_meta["Spatial_Block"].unique())
    block_index = {block: index for index, block in enumerate(blocks)}
    block_tree = np.zeros(len(blocks), dtype=float)
    block_rows = np.zeros(len(blocks), dtype=float)
    block_species = np.zeros((len(blocks), len(species)), dtype=float)
    species_index = {name: index for index, name in enumerate(species)}
    for row in tree_meta.itertuples(index=False):
        b = block_index[row.Spatial_Block]
        block_tree[b] += 1
        block_rows[b] += float(row_counts.loc[row.OID_])
        block_species[b, species_index[row.Species_Name_Model]] += 1

    total_trees = block_tree.sum()
    total_rows = block_rows.sum()
    overall_species = block_species.sum(axis=0) / total_trees
    n_blocks = len(blocks)
    n_validation = max(1, round(n_blocks * TARGET_SHARES["Validation"]))
    n_test = max(1, round(n_blocks * TARGET_SHARES["Test"]))
    n_training = n_blocks - n_validation - n_test
    rng = np.random.default_rng(random_state)
    best = None
    best_score = np.inf
    targets = np.array([0.70, 0.15, 0.15])

    for _ in range(iterations):
        permutation = rng.permutation(n_blocks)
        groups = [
            permutation[:n_training],
            permutation[n_training : n_training + n_validation],
            permutation[n_training + n_validation :],
        ]
        tree_shares = np.array([block_tree[group].sum() / total_trees for group in groups])
        row_shares = np.array([block_rows[group].sum() / total_rows for group in groups])
        score = float(np.sum(((tree_shares - targets) / 0.0125) ** 2))
        score += float(np.sum(((row_shares - targets) / 0.0125) ** 2))
        for group in groups:
            counts = block_species[group].sum(axis=0)
            if counts.sum() == 0:
                score += 1e6
                continue
            distribution = counts / counts.sum()
            score += 3.0 * float(np.mean(np.abs(distribution - overall_species)))
            score += 200.0 * float(np.sum(counts < 10))
        if score < best_score:
            best_score = score
            best = groups

    if best is None:
        raise RuntimeError("No spatial split candidate was created")
    labels = ["Training", "Validation", "Test"]
    split_map: dict[str, str] = {}
    for label, indices in zip(labels, best):
        for index in indices:
            split_map[blocks[int(index)]] = label

    diagnostics = {
        "objective": best_score,
        "block_counts": {label: int(len(indices)) for label, indices in zip(labels, best)},
        "block_size_note": "Blocks are assigned as indivisible groups; adjacent blocks may touch.",
    }
    return split_map, diagnostics


def add_species_dummies(data: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    dummies = pd.get_dummies(
        data["Species_Name_Model"], prefix="Species", dtype=np.float32
    )
    result = pd.concat(
        [data.reset_index(drop=True), dummies.reset_index(drop=True)], axis=1
    )
    return result, dummies.columns.tolist()


def matrices(
    data: pd.DataFrame,
    features: list[str],
    fit_mask: np.ndarray,
) -> tuple[pd.DataFrame, pd.Series]:
    medians = data.loc[fit_mask, features].median(numeric_only=True)
    x = data[features].fillna(medians).astype(np.float32)
    return x, medians


def evaluate(pipeline, data: pd.DataFrame, prediction: np.ndarray, split: str) -> dict:
    mask = data["Spatial_Split"].eq(split).to_numpy()
    actual = data.loc[mask, "Log_Specific_Growth_Rate"].to_numpy(float)
    record = pipeline.metric_record(
        actual,
        prediction,
        data.loc[mask, "Initial_Carbon"].to_numpy(float),
        data.loc[mask, "Years"].to_numpy(float),
    )
    return {
        "Split": split,
        "N_Rows": int(mask.sum()),
        "N_Trees": int(data.loc[mask, "OID_"].nunique()),
        **record,
    }


def fit_candidates(
    pipeline,
    data: pd.DataFrame,
    features: list[str],
) -> tuple[pd.DataFrame, str, dict]:
    training = data["Spatial_Split"].eq("Training").to_numpy()
    validation = data["Spatial_Split"].eq("Validation").to_numpy()
    x, medians = matrices(data, features, training)
    y = data["Log_Specific_Growth_Rate"].to_numpy(np.float32)
    scaler = StandardScaler().fit(x.loc[training])
    x_training_scaled = scaler.transform(x.loc[training])
    x_validation_scaled = scaler.transform(x.loc[validation])
    rows = []
    fitted = {}
    for name in MODELS:
        model = make_model(name, pipeline)
        use_scaled = name in {"OLS", "MLP"}
        fit_x = x_training_scaled if use_scaled else x.loc[training]
        validation_x = x_validation_scaled if use_scaled else x.loc[validation]
        started = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            model.fit(fit_x, y[training])
        seconds = time.perf_counter() - started
        train_prediction = model.predict(fit_x)
        validation_prediction = model.predict(validation_x)
        for split, prediction in [
            ("Training", train_prediction),
            ("Validation", validation_prediction),
        ]:
            record = evaluate(pipeline, data, prediction, split)
            rows.append({"Model": name, "Fit_Seconds": seconds, **record})
        fitted[name] = model
    comparison = pd.DataFrame(rows)
    validation_rows = comparison.loc[comparison["Split"].eq("Validation")]
    selected = str(
        validation_rows.sort_values(
            ["R2_LogSGR", "RMSE_LogSGR"], ascending=[False, True]
        ).iloc[0]["Model"]
    )
    context = {"training_medians": medians, "candidate_scaler": scaler, "fitted": fitted}
    return comparison, selected, context


def refit_selected(
    pipeline,
    data: pd.DataFrame,
    features: list[str],
    selected: str,
) -> dict:
    development = data["Spatial_Split"].isin(["Training", "Validation"]).to_numpy()
    test = data["Spatial_Split"].eq("Test").to_numpy()
    x, medians = matrices(data, features, development)
    y = data["Log_Specific_Growth_Rate"].to_numpy(np.float32)
    scaler = StandardScaler().fit(x.loc[development])
    use_scaled = selected in {"OLS", "MLP"}
    fit_x = scaler.transform(x.loc[development]) if use_scaled else x.loc[development]
    test_x = scaler.transform(x.loc[test]) if use_scaled else x.loc[test]
    model = make_model(selected, pipeline)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        model.fit(fit_x, y[development])
    return {
        "model": model,
        "scaler": scaler,
        "medians": medians,
        "x": x,
        "development_mask": development,
        "test_mask": test,
        "development_prediction": np.asarray(model.predict(fit_x), dtype=float),
        "test_prediction": np.asarray(model.predict(test_x), dtype=float),
        "use_scaled": use_scaled,
    }


def model_predict(context: dict, matrix: pd.DataFrame | np.ndarray) -> np.ndarray:
    values = context["scaler"].transform(matrix) if context["use_scaled"] else matrix
    return np.asarray(context["model"].predict(values), dtype=float)


def calibration(actual: np.ndarray, predicted: np.ndarray) -> tuple[float, float]:
    slope, intercept = np.polyfit(predicted, actual, 1)
    return float(intercept), float(slope)


def suitability_class(values: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    return np.digitize(np.asarray(values, dtype=float), thresholds, right=False) + 1


def morans_i(
    coords: np.ndarray,
    residuals: np.ndarray,
    permutations: int,
    random_state: int,
    k: int = 8,
) -> dict:
    n = len(residuals)
    if n <= k + 1:
        return {"Morans_I": np.nan, "Permutation_P_Two_Sided": np.nan, "N_Trees": n, "K": k}
    tree = cKDTree(coords)
    _, neighbours = tree.query(coords, k=k + 1)
    rows = np.repeat(np.arange(n), k)
    columns = neighbours[:, 1:].reshape(-1)
    directed = csr_matrix((np.ones(len(rows)), (rows, columns)), shape=(n, n))
    weights = ((directed + directed.T) > 0).astype(float).tocsr()
    weights.setdiag(0)
    weights.eliminate_zeros()
    s0 = float(weights.sum())
    centered = residuals - residuals.mean()
    denominator = float(centered @ centered)

    def statistic(values: np.ndarray) -> float:
        return float((n / s0) * (values @ (weights @ values)) / denominator)

    observed = statistic(centered)
    rng = np.random.default_rng(random_state)
    simulated = np.empty(permutations, dtype=float)
    for index in range(permutations):
        simulated[index] = statistic(rng.permutation(centered))
    expected = float(simulated.mean())
    p = (1 + np.sum(np.abs(simulated - expected) >= abs(observed - expected))) / (
        permutations + 1
    )
    return {
        "Morans_I": observed,
        "Permutation_P_Two_Sided": float(p),
        "Permutation_Mean": expected,
        "N_Trees": n,
        "K": k,
        "Permutations": permutations,
    }


def subgroup_metrics(
    pipeline,
    test_data: pd.DataFrame,
    actual: np.ndarray,
    predicted: np.ndarray,
) -> pd.DataFrame:
    groups: list[tuple[str, np.ndarray]] = [
        ("Overall", np.ones(len(test_data), dtype=bool)),
        ("Inside training min-max", test_data["Reliable_MinMax"].to_numpy(bool)),
        ("Outside training min-max", ~test_data["Reliable_MinMax"].to_numpy(bool)),
        ("Inside training P01-P99", test_data["Reliable_P01P99"].to_numpy(bool)),
        ("Outside training P01-P99", ~test_data["Reliable_P01P99"].to_numpy(bool)),
        ("Park", test_data["type_Puisto"].eq(1).to_numpy()),
        ("Street", test_data["type_Puisto"].eq(0).to_numpy()),
    ]
    groups.extend(
        (f"Species: {species}", test_data["Species_Name_Model"].eq(species).to_numpy())
        for species in sorted(test_data["Species_Name_Model"].unique())
    )
    rows = []
    for name, mask in groups:
        if mask.sum() < 2:
            continue
        record = pipeline.metric_record(
            actual[mask],
            predicted[mask],
            test_data.loc[mask, "Initial_Carbon"].to_numpy(float),
            test_data.loc[mask, "Years"].to_numpy(float),
        )
        rows.append(
            {
                "Group": name,
                "N_Rows": int(mask.sum()),
                "N_Trees": int(test_data.loc[mask, "OID_"].nunique()),
                **record,
            }
        )
    return pd.DataFrame(rows)


def clustered_bootstrap(
    pipeline,
    test_data: pd.DataFrame,
    actual: np.ndarray,
    predicted: np.ndarray,
    observed_baseline_residual: np.ndarray,
    environmental_delta: np.ndarray,
    repetitions: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ids = test_data["OID_"].drop_duplicates().to_numpy()
    oid_values = test_data["OID_"].to_numpy()
    lookup = {oid: np.flatnonzero(oid_values == oid) for oid in ids}
    rng = np.random.default_rng(random_state)
    rows = []
    for iteration in range(repetitions):
        sampled = rng.choice(ids, size=len(ids), replace=True)
        index = np.concatenate([lookup[oid] for oid in sampled])
        record = pipeline.metric_record(
            actual[index],
            predicted[index],
            test_data.loc[index, "Initial_Carbon"].to_numpy(float),
            test_data.loc[index, "Years"].to_numpy(float),
        )
        intercept, slope = calibration(actual[index], predicted[index])
        rho = spearmanr(
            environmental_delta[index], observed_baseline_residual[index]
        ).statistic
        rows.append(
            {
                "Iteration": iteration + 1,
                **record,
                "Calibration_Intercept": intercept,
                "Calibration_Slope": slope,
                "Environmental_Contrast_vs_Observed_Baseline_Residual_Spearman": rho,
            }
        )
    draws = pd.DataFrame(rows)
    summaries = []
    for column in draws.columns:
        if column == "Iteration":
            continue
        summaries.append(
            {
                "Metric": column,
                "Median": draws[column].median(),
                "CI95_Lower": draws[column].quantile(0.025),
                "CI95_Upper": draws[column].quantile(0.975),
            }
        )
    return draws, pd.DataFrame(summaries)


def save_figures(
    output: Path,
    tree_meta: pd.DataFrame,
    test_data: pd.DataFrame,
    actual: np.ndarray,
    predicted: np.ndarray,
    tree_residuals: pd.DataFrame,
    confusion: np.ndarray,
    diagnostics: dict,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 10.5), constrained_layout=True)
    ax = axes[0, 0]
    for split in ["Training", "Validation", "Test"]:
        subset = tree_meta.loc[tree_meta["Spatial_Split"].eq(split)]
        ax.scatter(
            (subset["X"] - 25_490_000.0) / 1000.0,
            (subset["Y"] - 6_670_000.0) / 1000.0,
            s=4, alpha=0.55,
            color=SPLIT_COLORS[split], label=f"{split} ({len(subset):,} trees)",
            rasterized=True,
        )
    ax.set_title("A  Spatially blocked 70/15/15 partition", loc="left", fontweight="bold")
    ax.set_xlabel("Easting offset from 25,490,000 m (km)")
    ax.set_ylabel("Northing offset from 6,670,000 m (km)")
    ax.legend(frameon=False, markerscale=2.5, fontsize=8)
    ax.set_aspect("equal", adjustable="datalim")

    ax = axes[0, 1]
    hb = ax.hexbin(predicted, actual, gridsize=55, mincnt=1, cmap="viridis", bins="log")
    lower = float(min(actual.min(), predicted.min()))
    upper = float(max(actual.max(), predicted.max()))
    ax.plot([lower, upper], [lower, upper], color="#333333", linewidth=1.2, label="1:1")
    intercept = diagnostics["Calibration_Intercept"]
    slope = diagnostics["Calibration_Slope"]
    x_line = np.array([lower, upper])
    ax.plot(x_line, intercept + slope * x_line, color="#D95F02", linewidth=1.5, label="Calibration")
    ax.set_title("B  Locked-test calibration", loc="left", fontweight="bold")
    ax.set_xlabel("Predicted log-SGR")
    ax.set_ylabel("Observed log-SGR")
    ax.legend(frameon=False, fontsize=8)
    colorbar = fig.colorbar(hb, ax=ax, shrink=0.82)
    colorbar.set_label("Observation density (log scale)")
    ax.text(
        0.03, 0.97,
        f"R² = {diagnostics['Test_R2_LogSGR']:.3f}\n"
        f"intercept = {intercept:.3f}\nslope = {slope:.3f}",
        transform=ax.transAxes, ha="left", va="top", fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "#C7CDD2", "alpha": 0.9},
    )

    ax = axes[1, 0]
    limit = float(np.quantile(np.abs(tree_residuals["Mean_Residual_LogSGR"]), 0.98))
    scatter = ax.scatter(
        (tree_residuals["X"] - 25_490_000.0) / 1000.0,
        (tree_residuals["Y"] - 6_670_000.0) / 1000.0,
        c=tree_residuals["Mean_Residual_LogSGR"], cmap="RdBu_r",
        vmin=-limit, vmax=limit, s=9, alpha=0.8, rasterized=True,
    )
    ax.set_title("C  Locked-test residuals by tree", loc="left", fontweight="bold")
    ax.set_xlabel("Easting offset from 25,490,000 m (km)")
    ax.set_ylabel("Northing offset from 6,670,000 m (km)")
    ax.set_aspect("equal", adjustable="datalim")
    colorbar = fig.colorbar(scatter, ax=ax, shrink=0.82)
    colorbar.set_label("Observed - predicted log-SGR")
    ax.text(
        0.03, 0.97,
        f"Moran's I = {diagnostics['Morans_I']:.3f}\n"
        f"permutation p = {diagnostics['Moran_P']:.3f}",
        transform=ax.transAxes, ha="left", va="top", fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "#C7CDD2", "alpha": 0.9},
    )

    ax = axes[1, 1]
    row_totals = confusion.sum(axis=1, keepdims=True)
    normalized = np.divide(
        confusion, row_totals, out=np.zeros_like(confusion, dtype=float), where=row_totals > 0
    )
    image = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=max(0.01, normalized.max()))
    for row in range(7):
        for column in range(7):
            value = normalized[row, column]
            if value >= 0.08:
                ax.text(column, row, f"{value:.2f}", ha="center", va="center",
                        fontsize=7, color="white" if value > 0.5 * normalized.max() else "#1E2A35")
    ax.set_title("D  Fixed-threshold suitability agreement", loc="left", fontweight="bold")
    ax.set_xlabel("Predicted level")
    ax.set_ylabel("Observed level")
    ax.set_xticks(range(7), range(1, 8))
    ax.set_yticks(range(7), range(1, 8))
    fig.colorbar(image, ax=ax, shrink=0.82, label="Row proportion")
    ax.text(
        0.03, 0.97,
        f"exact = {diagnostics['Suitability_Exact_Accuracy']:.2%}\n"
        f"within ±1 = {diagnostics['Suitability_Within_One']:.2%}\n"
        f"quadratic κ = {diagnostics['Suitability_Quadratic_Kappa']:.3f}",
        transform=ax.transAxes, ha="left", va="top", fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "#C7CDD2", "alpha": 0.9},
    )

    fig.suptitle(
        "Spatially blocked validation of the three-soil deployment model",
        fontsize=16, fontweight="bold",
    )
    fig.savefig(output / "Figure_spatial_block_validation_three_soil.png", dpi=400, bbox_inches="tight")
    fig.savefig(output / "Figure_spatial_block_validation_three_soil.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Output directory already exists: {args.output}")
    tables = args.output / "tables"
    plots = args.output / "plots"
    models = args.output / "models"
    for directory in [tables, plots, models]:
        directory.mkdir(parents=True, exist_ok=False)

    pipeline = load_module(args.pipeline_script, "relative_growth_pipeline")
    soil = load_module(args.soil_script, "soil_pipeline")
    raw = pd.read_csv(args.input)
    raw, soil_counts = soil.add_soil_indicators(raw)
    long_data, construction = pipeline.build_long_data(raw)
    long_data = add_spatial_blocks(long_data, args.block_size_m)

    tree_meta = long_data.groupby("OID_", as_index=False).agg(
        X=("X", "first"),
        Y=("Y", "first"),
        Species_Name_Model=("Species_Name_Model", "first"),
        Spatial_Block=("Spatial_Block", "first"),
    )
    row_counts = long_data.groupby("OID_").size()
    block_split, split_diagnostics = choose_block_split(
        tree_meta,
        row_counts,
        args.split_search_iterations,
        pipeline.RANDOM_STATE + 31000,
    )
    tree_meta["Spatial_Split"] = tree_meta["Spatial_Block"].map(block_split)
    long_data = long_data.merge(
        tree_meta[["OID_", "Spatial_Split"]], on="OID_", how="left", validate="many_to_one"
    )
    long_data, species_columns = add_species_dummies(long_data)

    if long_data.groupby("OID_")["Spatial_Split"].nunique().max() != 1:
        raise AssertionError("OID_ leakage across spatial splits")
    if long_data.groupby("Spatial_Block")["Spatial_Split"].nunique().max() != 1:
        raise AssertionError("Spatial block leakage across splits")

    environment_features = list(pipeline.ENVIRONMENT_FEATURES) + list(
        soil.REDUCED_SOIL_FEATURES
    )
    features = ["Log_Height"] + species_columns + environment_features
    candidate_metrics, selected, _ = fit_candidates(pipeline, long_data, features)
    final = refit_selected(pipeline, long_data, features, selected)

    development = final["development_mask"]
    test = final["test_mask"]
    test_data = long_data.loc[test].reset_index(drop=True)
    actual = test_data["Log_Specific_Growth_Rate"].to_numpy(float)
    predicted = final["test_prediction"]

    # Biological baseline for held-out environmental-contrast concordance.
    baseline_features = ["Log_Height"] + species_columns
    x_baseline, baseline_medians = matrices(long_data, baseline_features, development)
    baseline = XGBRegressor(**pipeline.XGB_PARAMETERS)
    baseline.fit(
        x_baseline.loc[development],
        long_data.loc[development, "Log_Specific_Growth_Rate"].to_numpy(np.float32),
    )
    baseline_test_prediction = np.asarray(baseline.predict(x_baseline.loc[test]), dtype=float)

    # Reference predictions preserve each test tree's height and species while
    # setting the environmental block to the development median.
    x_test = final["x"].loc[test].reset_index(drop=True)
    x_reference = x_test.copy()
    for feature in environment_features:
        x_reference[feature] = float(final["medians"][feature])
    reference_prediction = model_predict(final, x_reference)
    environmental_delta_log = predicted - reference_prediction
    environmental_delta_pp = pipeline.inverse_percentage(predicted) - pipeline.inverse_percentage(
        reference_prediction
    )
    observed_baseline_residual = actual - baseline_test_prediction

    # Reliability rules are frozen from the combined 85% development set.
    domain_features = ["Log_Height"] + environment_features
    domain_rows = []
    for feature in domain_features:
        values = final["x"].loc[development, feature].to_numpy(float)
        domain_rows.append(
            {
                "Feature": feature,
                "Minimum": float(np.min(values)),
                "P01": float(np.quantile(values, 0.01)),
                "Median": float(np.median(values)),
                "P99": float(np.quantile(values, 0.99)),
                "Maximum": float(np.max(values)),
            }
        )
    domain = pd.DataFrame(domain_rows).set_index("Feature")
    outside_minmax = np.zeros(len(test_data), dtype=int)
    outside_robust = np.zeros(len(test_data), dtype=int)
    for feature in domain_features:
        values = x_test[feature].to_numpy(float)
        outside_minmax += (
            (values < domain.loc[feature, "Minimum"])
            | (values > domain.loc[feature, "Maximum"])
        )
        outside_robust += (
            (values < domain.loc[feature, "P01"])
            | (values > domain.loc[feature, "P99"])
        )
    test_data["Outside_MinMax_Count"] = outside_minmax
    test_data["Reliable_MinMax"] = outside_minmax == 0
    test_data["Outside_P01P99_Count"] = outside_robust
    test_data["Reliable_P01P99"] = outside_robust == 0

    actual_pct = pipeline.inverse_percentage(actual)
    predicted_pct = pipeline.inverse_percentage(predicted)
    development_actual_pct = pipeline.inverse_percentage(
        long_data.loc[development, "Log_Specific_Growth_Rate"].to_numpy(float)
    )
    thresholds = np.quantile(development_actual_pct, np.arange(1, 7) / 7)
    actual_level = suitability_class(actual_pct, thresholds)
    predicted_level = suitability_class(predicted_pct, thresholds)
    suitability_confusion = confusion_matrix(actual_level, predicted_level, labels=np.arange(1, 8))
    suitability_exact = float(np.mean(actual_level == predicted_level))
    suitability_within_one = float(np.mean(np.abs(actual_level - predicted_level) <= 1))
    suitability_mae = float(np.mean(np.abs(actual_level - predicted_level)))
    suitability_kappa = float(
        cohen_kappa_score(actual_level, predicted_level, weights="quadratic")
    )

    calibration_intercept, calibration_slope = calibration(actual, predicted)
    diagnostic_rho = float(
        spearmanr(environmental_delta_log, observed_baseline_residual).statistic
    )
    sign_agreement = float(
        np.mean(np.sign(environmental_delta_log) == np.sign(observed_baseline_residual))
    )
    q20, q80 = np.quantile(environmental_delta_log, [0.2, 0.8])
    observed_residual_contrast = float(
        observed_baseline_residual[environmental_delta_log >= q80].mean()
        - observed_baseline_residual[environmental_delta_log <= q20].mean()
    )

    tree_residuals = (
        test_data.assign(Residual_LogSGR=actual - predicted)
        .groupby("OID_", as_index=False)
        .agg(
            X=("X", "first"),
            Y=("Y", "first"),
            Mean_Residual_LogSGR=("Residual_LogSGR", "mean"),
        )
    )
    moran = morans_i(
        tree_residuals[["X", "Y"]].to_numpy(float),
        tree_residuals["Mean_Residual_LogSGR"].to_numpy(float),
        args.moran_permutations,
        pipeline.RANDOM_STATE + 32000,
    )

    final_metric = evaluate(pipeline, long_data, predicted, "Test")
    subgroup = subgroup_metrics(pipeline, test_data, actual, predicted)
    bootstrap_draws, bootstrap_summary = clustered_bootstrap(
        pipeline,
        test_data,
        actual,
        predicted,
        observed_baseline_residual,
        environmental_delta_log,
        args.bootstrap_repetitions,
        pipeline.RANDOM_STATE + 33000,
    )

    diagnostics = {
        "Selected_Model": selected,
        "Test_R2_LogSGR": float(final_metric["R2_LogSGR"]),
        "Calibration_Intercept": calibration_intercept,
        "Calibration_Slope": calibration_slope,
        "Morans_I": float(moran["Morans_I"]),
        "Moran_P": float(moran["Permutation_P_Two_Sided"]),
        "Suitability_Exact_Accuracy": suitability_exact,
        "Suitability_Within_One": suitability_within_one,
        "Suitability_MAE_Levels": suitability_mae,
        "Suitability_Quadratic_Kappa": suitability_kappa,
        "Environmental_Contrast_vs_Observed_Baseline_Residual_Spearman": diagnostic_rho,
        "Environmental_Contrast_Sign_Agreement": sign_agreement,
        "Observed_Baseline_Residual_High_minus_Low_Diagnosis_Quintile": observed_residual_contrast,
        "Reliable_MinMax_Fraction": float(test_data["Reliable_MinMax"].mean()),
        "Reliable_P01P99_Fraction": float(test_data["Reliable_P01P99"].mean()),
        **{f"Test_{key}": value for key, value in final_metric.items() if key not in {"Split", "N_Rows", "N_Trees"}},
    }

    predictions = test_data[
        [
            "OID_", "Period", "Species_Name_Model", "X", "Y", "Spatial_Block",
            "Initial_Carbon", "Years", "Reliable_MinMax", "Outside_MinMax_Count",
            "Reliable_P01P99", "Outside_P01P99_Count",
        ]
    ].copy()
    predictions["Actual_LogSGR"] = actual
    predictions["Predicted_LogSGR"] = predicted
    predictions["Residual_LogSGR_Observed_minus_Predicted"] = actual - predicted
    predictions["Baseline_Predicted_LogSGR"] = baseline_test_prediction
    predictions["Observed_Baseline_Residual_LogSGR"] = observed_baseline_residual
    predictions["Reference_Environment_Predicted_LogSGR"] = reference_prediction
    predictions["Environmental_Deviation_LogSGR"] = environmental_delta_log
    predictions["Environmental_Deviation_Percentage_Points"] = environmental_delta_pp
    predictions["Actual_Annual_Growth_Percent"] = actual_pct
    predictions["Predicted_Annual_Growth_Percent"] = predicted_pct
    predictions["Actual_Suitability_Level"] = actual_level
    predictions["Predicted_Suitability_Level"] = predicted_level

    split_summary = []
    for split in ["Training", "Validation", "Test"]:
        mask = long_data["Spatial_Split"].eq(split)
        split_summary.append(
            {
                "Split": split,
                "N_Blocks": int(long_data.loc[mask, "Spatial_Block"].nunique()),
                "N_Trees": int(long_data.loc[mask, "OID_"].nunique()),
                "Tree_Share": float(long_data.loc[mask, "OID_"].nunique() / long_data["OID_"].nunique()),
                "N_Rows": int(mask.sum()),
                "Row_Share": float(mask.mean()),
            }
        )
    split_summary = pd.DataFrame(split_summary)

    candidate_metrics.to_csv(tables / "candidate_model_training_validation_metrics.csv", index=False)
    pd.DataFrame([final_metric]).to_csv(tables / "locked_test_selected_model_metrics.csv", index=False)
    subgroup.to_csv(tables / "locked_test_subgroup_and_reliability_metrics.csv", index=False)
    predictions.to_csv(tables / "locked_test_coordinate_predictions.csv", index=False)
    tree_residuals.to_csv(tables / "locked_test_tree_residuals.csv", index=False)
    split_summary.to_csv(tables / "spatial_split_summary.csv", index=False)
    tree_meta.to_csv(tables / "tree_spatial_split.csv", index=False)
    domain.reset_index().to_csv(tables / "development_training_domain.csv", index=False)
    pd.DataFrame(
        {"Threshold_Number": np.arange(1, 7), "Annual_Growth_Percent_Threshold": thresholds}
    ).to_csv(tables / "fixed_suitability_thresholds.csv", index=False)
    pd.DataFrame(
        suitability_confusion,
        index=[f"Observed_{i}" for i in range(1, 8)],
        columns=[f"Predicted_{i}" for i in range(1, 8)],
    ).to_csv(tables / "suitability_confusion_matrix.csv")
    pd.DataFrame([diagnostics]).to_csv(tables / "spatial_validation_diagnostics.csv", index=False)
    pd.DataFrame([moran]).to_csv(tables / "moran_residual_test.csv", index=False)
    bootstrap_draws.to_csv(tables / "locked_test_tree_cluster_bootstrap_draws.csv", index=False)
    bootstrap_summary.to_csv(tables / "locked_test_tree_cluster_bootstrap_summary.csv", index=False)
    soil_counts.to_csv(tables / "soil_code_mapping_and_counts.csv", index=False)

    save_figures(
        plots,
        tree_meta,
        test_data,
        actual,
        predicted,
        tree_residuals,
        suitability_confusion,
        diagnostics,
    )

    if selected == "XGB":
        final["model"].save_model(models / "selected_no_period_three_soil_xgb.json")
    joblib.dump(final["model"], models / "selected_no_period_three_soil_model.joblib")
    baseline.save_model(models / "biological_baseline_no_period_xgb.json")
    joblib.dump(
        {
            "selected_model": selected,
            "feature_columns": features,
            "feature_medians": final["medians"],
            "species_columns": species_columns,
            "environment_features": environment_features,
            "soil_features": list(soil.REDUCED_SOIL_FEATURES),
            "scaler": final["scaler"],
            "use_scaled": final["use_scaled"],
            "block_size_m": args.block_size_m,
            "suitability_thresholds_annual_growth_percent": thresholds,
        },
        models / "spatial_validation_preprocessing.joblib",
    )

    metadata = {
        "input": str(args.input),
        "pipeline_script": str(args.pipeline_script),
        "soil_script": str(args.soil_script),
        "target": "log annualized specific carbon growth (log-SGR)",
        "deployment_specification": "No monitoring-period indicators",
        "block_size_m": args.block_size_m,
        "crs": "EPSG:3879",
        "split_search_iterations": args.split_search_iterations,
        "split_search": split_diagnostics,
        "construction": construction,
        "features": features,
        "soil_features": list(soil.REDUCED_SOIL_FEATURES),
        "selected_model": selected,
        "split_summary": split_summary.to_dict(orient="records"),
        "locked_test_metrics": final_metric,
        "diagnostics": diagnostics,
        "moran": moran,
        "bootstrap_repetitions": args.bootstrap_repetitions,
    }
    (args.output / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    (args.output / "RUN_LOG.md").write_text(
        "\n".join(
            [
                "# Spatial-block validation of the three-soil deployment model",
                "",
                f"- Spatial unit: {args.block_size_m:.0f} m grid block in EPSG:3879.",
                "- All repeated periods from each OID_ and every tree in a block remain in one split.",
                "- Candidate algorithms were fitted on Training only and compared on Validation only.",
                f"- Selected algorithm by validation R2: {selected}.",
                "- The selected algorithm was refitted on Training + Validation and evaluated once on the locked Test blocks.",
                "- The deployment specification omits monitoring-period dummies and includes infill, bedrock, and moraine; clay and silt-sand are excluded.",
                "- Coordinate-linked environmental values are the pre-extracted spatial-layer values at held-out tree locations.",
                "- Environmental deviation compares local and development-median environmental inputs for the same held-out tree height and species.",
                "- Environmental deviation is a model contrast, not an observed residual or causal effect.",
                "- Fixed suitability thresholds are septiles derived from the combined development outcomes and are not recomputed on the test set.",
                f"- Locked-test R2 (log-SGR): {final_metric['R2_LogSGR']:.4f}.",
                f"- Calibration intercept/slope: {calibration_intercept:.4f} / {calibration_slope:.4f}.",
                f"- Residual Moran's I: {moran['Morans_I']:.4f} (permutation p={moran['Permutation_P_Two_Sided']:.4f}).",
                f"- Suitability exact / within-one accuracy: {suitability_exact:.3f} / {suitability_within_one:.3f}.",
                f"- Environmental contrast versus observed baseline-residual Spearman: {diagnostic_rho:.4f}.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
