#!/usr/bin/env python3
"""Apply the pooled relative-growth XGBoost model to the Helsinki 2 m grid.

The script intentionally uses the no-period deployment model while allowing
period-specific environmental layers to vary. It produces:

* annual relative carbon-growth percentage GeoTIFFs;
* local-minus-reference environmental-deviation GeoTIFFs;
* fixed training-derived suitability-level GeoTIFFs;
* three-zone diagnostic GeoTIFFs (constrained, typical, favorable);
* reliability masks based on the development-data feature domain;
* optional GIS-ready wide CSV predictions; and
* metadata, thresholds, reference predictions, and a run log.

No "Overall" species scenario is created. Every prediction sets exactly one
of the 11 observed one-hot species/category indicators.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import shutil
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import rasterio
import xgboost as xgb
from rasterio.transform import from_origin


SPECIES = {
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

PERIODS = {
    "15_17": {
        "Density25": "Density_15",
        "Mono_Rate": "Mono_Rate_",
        "avg_LST": "LST15_17",
        "avg_radiation": "RA15_17",
        "avg_svf": "svf15_17",
    },
    "17_21": {
        "Density25": "Density_17",
        "Mono_Rate": "Mono_Rate1",
        "avg_LST": "LST17_21",
        "avg_radiation": "RA17_21",
        "avg_svf": "svf17_21",
    },
    "21_23": {
        "Density25": "Density_21",
        "Mono_Rate": "Mono_Rat_1",
        "avg_LST": "LST21_23",
        "avg_radiation": "RA21_23",
        "avg_svf": "svf21_23",
    },
}

SHARED_SOURCE_COLUMNS = {
    "avg_noise_day": "noise",
    "lightemiss": "lightemiss",
    "soil_infill": "soil_infill",
    "soil_bedrock": "soil_bedrock",
    "soil_moraine": "soil_moraine",
}

IDENTIFIER_COLUMNS = ["OID_", "id", "X", "Y"]
ENVIRONMENT_FEATURES = [
    "avg_noise_day",
    "Density25",
    "Mono_Rate",
    "avg_svf",
    "avg_radiation",
    "avg_LST",
    "lightemiss",
    "type_Puisto",
    "soil_infill",
    "soil_bedrock",
    "soil_moraine",
]
DOMAIN_FEATURES = [
    "avg_noise_day",
    "Density25",
    "Mono_Rate",
    "avg_svf",
    "avg_radiation",
    "avg_LST",
    "lightemiss",
    "soil_infill",
    "soil_bedrock",
    "soil_moraine",
]

MODEL_FILENAMES = (
    "xgb_spatial_deployment_no_period_three_soil.json",
    "xgb_no_period.json",
)
PREPROCESSING_FILENAMES = (
    "preprocessing_spatial_deployment.joblib",
    "preprocessing.joblib",
)

PREDICTION_NODATA = -9999.0
RELIABILITY_NODATA = 255
QUIET_NOISE_DB = 40.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the relative-growth spatial diagnosis on a point grid."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--training-data", required=True, type=Path)
    parser.add_argument("--pipeline-script", required=True, type=Path)
    parser.add_argument(
        "--soil-script",
        type=Path,
        default=Path(__file__).with_name("run_soil_augmented_shap_analysis.py"),
        help="Script providing add_soil_indicators for three-soil deployment models.",
    )
    parser.add_argument(
        "--fixed-suitability-thresholds",
        type=Path,
        help=(
            "Optional CSV containing six locked annual-growth thresholds. "
            "Use the validation-selected tree-level threshold table for the "
            "diagnostic tool; otherwise the legacy standardized-scenario "
            "septiles are derived."
        ),
    )
    parser.add_argument("--reference-height-m", type=float, default=10.0)
    parser.add_argument("--park-context", type=float, choices=[0.0, 1.0], default=1.0)
    parser.add_argument("--resolution", type=float, default=2.0)
    parser.add_argument("--crs", default="EPSG:3879")
    parser.add_argument("--chunk-size", type=int, default=250_000)
    parser.add_argument(
        "--write-csv",
        action="store_true",
        help="Also write a wide GIS-ready CSV of percentage predictions.",
    )
    parser.add_argument(
        "--scan-only",
        action="store_true",
        help="Inspect grid geometry and input ranges without predicting.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional testing limit; omit for the complete grid.",
    )
    return parser.parse_args()


def load_pipeline_module(path: Path):
    spec = importlib.util.spec_from_file_location("relative_growth_pipeline", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load pipeline module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def inverse_percentage(log_sgr: np.ndarray) -> np.ndarray:
    """Back-transform log-SGR to annual compound percentage growth."""
    sgr = np.exp(np.clip(np.asarray(log_sgr, dtype=np.float64), -30.0, 5.0))
    return (100.0 * np.expm1(sgr)).astype(np.float32)


def required_input_columns() -> list[str]:
    columns = set(IDENTIFIER_COLUMNS)
    columns.update(SHARED_SOURCE_COLUMNS.values())
    for period in PERIODS.values():
        columns.update(period.values())
    return sorted(columns)


def resolve_artifact(model_dir: Path, candidates: tuple[str, ...]) -> Path:
    for filename in candidates:
        candidate = model_dir / filename
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"None of the expected artifacts exists in {model_dir}: {', '.join(candidates)}"
    )


def normalize_preprocessing(preprocessing: dict, feature_names: list[str]) -> dict:
    """Accept both the journal-release and legacy preprocessing schemas."""
    normalized = dict(preprocessing)
    normalized.setdefault(
        "species_columns",
        [name for name in feature_names if name.startswith("Species_")],
    )
    normalized.setdefault(
        "environment_features",
        [
            name
            for name in feature_names
            if name != "Log_Height"
            and not name.startswith("Species_")
            and not name.startswith("Period_")
        ],
    )
    if "no_period_medians" not in normalized:
        if "feature_medians" not in normalized:
            raise KeyError(
                "Preprocessing artifact contains neither no_period_medians nor feature_medians"
            )
        normalized["no_period_medians"] = normalized["feature_medians"]
    return normalized


def validate_inputs(args: argparse.Namespace) -> None:
    model_path = resolve_artifact(args.model_dir, MODEL_FILENAMES)
    preprocessing_path = resolve_artifact(args.model_dir, PREPROCESSING_FILENAMES)
    for path in [
        args.input,
        model_path,
        preprocessing_path,
        args.training_data,
        args.pipeline_script,
        args.soil_script,
    ]:
        if not path.exists():
            raise FileNotFoundError(path)
    if (
        args.fixed_suitability_thresholds is not None
        and not args.fixed_suitability_thresholds.exists()
    ):
        raise FileNotFoundError(args.fixed_suitability_thresholds)
    if args.output.exists():
        raise FileExistsError(
            f"Output directory already exists; choose a versioned path: {args.output}"
        )
    if args.reference_height_m <= 0:
        raise ValueError("--reference-height-m must be positive")
    if args.resolution <= 0:
        raise ValueError("--resolution must be positive")

    available = pd.read_csv(args.input, nrows=0).columns.tolist()
    missing = sorted(set(required_input_columns()) - set(available))
    if missing:
        raise ValueError(f"Input CSV is missing columns: {missing}")


def scan_grid(args: argparse.Namespace) -> dict:
    started = time.perf_counter()
    rows = 0
    min_x = math.inf
    max_x = -math.inf
    min_y = math.inf
    max_y = -math.inf
    source_ranges = {
        column: [math.inf, -math.inf] for column in required_input_columns()
        if column not in IDENTIFIER_COLUMNS
    }

    for chunk_index, chunk in enumerate(
        pd.read_csv(
            args.input,
            usecols=required_input_columns(),
            chunksize=args.chunk_size,
        ),
        start=1,
    ):
        if args.max_rows is not None:
            remaining = args.max_rows - rows
            if remaining <= 0:
                break
            chunk = chunk.iloc[:remaining].copy()
        if chunk.empty:
            break

        rows += len(chunk)
        min_x = min(min_x, float(chunk["X"].min()))
        max_x = max(max_x, float(chunk["X"].max()))
        min_y = min(min_y, float(chunk["Y"].min()))
        max_y = max(max_y, float(chunk["Y"].max()))
        for column, bounds in source_ranges.items():
            values = pd.to_numeric(chunk[column], errors="coerce")
            finite = values[np.isfinite(values)]
            if not finite.empty:
                bounds[0] = min(bounds[0], float(finite.min()))
                bounds[1] = max(bounds[1], float(finite.max()))

        print(
            f"[scan] chunk={chunk_index:,} rows={rows:,} "
            f"x=({min_x:.3f},{max_x:.3f}) y=({min_y:.3f},{max_y:.3f})",
            flush=True,
        )

    width = int(round((max_x - min_x) / args.resolution)) + 1
    height = int(round((max_y - min_y) / args.resolution)) + 1
    grid_cells = width * height
    metadata = {
        "input_rows": rows,
        "min_x_center": min_x,
        "max_x_center": max_x,
        "min_y_center": min_y,
        "max_y_center": max_y,
        "resolution": args.resolution,
        "width": width,
        "height": height,
        "full_grid_cells": grid_cells,
        "input_coverage_fraction": rows / grid_cells,
        "crs": args.crs,
        "source_ranges": {
            key: {
                "min": None if bounds[0] == math.inf else bounds[0],
                "max": None if bounds[1] == -math.inf else bounds[1],
            }
            for key, bounds in source_ranges.items()
        },
        "scan_seconds": time.perf_counter() - started,
        "max_rows_testing_limit": args.max_rows,
    }
    return metadata


def load_model_and_training(
    args: argparse.Namespace,
) -> tuple[xgb.Booster, dict, pd.DataFrame, np.ndarray, pd.DataFrame]:
    model = xgb.Booster()
    model_path = resolve_artifact(args.model_dir, MODEL_FILENAMES)
    preprocessing_path = resolve_artifact(args.model_dir, PREPROCESSING_FILENAMES)
    model.load_model(model_path)
    model.set_param({"nthread": max(1, (os_cpu_count() or 2) - 1)})

    preprocessing = joblib.load(preprocessing_path)
    feature_names = list(model.feature_names or [])
    preprocessing = normalize_preprocessing(preprocessing, feature_names)
    expected = (
        ["Log_Height"]
        + list(preprocessing["species_columns"])
        + list(preprocessing["environment_features"])
    )
    if feature_names != expected:
        raise ValueError(
            "No-period model feature order differs from preprocessing metadata.\n"
            f"Model: {feature_names}\nMetadata: {expected}"
        )

    pipeline = load_pipeline_module(args.pipeline_script)
    raw = pd.read_csv(args.training_data)
    if any(feature in feature_names for feature in ("soil_infill", "soil_bedrock", "soil_moraine")):
        soil = load_pipeline_module(args.soil_script)
        raw, _ = soil.add_soil_indicators(raw)
    long_data, _ = pipeline.build_long_data(raw)
    model_data, encoding = pipeline.add_split_and_dummies(long_data)
    if list(encoding["species_columns"]) != list(preprocessing["species_columns"]):
        raise ValueError("Training reconstruction produced different species columns")

    development = model_data["Split"].eq("Development")
    medians = preprocessing["no_period_medians"].reindex(feature_names).astype(float)
    x_development = (
        model_data.loc[development, feature_names]
        .fillna(medians)
        .astype(np.float32)
    )
    # Derive one common seven-level scale from the development environments,
    # while matching the standardized deployment scenario used by the maps.
    # Each development environment is evaluated once for every observed
    # species/category at the common reference height and park context. This
    # avoids deriving map classes from a mixture of observed tree heights and
    # unequal species frequencies.
    standardized_predictions = []
    species_columns = list(preprocessing["species_columns"])
    for species_name in SPECIES.values():
        scenario = x_development.copy()
        scenario["Log_Height"] = np.float32(np.log(args.reference_height_m))
        scenario["type_Puisto"] = np.float32(args.park_context)
        scenario.loc[:, species_columns] = np.float32(0.0)
        scenario[f"Species_{species_name}"] = np.float32(1.0)
        standardized_predictions.append(
            inverse_percentage(
                model.inplace_predict(scenario.to_numpy(dtype=np.float32))
            )
        )
    thresholds = np.quantile(
        np.concatenate(standardized_predictions),
        np.arange(1, 7, dtype=float) / 7.0,
    ).astype(np.float32)
    if args.fixed_suitability_thresholds is not None:
        threshold_table = pd.read_csv(args.fixed_suitability_thresholds)
        if "Threshold_Group" in threshold_table.columns:
            threshold_table = threshold_table.loc[
                threshold_table["Threshold_Group"].astype(str).eq("Overall")
            ]
        value_column = (
            "Threshold"
            if "Threshold" in threshold_table.columns
            else "Annual_Growth_Percent_Threshold"
            if "Annual_Growth_Percent_Threshold" in threshold_table.columns
            else None
        )
        if value_column is None:
            raise ValueError(
                "Fixed-threshold CSV must contain 'Threshold' or "
                "'Annual_Growth_Percent_Threshold'."
            )
        if "Threshold_Number" in threshold_table.columns:
            threshold_table = threshold_table.sort_values("Threshold_Number")
        thresholds = threshold_table[value_column].to_numpy(dtype=np.float32)
        if len(thresholds) != 6 or not np.all(np.isfinite(thresholds)):
            raise ValueError("Exactly six finite fixed suitability thresholds are required")
        if not np.all(np.diff(thresholds) > 0):
            raise ValueError("Fixed suitability thresholds must be strictly increasing")

    domain_rows = []
    domain_features = [
        feature
        for feature in feature_names
        if not feature.startswith("Species_") and not feature.startswith("Period_")
    ]
    for feature in domain_features:
        values = pd.to_numeric(
            model_data.loc[development, feature], errors="coerce"
        ).dropna()
        domain_rows.append(
            {
                "Feature": feature,
                "Median": float(medians[feature]),
                "Min": float(values.min()),
                "P01": float(values.quantile(0.01)),
                "P99": float(values.quantile(0.99)),
                "Max": float(values.max()),
            }
        )
    domain = pd.DataFrame(domain_rows).set_index("Feature")
    return model, preprocessing, domain, thresholds, x_development


def os_cpu_count() -> int | None:
    try:
        import os

        return os.cpu_count()
    except Exception:
        return None


def build_reference_predictions(
    model: xgb.Booster,
    preprocessing: dict,
    reference_height_m: float,
    park_context: float,
) -> pd.DataFrame:
    feature_names = list(model.feature_names or [])
    medians = preprocessing["no_period_medians"].reindex(feature_names).astype(float)
    matrix = np.tile(
        medians.to_numpy(dtype=np.float32),
        (len(SPECIES), 1),
    )
    matrix[:, feature_names.index("Log_Height")] = np.log(reference_height_m)
    matrix[:, feature_names.index("type_Puisto")] = park_context
    species_columns = list(preprocessing["species_columns"])
    for column in species_columns:
        matrix[:, feature_names.index(column)] = 0.0
    for row_index, (species_code, species_name) in enumerate(SPECIES.items()):
        matrix[
            row_index,
            feature_names.index(f"Species_{species_name}"),
        ] = 1.0

    predicted_y = model.inplace_predict(matrix)
    predicted_pct = inverse_percentage(predicted_y)
    return pd.DataFrame(
        {
            "Species_Code": list(SPECIES),
            "Species_Name": list(SPECIES.values()),
            "Reference_Height_m": reference_height_m,
            "Park_Context": park_context,
            "Reference_LogSGR": predicted_y.astype(np.float32),
            "Reference_Annual_Growth_Percent": predicted_pct,
        }
    )


def clean_period_environment(
    chunk: pd.DataFrame,
    period: str,
    preprocessing: dict,
    park_context: float,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    medians = preprocessing["no_period_medians"]
    source_map = {
        **SHARED_SOURCE_COLUMNS,
        **PERIODS[period],
    }
    cleaned: dict[str, np.ndarray] = {}
    missing: dict[str, np.ndarray] = {}
    active_environment = [
        feature
        for feature in preprocessing["environment_features"]
        if feature != "type_Puisto"
    ]
    for feature in active_environment:
        values = pd.to_numeric(
            chunk[source_map[feature]], errors="coerce"
        ).to_numpy(dtype=np.float64, copy=True)
        invalid = ~np.isfinite(values) | (values <= -9990.0)
        if feature == "avg_noise_day":
            # The source uses the sentinel for locations below the mapped
            # noise floor. Treat these as quiet cells rather than missing data,
            # as requested by the data owner.
            values[invalid] = QUIET_NOISE_DB
            invalid = np.zeros(len(values), dtype=bool)
        if feature == "lightemiss":
            invalid |= values < 0.0
        missing[feature] = invalid
        values[invalid] = float(medians[feature])
        cleaned[feature] = values.astype(np.float32)
    cleaned["type_Puisto"] = np.full(
        len(chunk), park_context, dtype=np.float32
    )
    missing["type_Puisto"] = np.zeros(len(chunk), dtype=bool)
    return cleaned, missing


def reliability_flags(
    cleaned: dict[str, np.ndarray],
    missing: dict[str, np.ndarray],
    domain: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    reliable_minmax = np.ones(len(next(iter(cleaned.values()))), dtype=bool)
    reliable_robust = reliable_minmax.copy()
    outside_count = np.zeros(len(reliable_minmax), dtype=np.uint8)
    for feature in cleaned:
        if feature == "type_Puisto" or feature not in domain.index:
            continue
        values = cleaned[feature]
        out_minmax = (
            missing[feature]
            | (values < float(domain.loc[feature, "Min"]))
            | (values > float(domain.loc[feature, "Max"]))
        )
        out_robust = (
            missing[feature]
            | (values < float(domain.loc[feature, "P01"]))
            | (values > float(domain.loc[feature, "P99"]))
        )
        reliable_minmax &= ~out_minmax
        reliable_robust &= ~out_robust
        outside_count += out_minmax.astype(np.uint8)
    return reliable_minmax, reliable_robust, outside_count


def grid_indices(
    x: np.ndarray,
    y: np.ndarray,
    grid: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    resolution = float(grid["resolution"])
    columns_float = (x - float(grid["min_x_center"])) / resolution
    rows_float = (float(grid["max_y_center"]) - y) / resolution
    columns = np.rint(columns_float).astype(np.int64)
    rows = np.rint(rows_float).astype(np.int64)
    residual = np.maximum(
        np.abs(columns_float - columns),
        np.abs(rows_float - rows),
    )
    valid = (
        (residual <= 1.0e-4)
        & (columns >= 0)
        & (columns < int(grid["width"]))
        & (rows >= 0)
        & (rows < int(grid["height"]))
    )
    return rows, columns, valid


def allocate_memmaps(
    temp_dir: Path,
    grid: dict,
) -> tuple[dict[str, np.memmap], dict[str, np.memmap], np.memmap]:
    height = int(grid["height"])
    width = int(grid["width"])
    predictions = {}
    reliability = {}
    for period in PERIODS:
        prediction = np.memmap(
            temp_dir / f"prediction_{period}.dat",
            mode="w+",
            dtype="float32",
            shape=(len(SPECIES), height, width),
        )
        prediction[:] = PREDICTION_NODATA
        predictions[period] = prediction
        mask = np.memmap(
            temp_dir / f"reliability_{period}.dat",
            mode="w+",
            dtype="uint8",
            shape=(3, height, width),
        )
        mask[:] = RELIABILITY_NODATA
        reliability[period] = mask
    seen = np.memmap(
        temp_dir / "seen.dat",
        mode="w+",
        dtype="uint8",
        shape=(height, width),
    )
    seen[:] = 0
    return predictions, reliability, seen


def prediction_matrix(
    model: xgb.Booster,
    preprocessing: dict,
    cleaned: dict[str, np.ndarray],
    reference_height_m: float,
) -> np.ndarray:
    feature_names = list(model.feature_names or [])
    n_rows = len(next(iter(cleaned.values())))
    matrix = np.zeros((n_rows, len(feature_names)), dtype=np.float32)
    matrix[:, feature_names.index("Log_Height")] = np.log(reference_height_m)
    for feature in preprocessing["environment_features"]:
        matrix[:, feature_names.index(feature)] = cleaned[feature]
    return matrix


def predict_species_scenarios(
    model: xgb.Booster,
    preprocessing: dict,
    base_matrix: np.ndarray,
) -> np.ndarray:
    feature_names = list(model.feature_names or [])
    species_columns = list(preprocessing["species_columns"])
    species_positions = [feature_names.index(column) for column in species_columns]
    predictions = np.empty(
        (len(SPECIES), len(base_matrix)),
        dtype=np.float32,
    )
    for position in species_positions:
        base_matrix[:, position] = 0.0
    for output_index, (_, species_name) in enumerate(SPECIES.items()):
        position = feature_names.index(f"Species_{species_name}")
        base_matrix[:, position] = 1.0
        predictions[output_index] = inverse_percentage(
            model.inplace_predict(base_matrix)
        )
        base_matrix[:, position] = 0.0
    return predictions


def write_geotiffs(
    output: Path,
    grid: dict,
    predictions: dict[str, np.memmap],
    reliability: dict[str, np.memmap],
    references: pd.DataFrame,
    thresholds: np.ndarray,
    crs: str,
) -> list[Path]:
    height = int(grid["height"])
    width = int(grid["width"])
    resolution = float(grid["resolution"])
    transform = from_origin(
        float(grid["min_x_center"]) - resolution / 2.0,
        float(grid["max_y_center"]) + resolution / 2.0,
        resolution,
        resolution,
    )
    base_profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "crs": crs,
        "transform": transform,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "compress": "DEFLATE",
        "predictor": 3,
        "zlevel": 6,
        "BIGTIFF": "IF_SAFER",
    }
    reference_lookup = references.set_index("Species_Code")[
        "Reference_Annual_Growth_Percent"
    ].to_dict()
    paths = []

    for period in PERIODS:
        pct_path = output / f"relative_growth_pct_{period}.tif"
        deviation_path = output / f"environmental_deviation_pp_{period}.tif"
        class_path = output / f"suitability_level_{period}.tif"
        zone_path = output / f"diagnostic_zone_{period}.tif"
        reliability_path = output / f"reliability_{period}.tif"
        paths.extend([pct_path, deviation_path, class_path, zone_path, reliability_path])

        pct_profile = {
            **base_profile,
            "count": len(SPECIES),
            "dtype": "float32",
            "nodata": PREDICTION_NODATA,
        }
        class_profile = {
            **base_profile,
            "count": len(SPECIES),
            "dtype": "uint8",
            "nodata": 0,
            "predictor": 2,
        }
        reliability_profile = {
            **base_profile,
            "count": 3,
            "dtype": "uint8",
            "nodata": RELIABILITY_NODATA,
            "predictor": 2,
        }

        with (
            rasterio.open(pct_path, "w", **pct_profile) as pct_dst,
            rasterio.open(deviation_path, "w", **pct_profile) as deviation_dst,
            rasterio.open(class_path, "w", **class_profile) as class_dst,
            rasterio.open(zone_path, "w", **class_profile) as zone_dst,
            rasterio.open(
                reliability_path, "w", **reliability_profile
            ) as reliability_dst,
        ):
            for band_index, (species_code, species_name) in enumerate(
                SPECIES.items(), start=1
            ):
                pct = np.asarray(
                    predictions[period][band_index - 1],
                    dtype=np.float32,
                )
                valid = pct != PREDICTION_NODATA
                deviation = np.full_like(pct, PREDICTION_NODATA)
                deviation[valid] = (
                    pct[valid] - float(reference_lookup[species_code])
                )
                suitability = np.zeros((height, width), dtype=np.uint8)
                suitability[valid] = (
                    np.digitize(pct[valid], thresholds, right=False) + 1
                ).astype(np.uint8)
                diagnostic_zone = np.zeros((height, width), dtype=np.uint8)
                diagnostic_zone[valid] = np.where(
                    suitability[valid] <= 2,
                    1,
                    np.where(suitability[valid] <= 5, 2, 3),
                ).astype(np.uint8)

                pct_dst.write(pct, band_index)
                deviation_dst.write(deviation, band_index)
                class_dst.write(suitability, band_index)
                zone_dst.write(diagnostic_zone, band_index)
                for dataset in [pct_dst, deviation_dst, class_dst, zone_dst]:
                    dataset.set_band_description(
                        band_index,
                        f"Sp{species_code}_{species_name}",
                    )
                    dataset.update_tags(
                        band_index,
                        species_code=species_code,
                        species_name=species_name,
                        reference_height_m=references[
                            "Reference_Height_m"
                        ].iloc[0],
                        park_context=references["Park_Context"].iloc[0],
                        period_environment=period,
                    )
                zone_dst.update_tags(
                    band_index,
                    zone_1="Constrained (levels 1-2)",
                    zone_2="Typical (levels 3-5)",
                    zone_3="Favorable (levels 6-7)",
                )

            reliability_dst.write(
                np.asarray(reliability[period][0], dtype=np.uint8), 1
            )
            reliability_dst.write(
                np.asarray(reliability[period][1], dtype=np.uint8), 2
            )
            reliability_dst.write(
                np.asarray(reliability[period][2], dtype=np.uint8), 3
            )
            reliability_dst.set_band_description(1, "Reliable_MinMax")
            reliability_dst.set_band_description(2, "Within_P01_P99")
            reliability_dst.set_band_description(3, "Outside_MinMax_Count")

        print(f"[raster] wrote period {period}", flush=True)
    return paths


def write_metadata_tables(
    output: Path,
    grid: dict,
    domain: pd.DataFrame,
    thresholds: np.ndarray,
    references: pd.DataFrame,
) -> None:
    (output / "grid_metadata.json").write_text(
        json.dumps(grid, indent=2),
        encoding="utf-8",
    )
    domain.reset_index().to_csv(output / "training_domain.csv", index=False)
    threshold_rows = []
    lower = -math.inf
    for level in range(1, 8):
        upper = float(thresholds[level - 1]) if level <= 6 else math.inf
        threshold_rows.append(
            {
                "Level": level,
                "Lower_Annual_Growth_Percent_Inclusive": lower,
                "Upper_Annual_Growth_Percent_Exclusive": upper,
            }
        )
        lower = upper
    pd.DataFrame(threshold_rows).to_csv(
        output / "suitability_thresholds.csv",
        index=False,
    )
    pd.DataFrame(
        [
            {
                "Zone": 1,
                "Diagnostic_Label": "Constrained",
                "Included_Levels": "1-2",
                "Lower_Annual_Growth_Percent_Inclusive": -math.inf,
                "Upper_Annual_Growth_Percent_Exclusive": float(thresholds[1]),
            },
            {
                "Zone": 2,
                "Diagnostic_Label": "Typical",
                "Included_Levels": "3-5",
                "Lower_Annual_Growth_Percent_Inclusive": float(thresholds[1]),
                "Upper_Annual_Growth_Percent_Exclusive": float(thresholds[4]),
            },
            {
                "Zone": 3,
                "Diagnostic_Label": "Favorable",
                "Included_Levels": "6-7",
                "Lower_Annual_Growth_Percent_Inclusive": float(thresholds[4]),
                "Upper_Annual_Growth_Percent_Exclusive": math.inf,
            },
        ]
    ).to_csv(output / "diagnostic_zones.csv", index=False)
    references.to_csv(output / "reference_predictions.csv", index=False)


def run_predictions(
    args: argparse.Namespace,
    grid: dict,
    model: xgb.Booster,
    preprocessing: dict,
    domain: pd.DataFrame,
    thresholds: np.ndarray,
    references: pd.DataFrame,
) -> dict:
    output = args.output
    temp_dir = output / "_temporary_memmaps"
    temp_dir.mkdir()
    prediction_maps, reliability_maps, seen = allocate_memmaps(temp_dir, grid)
    rows_processed = 0
    duplicate_cells = 0
    invalid_grid_coordinates = 0
    csv_path = output / "spatial_relative_growth_predictions.csv"
    first_csv_chunk = True
    started = time.perf_counter()

    try:
        for chunk_index, chunk in enumerate(
            pd.read_csv(
                args.input,
                usecols=required_input_columns(),
                chunksize=args.chunk_size,
            ),
            start=1,
        ):
            if args.max_rows is not None:
                remaining = args.max_rows - rows_processed
                if remaining <= 0:
                    break
                chunk = chunk.iloc[:remaining].copy()
            if chunk.empty:
                break

            rows, columns, valid_grid = grid_indices(
                chunk["X"].to_numpy(dtype=np.float64),
                chunk["Y"].to_numpy(dtype=np.float64),
                grid,
            )
            invalid_grid_coordinates += int((~valid_grid).sum())
            if not np.all(valid_grid):
                chunk = chunk.loc[valid_grid].reset_index(drop=True)
                rows = rows[valid_grid]
                columns = columns[valid_grid]
            already_seen = seen[rows, columns] != 0
            duplicate_cells += int(already_seen.sum())
            seen[rows, columns] = 1

            csv_chunk = chunk[IDENTIFIER_COLUMNS].copy() if args.write_csv else None
            for period in PERIODS:
                cleaned, missing = clean_period_environment(
                    chunk,
                    period,
                    preprocessing,
                    args.park_context,
                )
                minmax, robust, outside_count = reliability_flags(
                    cleaned,
                    missing,
                    domain,
                )
                base = prediction_matrix(
                    model,
                    preprocessing,
                    cleaned,
                    args.reference_height_m,
                )
                pct = predict_species_scenarios(model, preprocessing, base)
                for species_index, species_code in enumerate(SPECIES):
                    prediction_maps[period][
                        species_index,
                        rows,
                        columns,
                    ] = pct[species_index]
                    if csv_chunk is not None:
                        csv_chunk[
                            f"Pct_{period}_Sp{species_code}"
                        ] = np.round(pct[species_index], 4)
                        level = (
                            np.digitize(
                                pct[species_index], thresholds, right=False
                            )
                            + 1
                        ).astype(np.uint8)
                        zone = np.where(
                            level <= 2,
                            1,
                            np.where(level <= 5, 2, 3),
                        ).astype(np.uint8)
                        csv_chunk[f"Level_{period}_Sp{species_code}"] = level
                        csv_chunk[f"Zone_{period}_Sp{species_code}"] = zone
                reliability_maps[period][0, rows, columns] = minmax.astype(np.uint8)
                reliability_maps[period][1, rows, columns] = robust.astype(np.uint8)
                reliability_maps[period][2, rows, columns] = outside_count
                if csv_chunk is not None:
                    csv_chunk[f"Reliable_{period}"] = minmax.astype(np.uint8)
                    csv_chunk[f"WithinP01P99_{period}"] = robust.astype(np.uint8)
                    csv_chunk[f"OutsideCount_{period}"] = outside_count

            if csv_chunk is not None:
                csv_chunk.to_csv(
                    csv_path,
                    mode="w" if first_csv_chunk else "a",
                    header=first_csv_chunk,
                    index=False,
                    float_format="%.4f",
                )
                first_csv_chunk = False

            rows_processed += len(chunk)
            elapsed = time.perf_counter() - started
            rate = rows_processed / elapsed if elapsed > 0 else 0.0
            print(
                f"[predict] chunk={chunk_index:,} rows={rows_processed:,} "
                f"rate={rate:,.0f} rows/s elapsed={elapsed/60:.1f} min",
                flush=True,
            )

        for mapping in [prediction_maps, reliability_maps]:
            for memmap in mapping.values():
                memmap.flush()
        seen.flush()

        raster_paths = write_geotiffs(
            output,
            grid,
            prediction_maps,
            reliability_maps,
            references,
            thresholds,
            args.crs,
        )
    finally:
        for mapping_name in ["prediction_maps", "reliability_maps"]:
            mapping = locals().get(mapping_name, {})
            for memmap in mapping.values():
                try:
                    memmap._mmap.close()
                except Exception:
                    pass
        try:
            seen._mmap.close()
        except Exception:
            pass
        shutil.rmtree(temp_dir, ignore_errors=True)

    return {
        "rows_processed": rows_processed,
        "duplicate_grid_cells": duplicate_cells,
        "invalid_grid_coordinates": invalid_grid_coordinates,
        "prediction_seconds": time.perf_counter() - started,
        "csv_written": args.write_csv,
        "csv_path": str(csv_path) if args.write_csv else None,
        "raster_paths": [str(path) for path in raster_paths],
    }


def write_run_log(
    args: argparse.Namespace,
    grid: dict,
    run: dict | None,
    thresholds: np.ndarray | None,
) -> None:
    lines = [
        "# Spatial relative-growth diagnosis run log",
        "",
        f"- Input: `{args.input}`",
        f"- Deployment model: `{args.model_dir / 'xgb_no_period.json'}`",
        "- Target: `y = ln(([ln(C_end)-ln(C_start)]/years))`",
        "- Map output: annual compound relative growth percentage",
        f"- Reference height: {args.reference_height_m:g} m",
        f"- Park-context indicator: {args.park_context:g}",
        "- Species representation: exactly one pooled-model one-hot category",
        "- Monitoring-period indicators: omitted for deployment comparability",
        f"- CRS: {args.crs}",
        f"- Resolution: {args.resolution:g} m",
        f"- Input rows scanned: {grid['input_rows']:,}",
        (
            f"- Raster dimensions: {grid['width']:,} x {grid['height']:,} "
            f"({grid['full_grid_cells']:,} cells)"
        ),
        (
            "- Reliability: cells with missing/sentinel values (except the noise "
            "floor described below) or any environmental feature outside the "
            "development-data min-max domain are flagged."
        ),
        (
            f"- Missing/sentinel daytime-noise cells are interpreted as quiet "
            f"locations and assigned {QUIET_NOISE_DB:g} dB; they are not flagged "
            "as missing."
        ),
        (
            "- Robust-domain band: additionally reports whether all environmental "
            "features fall within their development-data P01-P99 ranges."
        ),
        (
            "- Suitability thresholds: validation-selected fixed tree-level "
            f"thresholds loaded from `{args.fixed_suitability_thresholds}`."
            if args.fixed_suitability_thresholds is not None
            else
            "- Suitability thresholds are legacy fixed septiles from pooled "
            "development environments evaluated for all 11 species/categories "
            f"at {args.reference_height_m:g} m and park={args.park_context:g}."
        ),
        (
            "- Primary diagnostic zones: constrained = levels 1-2, typical = "
            "levels 3-5, favorable = levels 6-7; seven levels are retained as "
            "detailed sublevels."
        ),
        "- Environmental deviation is local percentage minus the species-specific "
        "median-environment reference prediction.",
        "- No kg C map is produced because the grid has no explicit initial carbon stock.",
    ]
    if thresholds is not None:
        lines.append(
            "- Level cut points (annual percentage): "
            + ", ".join(f"{value:.4f}" for value in thresholds)
        )
    if run is not None:
        lines.extend(
            [
                f"- Rows predicted: {run['rows_processed']:,}",
                f"- Duplicate grid cells: {run['duplicate_grid_cells']:,}",
                f"- Invalid grid coordinates: {run['invalid_grid_coordinates']:,}",
                f"- Prediction and raster time: {run['prediction_seconds']/60:.1f} min",
                f"- Wide CSV written: {run['csv_written']}",
            ]
        )
    (args.output / "RUN_LOG.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    validate_inputs(args)
    args.output.mkdir(parents=True)
    grid = scan_grid(args)
    (args.output / "grid_metadata.json").write_text(
        json.dumps(grid, indent=2),
        encoding="utf-8",
    )

    if args.scan_only:
        write_run_log(args, grid, run=None, thresholds=None)
        print(json.dumps(grid, indent=2), flush=True)
        return

    model, preprocessing, domain, thresholds, _ = load_model_and_training(args)
    references = build_reference_predictions(
        model,
        preprocessing,
        args.reference_height_m,
        args.park_context,
    )
    write_metadata_tables(args.output, grid, domain, thresholds, references)
    run = run_predictions(
        args,
        grid,
        model,
        preprocessing,
        domain,
        thresholds,
        references,
    )
    write_run_log(args, grid, run, thresholds)
    (args.output / "run_metadata.json").write_text(
        json.dumps(
            {
                "arguments": {
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in vars(args).items()
                },
                "grid": grid,
                "run": run,
                "suitability_thresholds": thresholds.tolist(),
                "suitability_threshold_method": (
                    "validation-selected fixed tree-level annual-growth thresholds"
                    if args.fixed_suitability_thresholds is not None
                    else
                    "legacy fixed septiles of development environments evaluated for all "
                    "11 species/categories at the mapped reference height and park context"
                ),
                "suitability_threshold_source": (
                    str(args.fixed_suitability_thresholds)
                    if args.fixed_suitability_thresholds is not None
                    else None
                ),
                "diagnostic_zone_mapping": {
                    "1": "Constrained (levels 1-2)",
                    "2": "Typical (levels 3-5)",
                    "3": "Favorable (levels 6-7)",
                },
                "quiet_noise_replacement_db": QUIET_NOISE_DB,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Completed: {args.output}", flush=True)


if __name__ == "__main__":
    main()
