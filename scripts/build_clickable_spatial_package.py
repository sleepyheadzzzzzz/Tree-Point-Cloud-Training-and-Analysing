"""Generate matched suitability, deviation and change GeoTIFFs for click SHAP.

No model fitting occurs. Inputs are an aligned point grid with X/Y coordinates
and environmental columns (canonical per-period names or the legacy grid names).
Soil columns are mandatory; no spatial soil values are invented. Missing noise
is the data-owner-defined 40 dB quiet floor; other missing inputs produce NoData.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import math
import shutil
import tempfile
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import rasterio
import xgboost as xgb
from rasterio.transform import from_origin
from rasterio.windows import Window

from spatial_waterfall_core import (ENVIRONMENT, GROUPS, SPECIES, NODATA, domain_codes,
                                    matrix_from_environment, percentage, sha256)

PERIODS = ["15_17", "17_21", "21_23"]
LEGACY = {
    "15_17": dict(Density25="Density_15", Mono_Rate="Mono_Rate_", avg_svf="svf15_17", avg_radiation="RA15_17", avg_LST="LST15_17"),
    "17_21": dict(Density25="Density_17", Mono_Rate="Mono_Rate1", avg_svf="svf17_21", avg_radiation="RA17_21", avg_LST="LST17_21"),
    "21_23": dict(Density25="Density_21", Mono_Rate="Mono_Rat_1", avg_svf="svf21_23", avg_radiation="RA21_23", avg_LST="LST21_23"),
}
ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", type=Path, default=ROOT / "models/xgb_spatial_deployment_no_period_three_soil.json")
    p.add_argument("--preprocessing", type=Path, default=ROOT / "models/preprocessing_spatial_deployment.joblib")
    p.add_argument("--domain", type=Path, default=ROOT / "results/spatial_validation/development_training_domain.csv")
    p.add_argument("--thresholds", type=Path, default=ROOT / "results/suitability/fixed_selected_seven_level_thresholds.csv")
    p.add_argument("--template-raster", type=Path, help="Use its exact extent/grid; source points outside it are excluded")
    p.add_argument("--resolution", type=float, default=2.0)
    p.add_argument("--crs", default="EPSG:3879", help="Coordinate reference system of input X/Y")
    p.add_argument("--height", type=float, default=10.0)
    p.add_argument("--park-context", type=int, choices=[0, 1], default=1, help="Fallback only when no park column is supplied; also reference park setting")
    p.add_argument("--periods", nargs="+", default=PERIODS, choices=PERIODS)
    p.add_argument("--earlier", default="15_17", choices=PERIODS)
    p.add_argument("--later", default="21_23", choices=PERIODS)
    p.add_argument("--chunk-size", type=int, default=100000)
    p.add_argument("--scope", choices=["wall_to_wall_input_grid", "observed_cells_demo"], default="wall_to_wall_input_grid")
    return p.parse_args()


def resolve_columns(columns, periods):
    sources = {}
    for period in periods:
        sources[period] = {}
        for feature in ENVIRONMENT:
            candidates = [f"{feature}_{period}", LEGACY[period].get(feature, ""), feature]
            if feature == "avg_noise_day":
                candidates += [f"noise_{period}", "noise"]
            name = next((c for c in candidates if c and c in columns), None)
            if name is None and feature != "type_Puisto":
                raise ValueError(f"Missing {feature} for {period}; supply a canonical or legacy grid column. Soil is required.")
            sources[period][feature] = name
    return sources


def grid_from_input(args):
    if args.template_raster:
        with rasterio.open(args.template_raster) as src:
            if src.crs != rasterio.crs.CRS.from_user_input(args.crs):
                raise ValueError("Template CRS differs from input X/Y CRS; explicitly reproject inputs first")
            transform = src.transform
            if transform.b != 0 or transform.d != 0 or transform.a <= 0 or transform.e >= 0:
                raise ValueError("Template must be a north-up raster")
            return src.width, src.height, transform
    minima, maxima = np.full(2, np.inf), np.full(2, -np.inf)
    for chunk in pd.read_csv(args.input, usecols=["X", "Y"], chunksize=args.chunk_size):
        points = chunk[["X", "Y"]].to_numpy(float)
        if not np.isfinite(points).all():
            raise ValueError("Input coordinates must be finite")
        minima = np.minimum(minima, points.min(axis=0))
        maxima = np.maximum(maxima, points.max(axis=0))
    if not np.isfinite([minima, maxima]).all():
        raise ValueError("Empty input grid")
    dimensions = np.rint((maxima-minima)/args.resolution).astype(int)+1
    return int(dimensions[0]), int(dimensions[1]), from_origin(minima[0]-args.resolution/2, maxima[1]+args.resolution/2, args.resolution, args.resolution)


def windows(width, height, size=256):
    for row in range(0, height, size):
        for col in range(0, width, size):
            yield Window(col, row, min(size, width-col), min(size, height-row))


def profile(width, height, transform, crs, count, dtype, nodata):
    return dict(driver="GTiff", width=width, height=height, transform=transform, crs=crs,
                count=count, dtype=dtype, nodata=nodata, compress="deflate", tiled=True,
                blockxsize=256, blockysize=256, BIGTIFF="IF_SAFER", predictor=3 if dtype == "float32" else 2)


def set_descriptions(dst, labels, unit):
    for band, label in enumerate(labels, 1):
        dst.set_band_description(band, label)
        dst.update_tags(band, unit=unit)


def build(args):
    started = time.perf_counter()
    if args.output.exists():
        raise FileExistsError("Choose a new versioned output directory")
    if not np.isfinite(args.height) or args.height <= 0 or args.resolution <= 0:
        raise ValueError("Positive finite height and resolution are required")
    if args.earlier == args.later or args.earlier not in args.periods or args.later not in args.periods:
        raise ValueError("Both distinct change endpoints must be included")
    available = pd.read_csv(args.input, nrows=0).columns
    sources = resolve_columns(available, args.periods)
    width, height, transform = grid_from_input(args)
    booster = xgb.Booster(params={"nthread": 4})
    booster.load_model(args.model)
    preprocessing = joblib.load(args.preprocessing)  # trusted scientific artifact only
    if preprocessing.get("use_scaled", False):
        raise ValueError("This exporter expects the unscaled frozen XGBoost model")
    reference = {f: float(preprocessing["feature_medians"][f]) for f in ENVIRONMENT}
    reference["type_Puisto"] = float(args.park_context)
    reference_env = np.array([reference[f] for f in ENVIRONMENT], np.float32)
    reference_predictions = [float(percentage(booster.inplace_predict(matrix_from_environment(booster, reference_env, args.height, s)))[0]) for s in SPECIES]
    domain_frame = pd.read_csv(args.domain).set_index("Feature")
    domain = {f: {key: float(domain_frame.loc[f, key]) for key in ["Minimum", "Maximum", "P01", "P99"]} for f in ["Log_Height", *ENVIRONMENT]}
    threshold_frame = pd.read_csv(args.thresholds)
    if "Threshold_Group" in threshold_frame:
        threshold_frame = threshold_frame[threshold_frame.Threshold_Group == "Overall"]
    thresholds = threshold_frame.sort_values("Threshold_Number")["Threshold"].to_numpy(float)
    if len(thresholds) != 6 or not np.isfinite(thresholds).all() or not (np.diff(thresholds) > 0).all():
        raise ValueError("Six fixed increasing overall thresholds are required")
    args.output.mkdir(parents=True)
    rasters = args.output / "rasters"
    rasters.mkdir()
    shutil.copy2(args.model, args.output / "model.json")
    period_files = {period: {kind: f"rasters/{kind}_{period}.tif" for kind in ["environment", "growth", "deviation", "suitability", "reliability", "within_p01_p99"]} for period in args.periods}
    stats = {p: dict(valid_cells=0, out_of_range_cells=0, quiet_noise_fills=0) for p in args.periods}
    total, included = 0, 0
    with tempfile.TemporaryDirectory(prefix="raster_build_", dir=args.output) as temp, ExitStack() as handles:
        stacks = {p: np.memmap(Path(temp)/f"{p}.dat", mode="w+", dtype="float32", shape=(len(ENVIRONMENT), height, width)) for p in args.periods}
        for stack in stacks.values():
            handles.callback(stack._mmap.close)
            stack[:] = NODATA
        seen = np.memmap(Path(temp)/"seen.dat", mode="w+", dtype="uint8", shape=(height, width))
        handles.callback(seen._mmap.close)
        seen[:] = 0
        needed = {"X", "Y"} | {s for block in sources.values() for s in block.values() if s}
        for chunk in pd.read_csv(args.input, usecols=list(needed), chunksize=args.chunk_size):
            total += len(chunk)
            x, y = chunk.X.to_numpy(float), chunk.Y.to_numpy(float)
            if not np.isfinite([x, y]).all():
                raise ValueError("Non-finite coordinates")
            cf = (x-transform.c)/transform.a-0.5
            rf = (y-transform.f)/transform.e-0.5
            cols, rows = np.rint(cf).astype(int), np.rint(rf).astype(int)
            use = (cols >= 0) & (cols < width) & (rows >= 0) & (rows < height)
            if np.any(np.maximum(abs(cf[use]-cols[use]), abs(rf[use]-rows[use])) > 1e-4):
                raise ValueError("Source X/Y are not aligned with cell centers; no silent resampling is performed")
            cols, rows, chunk = cols[use], rows[use], chunk.loc[use]
            if len(set(zip(rows.tolist(), cols.tolist()))) != len(rows) or np.any(seen[rows, cols]):
                raise ValueError("Duplicate grid cells: explicitly resolve duplicates before exporting")
            seen[rows, cols] = 1
            included += len(rows)
            for period in args.periods:
                for j, feature in enumerate(ENVIRONMENT):
                    name = sources[period][feature]
                    values = (pd.to_numeric(chunk[name], errors="coerce").to_numpy(np.float32)
                              if name else np.full(len(chunk), args.park_context, np.float32))
                    invalid = ~np.isfinite(values) | (values <= -9990)
                    if feature == "avg_noise_day":
                        stats[period]["quiet_noise_fills"] += int(invalid.sum())
                        values[invalid] = 40.0
                        invalid[:] = False
                    if feature in ["type_Puisto", *ENVIRONMENT[8:]]:
                        invalid |= ~np.isin(values, [0, 1])
                    values[invalid] = NODATA
                    stacks[period][j, rows, cols] = values
            print(f"Input rows {total:,}; included cells {included:,}", flush=True)
        if included == 0:
            raise ValueError("No source cells overlap the template")
        for period, stack in stacks.items():
            files = period_files[period]
            datasets = {}
            try:
                for kind in files:
                    count = len(ENVIRONMENT) if kind == "environment" else 11 if kind in ["growth", "deviation", "suitability"] else 1
                    floating = kind in ["environment", "growth", "deviation"]
                    datasets[kind] = rasterio.open(args.output/files[kind], "w", **profile(width, height, transform, args.crs, count, "float32" if floating else "uint8", NODATA if floating else 0))
                    labels = ENVIRONMENT if kind == "environment" else list(SPECIES.values()) if count == 11 else [kind]
                    set_descriptions(datasets[kind], labels, "percentage_points" if kind == "deviation" else "annual_growth_percent" if kind == "growth" else "category_or_input")
                    datasets[kind].update_tags(scope=args.scope, model_sha256=sha256(args.model), reference_height_m=args.height,
                                              reliability_definition="0 missing; 1 development min-max; 2 out of range; not confidence")
                for win in windows(width, height):
                    r, c, h, w = int(win.row_off), int(win.col_off), int(win.height), int(win.width)
                    block = np.array(stack[:, r:r+h, c:c+w])
                    env = block.reshape(len(ENVIRONMENT), -1).T
                    codes, robust = domain_codes(env, args.height, domain)
                    valid = codes > 0
                    stats[period]["valid_cells"] += int(valid.sum())
                    stats[period]["out_of_range_cells"] += int((codes == 2).sum())
                    datasets["environment"].write(block, window=win)
                    datasets["reliability"].write(codes.reshape(h,w), 1, window=win)
                    datasets["within_p01_p99"].write(robust.reshape(h,w), 1, window=win)
                    growth = np.full((11, h*w), NODATA, np.float32)
                    deviation = growth.copy()
                    suitability = np.zeros((11, h*w), np.uint8)
                    if valid.any():
                        matrix = matrix_from_environment(booster, env[valid], args.height, 1)
                        for species in SPECIES:
                            for name in SPECIES.values():
                                matrix[:, booster.feature_names.index("Species_"+name)] = 0
                            matrix[:, booster.feature_names.index("Species_"+SPECIES[species])] = 1
                            predicted = percentage(booster.inplace_predict(matrix))
                            growth[species-1, valid] = predicted
                            deviation[species-1, valid] = predicted-reference_predictions[species-1]
                            suitability[species-1, valid] = np.searchsorted(thresholds, predicted, side="right")+1
                    for kind, values in [("growth", growth), ("deviation", deviation), ("suitability", suitability)]:
                        datasets[kind].write(values.reshape(11,h,w), window=win)
            finally:
                for dataset in datasets.values():
                    dataset.close()
            print(f"Mapped {period}: {stats[period]}", flush=True)
        # Close all Windows memory-map handles before TemporaryDirectory cleanup.
        for stack in stacks.values():
            stack.flush()
    change = dict(earlier=args.earlier, later=args.later,
                  growth_change="rasters/growth_change_pp.tif", suitability_change="rasters/suitability_level_change.tif",
                  reliability="rasters/change_reliability.tif")
    early, late = period_files[args.earlier], period_files[args.later]
    with rasterio.open(args.output/early["growth"]) as ge, rasterio.open(args.output/late["growth"]) as gl, \
         rasterio.open(args.output/early["suitability"]) as se, rasterio.open(args.output/late["suitability"]) as sl, \
         rasterio.open(args.output/early["reliability"]) as re, rasterio.open(args.output/late["reliability"]) as rl, \
         rasterio.open(args.output/change["growth_change"], "w", **profile(width,height,transform,args.crs,11,"float32",NODATA)) as gc, \
         rasterio.open(args.output/change["suitability_change"], "w", **profile(width,height,transform,args.crs,11,"int16",-128)) as sc, \
         rasterio.open(args.output/change["reliability"], "w", **profile(width,height,transform,args.crs,1,"uint8",0)) as rc:
        set_descriptions(gc, list(SPECIES.values()), "percentage_points_later_minus_earlier")
        set_descriptions(sc, list(SPECIES.values()), "level_later_minus_earlier")
        for win in windows(width,height):
            a, b = re.read(1,window=win), rl.read(1,window=win)
            valid = (a>0)&(b>0)
            code = np.where(valid,np.maximum(a,b),0).astype(np.uint8)
            gc.write(np.where(valid[None],gl.read(window=win)-ge.read(window=win),NODATA).astype(np.float32),window=win)
            sc.write(np.where(valid[None],sl.read(window=win).astype(np.int16)-se.read(window=win).astype(np.int16),-128).astype(np.int16),window=win)
            rc.write(code,1,window=win)
    scope_note = ("Observed-cell demonstration only; other cells are NoData, not interpolated."
                  if args.scope == "observed_cells_demo" else "Predictions cover supplied grid cells with complete environmental inputs.")
    manifest = dict(schema_version=1, scope=args.scope, scope_note=scope_note, crs=args.crs,
                    grid=dict(width=width,height=height,transform=list(transform)[:6]),
                    model=dict(file="model.json",sha256=sha256(args.model),feature_names=booster.feature_names),
                    environment_features=ENVIRONMENT,groups=GROUPS,species=SPECIES,reference_height_m=args.height,
                    reference_environment=reference,reference_growth_percent=dict(zip(SPECIES,reference_predictions)),
                    thresholds_annual_growth_percent=thresholds.tolist(),domain=domain,periods=period_files,change=change,
                    source_columns=sources, source_input_sha256=sha256(args.input),statistics=stats,
                    missing_policy="Noise missing/sentinel = 40 dB by data-owner convention; other missing inputs = no diagnosis.",
                    interpretation="Exact reference-based grouped Shapley on back-transformed annual growth; no causal attribution.",
                    method="All subsets of changed environmental groups, with a single matched endpoint background.",
                    input_rows=total,included_cells=included,elapsed_seconds=time.perf_counter()-started)
    (args.output/"manifest.json").write_text(json.dumps(manifest,indent=2,allow_nan=False),encoding="utf-8")
    (args.output/"RUN_LOG.md").write_text(
        f"# Clickable spatial package\n\n{scope_note}\n\nFrozen model: `{manifest['model']['sha256']}`\n\n"
        f"Cells supplied: {included:,}. Height: {args.height:g} m. Periods: {', '.join(args.periods)}.\n\n"
        "No retraining or threshold selection. Soil is included jointly in waterfall explanations. "
        "Growth differences are percentage points; suitability change is a separate ordinal output. "
        "The period-free model is identical at both change endpoints. NoData is never zero SHAP.\n",
        encoding="utf-8")
    print("Package complete:",args.output/"manifest.json",flush=True)
    return manifest


if __name__ == "__main__":
    build(parse_args())
