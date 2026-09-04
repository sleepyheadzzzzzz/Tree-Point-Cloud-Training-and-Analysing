"""Reference-matched, exact grouped Shapley explanations in annual-growth pp.

This is NOT a relabelled raw/log-SGR TreeSHAP explanation. The cooperative game
evaluates the back-transformed frozen model for every subset of changed groups.
Species and height stay fixed. Soil indicators move jointly as one group.
"""
from __future__ import annotations

import hashlib
import html
import json
import math
from pathlib import Path

import numpy as np
import rasterio
import xgboost as xgb

ENVIRONMENT = [
    "avg_noise_day", "Density25", "Mono_Rate", "avg_svf", "avg_radiation",
    "avg_LST", "lightemiss", "type_Puisto", "soil_infill", "soil_bedrock", "soil_moraine",
]
LABELS = [
    "Daytime noise", "Surrounding-tree density", "Monoculture rate", "Sky-view factor",
    "Solar radiation", "Land-surface temperature", "Nighttime illumination", "Park context",
]
GROUPS = [{"label": label, "features": [feature]} for label, feature in zip(LABELS, ENVIRONMENT[:8])]
GROUPS += [{"label": "Soil context (joint)", "features": ENVIRONMENT[8:]}]
SPECIES = {1: "General_Conifer", 2: "General_Broadleaf", 3: "Acer", 4: "Alnus",
           5: "Betula", 6: "Pinus", 7: "Prunus", 8: "Quercus", 9: "Sorbus",
           10: "Tilia", 11: "Ulmus"}
NODATA = -9999.0


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def percentage(log_sgr):
    with np.errstate(over="raise", invalid="raise"):
        return 100.0 * np.expm1(np.exp(np.asarray(log_sgr, dtype=np.float64)))


def exact_grouped_contrast(predict_log, start, end, groups):
    """Enumerate the reference-replacement game; return exact pp and log terms.

Inactive groups have zero terms. The baseline is one specified reference row,
not a population background. Interactions are allocated by Shapley weights.
"""
    start, end = np.asarray(start, np.float32), np.asarray(end, np.float32)
    if start.shape != end.shape or start.ndim != 1 or not np.isfinite([start, end]).all():
        raise ValueError("Endpoints must be equally sized finite one-dimensional arrays")
    flat = [i for group in groups for i in group]
    if len(flat) != len(set(flat)):
        raise ValueError("Groups overlap")
    changed = set(np.flatnonzero(start != end).tolist())
    if not changed.issubset(flat):
        raise ValueError("A changed non-environmental feature was not held fixed")
    active = [j for j, indices in enumerate(groups) if np.any(start[indices] != end[indices])]
    m = len(active)
    if m > 15:
        raise ValueError("Exact enumeration limited to 15 changed groups")
    masks = np.arange(1 << m, dtype=np.int64)
    matrix = np.tile(start, (len(masks), 1))
    for bit, j in enumerate(active):
        rows = (masks & (1 << bit)) != 0
        matrix[np.ix_(rows, groups[j])] = end[groups[j]]
    log_values = np.asarray(predict_log(matrix), np.float64)
    values = percentage(log_values)
    terms, log_terms = np.zeros(len(groups)), np.zeros(len(groups))
    sizes = np.array([int(mask).bit_count() for mask in masks])
    for bit, j in enumerate(active):
        absent = masks[(masks & (1 << bit)) == 0]
        weights = np.array([1.0 / (m * math.comb(m - 1, int(sizes[s]))) for s in absent])
        terms[j] = np.dot(weights, values[absent | (1 << bit)] - values[absent])
        log_terms[j] = np.dot(weights, log_values[absent | (1 << bit)] - log_values[absent])
    residual = float(terms.sum() - (values[-1] - values[0]))
    if abs(residual) > 1e-8 * max(1.0, abs(values[-1]), abs(values[0])):
        raise AssertionError("Shapley additivity failed")
    return dict(start_growth_percent=float(values[0]), end_growth_percent=float(values[-1]),
                delta_pp=float(values[-1] - values[0]), contributions_pp=terms.tolist(),
                contributions_log_sgr=log_terms.tolist(), additivity_error_pp=residual,
                coalitions_evaluated=len(masks))


class ModelPredictor:
    """Unified frozen predictor. Load joblib only from trusted local training runs."""
    def __init__(self, path, format="xgboost_json"):
        self.format = format
        if format == "xgboost_json":
            self.model = xgb.Booster(params={"nthread": 1})
            self.model.load_model(path)
            self.feature_names = self.model.feature_names
        elif format == "trusted_joblib":
            import joblib
            self.bundle = joblib.load(path)
            self.model = self.bundle["model"]
            self.feature_names = self.bundle["feature_columns"]
        else:
            raise ValueError("Unsupported frozen model format")

    def inplace_predict(self, matrix):
        if self.format == "xgboost_json":
            return self.model.inplace_predict(matrix)
        values = np.asarray(matrix, np.float32)
        if self.bundle["use_scaled"]:
            values = self.bundle["scaler"].transform(values)
        return np.asarray(self.model.predict(values), float)


def matrix_from_environment(booster, environment, height, species_code):
    if species_code not in SPECIES or not np.isfinite(height) or height <= 0:
        raise ValueError("Choose species 1-11 and a positive height")
    environment = np.atleast_2d(np.asarray(environment, np.float32))
    names = booster.feature_names
    environment_names = [f for f in ENVIRONMENT if f in (names or [])]
    if environment_names not in [ENVIRONMENT, ENVIRONMENT[:8]]:
        raise ValueError("Expected either the archived eight-input or current eleven-input specification")
    expected = {"Log_Height", *environment_names, *("Species_" + s for s in SPECIES.values())}
    if set(names or []) != expected:
        raise ValueError("Expected a frozen period-free pooled one-hot species model")
    if environment.shape[1] != len(environment_names):
        raise ValueError("Environmental input count does not match the model")
    matrix = np.zeros((len(environment), len(names)), np.float32)
    matrix[:, names.index("Log_Height")] = np.log(height)
    matrix[:, names.index("Species_" + SPECIES[species_code])] = 1
    for j, name in enumerate(environment_names):
        matrix[:, names.index(name)] = environment[:, j]
    return matrix


def domain_codes(environment, height, domain, environment_names=None):
    """0 missing; 1 inside development min-max; 2 outside. Not confidence."""
    env = np.atleast_2d(environment)
    environment_names = environment_names or ENVIRONMENT
    valid = np.isfinite(env).all(axis=1) & (env > -9990).all(axis=1)
    inside, robust = valid.copy(), valid.copy()
    for feature in ["Log_Height", *environment_names]:
        bounds = domain[feature]
        values = np.full(len(env), np.log(height)) if feature == "Log_Height" else env[:, environment_names.index(feature)]
        inside &= (values >= bounds["Minimum"]) & (values <= bounds["Maximum"])
        robust &= (values >= bounds["P01"]) & (values <= bounds["P99"])
    return np.where(~valid, 0, np.where(inside, 1, 2)).astype("uint8"), robust.astype("uint8")


def safe_package_path(root, relative):
    root = Path(root).resolve()
    result = (root / relative).resolve()
    if not result.is_relative_to(root):
        raise ValueError("Package path escapes its root")
    return result


class SpatialPackage:
    def __init__(self, manifest_path):
        self.path = Path(manifest_path).resolve()
        self.root = self.path.parent
        self.meta = json.loads(self.path.read_text(encoding="utf-8"))
        if self.meta["schema_version"] != 1:
            raise ValueError("Unsupported spatial package schema")
        model_path = self.file(self.meta["model"]["file"])
        if sha256(model_path) != self.meta["model"]["sha256"]:
            raise ValueError("Frozen model checksum mismatch")
        self.booster = ModelPredictor(model_path, self.meta["model"].get("format", "xgboost_json"))
        if self.booster.feature_names != self.meta["model"]["feature_names"]:
            raise ValueError("Model feature-order mismatch")
        self.thresholds = np.asarray(self.meta["thresholds_annual_growth_percent"])
        self.environment_names = self.meta["environment_features"]
        self.groups = self.meta["groups"]

    def file(self, relative):
        return safe_package_path(self.root, relative)

    def check_grid(self, raster):
        grid = self.meta["grid"]
        if ((raster.width,raster.height) != (grid["width"],grid["height"])
                or not np.allclose(list(raster.transform)[:6],grid["transform"],rtol=0,atol=1e-8)
                or raster.crs != rasterio.crs.CRS.from_user_input(self.meta["crs"])):
            raise ValueError("Raster geometry/CRS differs from the matched package")

    def sample_period(self, period, x, y):
        records = self.meta["periods"][period]
        with rasterio.open(self.file(records["environment"])) as src:
            self.check_grid(src)
            row, col = src.index(x, y)
            if not (0 <= row < src.height and 0 <= col < src.width):
                return None
            if list(src.descriptions) != self.environment_names:
                raise ValueError("Environment raster band order differs from manifest contract")
            vals = src.read(window=((row, row + 1), (col, col + 1)))[:, 0, 0]
        if not np.isfinite(vals).all() or np.any(vals <= -9990):
            return None
        return vals, row, col

    def explain(self, x, y, species_code=2, mode="local", period="21_23"):
        if mode not in {"local", "change"}:
            raise ValueError("mode must be local or change")
        if str(species_code) not in self.meta["species"]:
            raise ValueError("Species code must be 1-11")
        early, late = self.meta["change"]["earlier"], self.meta["change"]["later"]
        target_period = late if mode == "change" else period
        target = self.sample_period(target_period, x, y)
        source = self.sample_period(early, x, y) if mode == "change" else None
        if target is None or (mode == "change" and source is None):
            return dict(status="missing", reliability_code=0,
                        message="No diagnosis: no valid environmental inputs for this cell and required period(s).")
        env_end, row, col = target
        env_start = source[0] if source else np.array([self.meta["reference_environment"][f] for f in self.environment_names], np.float32)
        height = self.meta["reference_height_m"]
        endpoints = matrix_from_environment(self.booster, np.stack([env_start, env_end]), height, species_code)
        groups = [[self.booster.feature_names.index(f) for f in group["features"]] for group in self.groups]
        result = exact_grouped_contrast(self.booster.inplace_predict, endpoints[0], endpoints[1], groups)
        reliability, robust = domain_codes(np.stack([env_start, env_end]), height, self.meta["domain"], self.environment_names)
        warnings = []
        imputed = []
        for current in ([early, target_period] if mode == "change" else [target_period]):
            path = self.meta["periods"][current].get("imputed_inputs")
            if path:
                with rasterio.open(self.file(path)) as src:
                    self.check_grid(src)
                    flags = src.read(window=((row,row+1),(col,col+1)))[:,0,0]
                imputed += [f"{current}: {f}" for f, flag in zip(self.environment_names,flags) if flag == 1]
        if imputed:
            warnings.append("Missing inputs were median-imputed by the archived exporter: " + ", ".join(imputed))
        if np.any(reliability == 2):
            warnings.append("OUT OF RANGE: at least one endpoint is outside development min-max limits.")
        elif not robust.all():
            warnings.append("Inside min-max, but at least one endpoint is outside P01-P99 limits.")
        if self.meta.get("scope") != "wall_to_wall_input_grid":
            warnings.append(self.meta["scope_note"])
        if self.meta.get("model_vintage_note"):
            warnings.append(self.meta["model_vintage_note"])
        levels = [int(np.searchsorted(self.thresholds, value, side="right") + 1)
                  for value in [result["start_growth_percent"], result["end_growth_percent"]]]
        # Refuse an explanation that does not match the mapped predictions.
        to_check = [(target_period, result["end_growth_percent"])]
        if mode == "change":
            to_check.append((early, result["start_growth_percent"]))
        errors = []
        for current, expected in to_check:
            with rasterio.open(self.file(self.meta["periods"][current]["growth"])) as src:
                self.check_grid(src)
                expected_label = self.meta.get("species_band_labels",{}).get(str(species_code),SPECIES[species_code])
                if src.descriptions[species_code - 1] != expected_label:
                    raise ValueError("Growth raster species-band mismatch")
                mapped = float(src.read(species_code, window=((row, row + 1), (col, col + 1)))[0, 0])
            errors.append(abs(mapped - expected))
            if not math.isclose(mapped, expected, rel_tol=2e-6, abs_tol=2e-5):
                raise ValueError("Raster/model prediction mismatch; regenerate a matched package")
        delta_raster = self.meta["change"]["growth_change"] if mode == "change" else self.meta["periods"][period]["deviation"]
        with rasterio.open(self.file(delta_raster)) as src:
            self.check_grid(src)
            mapped_delta = float(src.read(species_code, window=((row, row + 1), (col, col + 1)))[0, 0])
        if not math.isclose(mapped_delta, result["delta_pp"], rel_tol=2e-5, abs_tol=3e-5):
            raise ValueError("Mapped deviation/change does not match waterfall")
        errors.append(abs(mapped_delta - result["delta_pp"]))
        result.update(status="ok", mode=mode, period=target_period, earlier_period=early,
                      species=SPECIES[species_code], species_code=species_code, reference_height_m=height,
                      x=float(x), y=float(y), row=row, column=col,
                      start_level=levels[0], end_level=levels[1], level_change=levels[1]-levels[0],
                      reliability_code=int(0 if imputed else 2 if np.any(reliability == 2) else 1),
                      within_p01_p99=bool(robust.all()), warnings=warnings,
                      max_raster_parity_error_pp=max(errors),
                      start_environment=dict(zip(self.environment_names, map(float, env_start))),
                      end_environment=dict(zip(self.environment_names, map(float, env_end))), groups=self.groups,
                      imputed_inputs=imputed,
                      interpretation="Reference-dependent model associations, not causal effects or measured offsets.")
        return result


def waterfall_svg(result):
    if result["status"] != "ok":
        return '<svg xmlns="http://www.w3.org/2000/svg" width="900" height="130"><rect width="900" height="130" fill="white"/><text x="20" y="65" font-family="Arial" font-size="15">' + html.escape(result["message"]) + '</text></svg>'
    order = np.argsort(-np.abs(result["contributions_pp"]))
    start, end = result["start_growth_percent"], result["end_growth_percent"]
    levels = [start]
    for j in order:
        levels.append(levels[-1] + result["contributions_pp"][j])
    lo, hi = min([start, end, *levels]), max([start, end, *levels])
    padding = max((hi-lo)*0.18, 0.2)
    lo, hi = lo-padding, hi+padding
    scale = lambda v: 340 + 420 * (v-lo)/(hi-lo)
    import textwrap
    warning_lines = [line for warning in result["warnings"] for line in textwrap.wrap(warning,112)]
    height = 245 + (len(order)+2)*39 + 21*len(warning_lines)
    items = [f'<svg xmlns="http://www.w3.org/2000/svg" width="900" height="{height}" viewBox="0 0 900 {height}">', f'<rect width="900" height="{height}" fill="white"/>']
    def text(x, y, s, size=13, color="#25323b", weight="normal", anchor="start"):
        items.append(f'<text x="{x:.2f}" y="{y:.2f}" font-family="Arial,sans-serif" font-size="{size}" fill="{color}" font-weight="{weight}" text-anchor="{anchor}">{html.escape(str(s))}</text>')
    change = result["mode"] == "change"
    title = "Environmental explanation of growth change" if change else "Environmental explanation of local deviation"
    text(18, 28, title, 21, weight="bold")
    text(18, 53, f'{result["species"]} | {result["reference_height_m"]:g} m | cell {result["row"]}, {result["column"]}', 14)
    text(18, 78, f'{start:.3f}% to {end:.3f}% per year | difference {result["delta_pp"]:+.3f} percentage points', 15, weight="bold")
    text(18, 101, f'Suitability {result["start_level"]} to {result["end_level"]} | domain code {result["reliability_code"]} (not confidence)', 13)
    names = ["Earlier growth" if change else "Reference growth"]
    names += [result["groups"][j]["label"] for j in order]
    names += ["Later growth" if change else "Local growth"]
    for i, name in enumerate(names):
        y = 139 + i*39
        text(18, y+4, name, 13, weight="bold" if i in (0, len(names)-1) else "normal")
        if i in (0, len(names)-1):
            value = start if i == 0 else end
            items.append(f'<circle cx="{scale(value):.2f}" cy="{y}" r="6" fill="#25323b"/>')
            text(790, y+4, f'{value:.3f}%', 13, weight="bold")
        else:
            j = order[i-1]
            a, b = levels[i-1], levels[i]
            term = result["contributions_pp"][j]
            color = "#208751" if term > 1e-10 else "#c73b49" if term < -1e-10 else "#a5adb3"
            items.append(f'<rect x="{min(scale(a),scale(b)):.2f}" y="{y-9}" width="{max(abs(scale(b)-scale(a)),1):.2f}" height="18" fill="{color}"/>')
            items.append(f'<line x1="{scale(a):.2f}" y1="{y-30}" x2="{scale(a):.2f}" y2="{y-10}" stroke="#b4bdc3" stroke-dasharray="3 3"/>')
            text(790, y+4, f'{term:+.4f} pp', 12, color=color)
            vals = []
            for feature in result["groups"][j]["features"]:
                vals.append(f'{result["start_environment"][feature]:.3g} to {result["end_environment"][feature]:.3g}')
            text(28, y+19, "; ".join(vals), 10, color="#66747d")
    axis_y = 139 + len(names)*39
    items.append(f'<line x1="340" x2="760" y1="{axis_y}" y2="{axis_y}" stroke="#66747d"/>')
    for value in np.linspace(lo, hi, 5):
        text(scale(value), axis_y+19, f'{value:.2f}', 11, anchor="middle")
    text(550, axis_y+39, "Annual carbon growth (% per year)", 13, anchor="middle")
    text(18, axis_y+64, "Green: positive contribution | Red: negative | Bars sum to the mapped difference.", 12)
    text(18, axis_y+84, "Reference-dependent model associations, not causal effects or observed growth offsets.", 12)
    for i, warning in enumerate(warning_lines):
        text(18, axis_y+107+i*21, warning, 11, color="#9b5a12")
    items.append("</svg>")
    return "\n".join(items)
