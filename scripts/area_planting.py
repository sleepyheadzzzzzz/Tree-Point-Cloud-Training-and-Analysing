"""Area-based planting scenarios from fixed-level suitability, without retraining.

This is a screening heuristic, not a globally optimal or field-validated design.
The nine genus bands are explicitly mapped; pooled conifer/broadleaf are excluded.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.features import geometry_mask, shapes
from scipy.ndimage import label

from create_species_densification_plan import (
    SPECIES, box_mean, optimize_diversity_assignment,
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def retain_patches(mask, cell_area, minimum):
    """Four-connected patches; strictly greater than the supplied square metres."""
    labels, _ = label(mask)
    sizes = np.bincount(labels.ravel())
    keep = sizes * cell_area > minimum
    keep[0] = False
    return keep[labels]


def allocate(levels, valid, cell_area, minimum=10, min_level=5, max_loss=2):
    # All configured genera must be known; missing is not equivalent to unsuitable.
    known = valid.all(axis=0)
    eligible = np.stack([retain_patches(v & known & (a >= min_level), cell_area, minimum)
                         for a, v in zip(levels, valid)])
    count = eligible.sum(axis=0).astype("uint8")
    top = np.where(eligible, levels, 0).max(axis=0)
    support = np.stack([box_mean(a, v & known, 2) for a, v in zip(levels, valid)])
    scores = np.where(eligible, levels * 1000.0 + support * 10.0
                      - np.arange(len(levels))[:, None, None] * .001, -1e9)
    highest = scores.argmax(axis=0).astype("int16")
    highest[count == 0] = -1
    # Do not replace an isolated winner with an inferior genus and still label it
    # "highest suitability". Drop the fragment and report its unassigned area.
    for index in range(len(levels)):
        mask = highest == index
        highest[mask & ~retain_patches(mask, cell_area, minimum)] = -1
    allowed = eligible & (levels >= top[None] - max_loss)
    allowed = np.stack([retain_patches(m, cell_area, minimum) for m in allowed])
    diversity = np.full(highest.shape, -1, dtype="int16")
    diagnostics = {"note": "No eligible area"}
    if allowed.any():
        diversity, _, _, _, diagnostics = optimize_diversity_assignment(allowed, levels, support)
        # Dropping tiny assignments preserves the exact eligible/loss constraint.
        # Reassignment could cycle or invalidate a strict maximum-level guarantee.
        for index in range(len(levels)):
            mask = diversity == index
            diversity[mask & ~retain_patches(mask, cell_area, minimum)] = -1
    count[~known] = 255
    return eligible, count, highest, diversity, top, diagnostics


def polygon_features(mask, values, transform, cell_area, code, genus, top=None):
    labels, _ = label(mask)
    sizes = np.bincount(labels.ravel())
    sums = np.bincount(labels.ravel(), weights=values.ravel())
    loss = np.bincount(labels.ravel(), weights=(top-values).ravel()) if top is not None else None
    features = []
    for geometry, identifier in shapes(labels.astype("int32"), mask=mask,
                                       transform=transform, connectivity=4):
        i = int(identifier)
        props = dict(sp_code=int(code), genus=genus, cells=int(sizes[i]),
                     area_m2=float(sizes[i]*cell_area), suit_mean=float(sums[i]/sizes[i]))
        if loss is not None:
            props["loss_mean"] = float(loss[i]/sizes[i])
        features.append(dict(type="Feature", geometry=geometry, properties=props))
    return features


def run(config):
    source_path = Path(config["input"])
    output = Path(config["output"])
    if output.exists():
        raise FileExistsError("Choose a new output directory")
    bands = list(map(int, config.get("bands", range(3, 12))))
    if len(bands) != 9 or len(set(bands)) != 9 or min(bands) < 1:
        raise ValueError("Provide nine distinct band numbers in Acer..Ulmus order")
    minimum = float(config.get("min_area_m2", 10))
    min_level = int(config.get("min_level", 5))
    max_loss = int(config.get("diversity_max_level_loss", 2))
    if not np.isfinite(minimum) or minimum < 0 or not 1 <= min_level <= 7 or not 0 <= max_loss <= 6:
        raise ValueError("Invalid area, minimum level, or diversity level-loss limit")
    with rasterio.open(source_path) as source:
        if source.crs is None or not source.crs.is_projected:
            raise ValueError("A projected CRS with known linear units is required")
        if max(bands) > source.count:
            raise ValueError("A requested genus band does not exist")
        if source.width*source.height > 2_000_000:
            raise ValueError("Clip to a planning site first (maximum 2 million grid cells)")
        raw = source.read(bands, masked=True)
        levels = raw.filled(0).astype("float64")
        valid = ~np.ma.getmaskarray(raw) & np.isfinite(levels)
        invalid = valid & ((levels < 1) | (levels > 7) | (levels != np.rint(levels)))
        if invalid.any():
            raise ValueError("Input must contain integer suitability levels 1-7, not growth or change values")
        transform, crs = source.transform, source.crs
        profile = source.profile.copy()
        descriptions = [source.descriptions[b-1] for b in bands]
        # Descriptions, when present, must agree with the explicitly configured order.
        for name, description in zip(SPECIES.values(), descriptions):
            if description and name.lower() not in description.lower():
                raise ValueError(f"Band description {description!r} does not match {name}; verify band mapping")
        unit_factor = float(crs.linear_units_factor[1])
        cell_area = abs(transform.a*transform.e-transform.b*transform.d)*unit_factor**2
        if not np.isfinite(cell_area) or cell_area <= 0:
            raise ValueError("Invalid cell area")
        domain = np.ones(levels.shape[1:], bool)
        for key, exclude in [("boundary_geometries", False), ("exclusion_geometries", True)]:
            if key in config:
                if not config[key]:
                    raise ValueError(f"{key} is empty; omit it to disable this mask")
                inside = geometry_mask(config[key], domain.shape, transform, invert=True, all_touched=False)
                domain &= ~inside if exclude else inside
        reliability_checked = False
        if config.get("reliability"):
            with rasterio.open(config["reliability"]) as reliability:
                if (reliability.crs != crs or reliability.transform != transform
                        or reliability.shape != domain.shape):
                    raise ValueError("Reliability raster must have exactly the same CRS/grid as suitability")
                rb = [1] if reliability.count == 1 else bands
                if max(rb) > reliability.count:
                    raise ValueError("Reliability must have one band or the same genus-band layout")
                values = reliability.read(rb, masked=True)
                valid &= ((values.filled(0) == 1) & ~np.ma.getmaskarray(values))
                reliability_checked = True
        valid &= domain[None]
    print("Filtering suitable patches and allocating two alternatives...", flush=True)
    eligible, count, highest, diversity, top, diagnostics = allocate(
        levels, valid, cell_area, minimum, min_level, max_loss)
    output.mkdir(parents=True)
    profile.update(driver="GTiff", count=1, dtype="uint8", nodata=255, compress="deflate")
    rasters, vectors = {}, {}

    def save_raster(name, array, description):
        path = output/f"{name}.tif"
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(array.astype("uint8"), 1)
            dst.set_band_description(1, description)
            dst.update_tags(genus_codes=json.dumps(SPECIES), nodata_meaning="255 unknown or outside mask",
                            zero_meaning="No eligible genus or no retained assignment")
        rasters[name] = path.name

    def save_features(name, features):
        for i, feature in enumerate(features, 1):
            feature["properties"]["poly_id"] = i
        path = output/f"{name}.geojson"
        # OGR-compatible projected GeoJSON; CRS is explicit (not RFC 7946).
        path.write_text(json.dumps(dict(type="FeatureCollection", name=name,
            crs=dict(type="name", properties=dict(name=crs.to_string())), features=features)), encoding="utf-8")
        vectors[name] = path.name

    save_raster("suitable_genus_count", count, "Number of suitable genera (0-9); NoData=255")
    candidates = []
    for i, (code, genus) in enumerate(SPECIES.items()):
        candidates += polygon_features(eligible[i], levels[i], transform, cell_area, code, genus)
    save_features("species_suitable_areas", candidates)
    summaries = {}
    for strategy, assignment in [("highest_suitability", highest), ("diversity_oriented", diversity)]:
        codes = np.zeros(count.shape, dtype="uint8")
        assigned_levels = codes.copy()
        features, areas = [], {}
        for i, (code, genus) in enumerate(SPECIES.items()):
            mask = assignment == i
            codes[mask] = code
            assigned_levels[mask] = levels[i, mask].astype("uint8")
            features += polygon_features(mask, levels[i], transform, cell_area, code, genus, top)
            areas[genus] = float(mask.sum()*cell_area)
        assigned = assignment >= 0
        shares = np.array(list(areas.values()), float)
        shares = shares[shares > 0]/max(shares.sum(), 1e-12)
        summaries[strategy] = dict(area_by_genus_m2=areas, assigned_area_m2=float(assigned.sum()*cell_area),
            unassigned_candidate_area_m2=float(((count > 0) & (count != 255) & ~assigned).sum()*cell_area),
            genus_richness=int(len(shares)), area_shannon=float(-np.sum(shares*np.log(shares))),
            mean_level_loss=float(np.mean(top[assigned]-assigned_levels[assigned])) if assigned.any() else None,
            polygons=len(features))
        codes[count == 255] = 255
        assigned_levels[count == 255] = 255
        save_raster(strategy+"_genus", codes, strategy+" genus code; 0 unassigned; 255 unknown")
        save_raster(strategy+"_level", assigned_levels, strategy+" suitability level")
        save_features(strategy, features)
    warnings = ["Screening proposals, not planting approval or causal benefits.",
        "Boundary/exclusions use cell centres; polygon edges follow cells, not an exact vector clip.",
        "All nine genus bands must be known at a cell; missing inputs are not zero suitability.",
        "Area balancing is heuristic; maximum richness/evenness is not guaranteed.",
        "Minimum patch filtering can leave candidate land unassigned."]
    if not reliability_checked:
        warnings.append("Reliability was NOT checked; input suitability alone does not certify training-domain support.")
    if "boundary_geometries" not in config:
        warnings.append("No planning boundary supplied: input raster footprint was used.")
    if "exclusion_geometries" not in config:
        warnings.append("No exclusions supplied: buildings, utilities, water and existing crowns were NOT independently removed.")
    report = dict(schema_version=1, source_sha256=sha256(source_path), configuration=config,
        crs=crs.to_string(), cell_area_m2=cell_area, band_order=dict(zip(SPECIES.values(), bands)),
        suitable_levels=[min_level, 7], minimum_patch_area_exclusive_m2=minimum,
        diversity_max_level_loss=max_loss, count_legend=[0, 9],
        count_observed_max=int(count[count != 255].max()) if np.any(count != 255) else None,
        known_cells=int((count != 255).sum()), unknown_or_excluded_cells=int((count == 255).sum()),
        candidate_area_m2=float(((count > 0) & (count != 255)).sum()*cell_area),
        summaries=summaries, balancing_diagnostics=diagnostics, warnings=warnings,
        rasters=rasters, vectors=vectors)
    (output/"planting_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output/"RUN_LOG.md").write_text("# Area planting run\n\n"+"\n".join("- "+w for w in warnings)
        +"\n\nSee planting_report.json for input hash, settings and measured output statistics.\n", encoding="utf-8")
    print(json.dumps(summaries, indent=2), flush=True)
    return report


def make_demo(output):
    """Public, deterministic synthetic suitability; no study observations."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    yy, xx = np.indices((40, 50))
    arrays = np.stack([np.clip(4+np.sin(xx/9+i)+np.cos(yy/8-i), 1, 7).round()
                       for i in range(11)]).astype("uint8")
    arrays[:, :2] = 0
    arrays[8] = np.where(arrays[8] != 0, 3, 0)  # no suitable Sorbus, in this demo only
    with rasterio.open(output/"suitability.tif", "w", driver="GTiff", height=40, width=50,
                       count=11, dtype="uint8", crs="EPSG:3879", nodata=0,
                       transform=rasterio.transform.from_origin(25496000, 6674000, 2, 2)) as dst:
        dst.write(arrays)
        for code, name in SPECIES.items():
            dst.set_band_description(code, name)
    (output/"README.txt").write_text("SYNTHETIC test data. Not Helsinki measurements or a real planting plan.\n", encoding="utf-8")
    print(output/"suitability.tif")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--config", type=Path)
    group.add_argument("--demo", type=Path)
    args = parser.parse_args()
    if args.demo:
        make_demo(args.demo)
    else:
        run(json.loads(args.config.read_text(encoding="utf-8")))
