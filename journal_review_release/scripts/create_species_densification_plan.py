#!/usr/bin/env python3
"""Create species suitability polygons and an exclusive densification proposal.

The input is a multiband, fixed-threshold suitability raster. Bands 3--11 are
the nine genus scenarios used by the manuscript's spatial diagnosis. Suitable
cells are levels 5--7. Connected patches at or below the requested minimum
area are excluded.

The exclusive proposal selects the highest suitability level at every eligible
cell. Ties are resolved with the higher 5x5-cell neighbourhood mean; residual
ties use the stable source-band order. If the resulting assigned patch is too
small, its cells are offered to their next-best eligible genus before being
dropped.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image, ImageDraw, ImageFont
from rasterio.features import shapes


SPECIES = {
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

SPECIES_COLORS = {
    3: (31, 119, 180),
    4: (44, 160, 44),
    5: (148, 103, 189),
    6: (0, 109, 44),
    7: (227, 119, 194),
    8: (140, 86, 75),
    9: (255, 127, 14),
    10: (188, 189, 34),
    11: (23, 190, 207),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-area-m2", type=float, default=10.0)
    parser.add_argument("--min-level", type=int, default=5)
    parser.add_argument("--max-level", type=int, default=7)
    parser.add_argument("--tie-radius-cells", type=int, default=2)
    return parser.parse_args()


def label_components(mask: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    """Label four-neighbour connected cells and retain their flat indices."""
    height, width = mask.shape
    labels = np.zeros(mask.shape, dtype=np.int32)
    components: list[dict] = []
    current = 0
    for row, col in np.argwhere(mask):
        if labels[row, col] != 0:
            continue
        current += 1
        queue: deque[tuple[int, int]] = deque([(int(row), int(col))])
        labels[row, col] = current
        flat_indices: list[int] = []
        while queue:
            rr, cc = queue.popleft()
            flat_indices.append(rr * width + cc)
            for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                if (
                    0 <= nr < height
                    and 0 <= nc < width
                    and mask[nr, nc]
                    and labels[nr, nc] == 0
                ):
                    labels[nr, nc] = current
                    queue.append((nr, nc))
        components.append(
            {
                "label": current,
                "flat_indices": np.asarray(flat_indices, dtype=np.int64),
            }
        )
    return labels, components


def box_mean(array: np.ndarray, valid: np.ndarray, radius: int) -> np.ndarray:
    """Return a square-window mean without scipy."""
    values = np.where(valid, array, 0.0).astype(np.float64)
    counts = valid.astype(np.float64)
    pad = max(0, int(radius))
    values = np.pad(values, pad, mode="constant")
    counts = np.pad(counts, pad, mode="constant")

    def integral_sum(source: np.ndarray) -> np.ndarray:
        integral = np.pad(source, ((1, 0), (1, 0)), mode="constant")
        integral = integral.cumsum(axis=0).cumsum(axis=1)
        window = 2 * pad + 1
        return (
            integral[window:, window:]
            - integral[:-window, window:]
            - integral[window:, :-window]
            + integral[:-window, :-window]
        )

    value_sum = integral_sum(values)
    count_sum = integral_sum(counts)
    return np.divide(
        value_sum,
        count_sum,
        out=np.zeros_like(value_sum),
        where=count_sum > 0,
    )


def max_min_diversity_targets(
    capacities: np.ndarray,
    lower_bounds: np.ndarray,
    total_cells: int,
) -> np.ndarray:
    """Return equal-area targets constrained by feasible lower/upper bounds."""
    capacities = capacities.astype(np.float64)
    lower_bounds = lower_bounds.astype(np.float64)
    low = 0.0
    high = max(float(np.max(capacities)), float(total_cells))
    for _ in range(100):
        midpoint = 0.5 * (low + high)
        targets = np.clip(midpoint, lower_bounds, capacities)
        if float(targets.sum()) < total_cells:
            low = midpoint
        else:
            high = midpoint
    targets = np.clip(0.5 * (low + high), lower_bounds, capacities)
    difference = float(total_cells - targets.sum())
    adjustable = (targets > lower_bounds + 1.0e-9) & (targets < capacities - 1.0e-9)
    if np.any(adjustable):
        targets[adjustable] += difference / int(adjustable.sum())
    return targets


def optimize_diversity_assignment(
    eligible_masks: np.ndarray,
    levels_array: np.ndarray,
    local_support: np.ndarray,
    iterations: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Balance assigned area across feasible genera, then prefer suitability."""
    species_count, height, width = eligible_masks.shape
    eligible_count = eligible_masks.sum(axis=0).astype(np.int16)
    active_cells = eligible_count > 0
    total_cells = int(active_cells.sum())
    capacities = eligible_masks.reshape(species_count, -1).sum(axis=1).astype(np.float64)
    unique_candidate = eligible_count == 1
    lower_bounds = np.asarray(
        [np.sum(eligible_masks[index] & unique_candidate) for index in range(species_count)],
        dtype=np.float64,
    )
    targets = max_min_diversity_targets(capacities, lower_bounds, total_cells)
    integer_targets = np.floor(targets).astype(np.int64)
    remainder = total_cells - int(integer_targets.sum())
    if remainder > 0:
        fractions = targets - integer_targets
        for index in np.argsort(-fractions, kind="stable")[:remainder]:
            integer_targets[index] += 1

    base_scores = np.full(eligible_masks.shape, -1.0e9, dtype=np.float64)
    for index in range(species_count):
        base_scores[index, eligible_masks[index]] = (
            levels_array[index, eligible_masks[index]] * 100.0
            + local_support[index, eligible_masks[index]] * 2.0
            - index * 0.0001
        )

    biases = np.zeros(species_count, dtype=np.float64)
    eligible_species = capacities > 0
    best_loss = float("inf")
    best_winner = np.full((height, width), -1, dtype=np.int16)
    best_biases = biases.copy()
    best_counts = np.zeros(species_count, dtype=np.int64)
    for iteration in range(iterations):
        adjusted = base_scores + biases[:, None, None]
        winner = np.argmax(adjusted, axis=0).astype(np.int16)
        winner[~active_cells] = -1
        counts = np.bincount(winner[active_cells], minlength=species_count)
        loss = float(np.abs(counts - integer_targets).sum() / max(total_cells, 1))
        if loss < best_loss:
            best_loss = loss
            best_winner = winner.copy()
            best_biases = biases.copy()
            best_counts = counts.copy()
        if np.max(np.abs(counts[eligible_species] - integer_targets[eligible_species])) <= 2:
            break
        # Coordinate update: holding every other genus fixed, calculate the
        # additive score bias required for this genus to win its target number
        # of eligible cells. Damping avoids cycling between overlapping masks.
        for index in np.where(eligible_species)[0]:
            other_indices = [other for other in range(species_count) if other != index]
            other_max = np.max(
                base_scores[other_indices] + biases[other_indices, None, None],
                axis=0,
            )
            required = (
                other_max[eligible_masks[index]]
                - base_scores[index, eligible_masks[index]]
            )
            target = int(integer_targets[index])
            if target <= 0:
                proposed_bias = -1.0e6
            elif target >= len(required):
                proposed_bias = float(np.max(required) + 0.01)
            else:
                partitioned = np.partition(required, (target - 1, target))
                proposed_bias = float(
                    0.5 * (partitioned[target - 1] + partitioned[target])
                )
            biases[index] = 0.55 * biases[index] + 0.45 * proposed_bias
        biases[eligible_species] -= np.mean(biases[eligible_species])

    final_scores = base_scores + best_biases[:, None, None]
    candidate_order = np.argsort(-final_scores, axis=0)
    diagnostics = {
        "eligible_capacity_cells": capacities.tolist(),
        "single_option_lower_bound_cells": lower_bounds.tolist(),
        "target_cells": targets.tolist(),
        "integer_target_cells": integer_targets.tolist(),
        "pre_patch_filter_cells": best_counts.tolist(),
        "allocation_l1_error_fraction": best_loss,
        "dual_biases": best_biases.tolist(),
    }
    return best_winner, candidate_order, eligible_count, targets, diagnostics


def enforce_minimum_assigned_patches(
    assignment: np.ndarray,
    candidate_order: np.ndarray,
    eligible_count: np.ndarray,
    cell_area: float,
    min_area_m2: float,
) -> np.ndarray:
    """Offer cells in small assigned fragments to their next-best candidate."""
    height, width = assignment.shape
    active = eligible_count > 0
    ranks = np.zeros((height, width), dtype=np.int16)
    rows, cols = np.where(active)
    # The supplied assignment is expected to be the first candidate, but this
    # lookup keeps the function safe if an externally prepared assignment is used.
    for row, col in zip(rows, cols):
        matches = np.where(candidate_order[:, row, col] == assignment[row, col])[0]
        if len(matches):
            ranks[row, col] = int(matches[0])
    result = assignment.copy()
    species_count = candidate_order.shape[0]
    for _ in range(species_count + 1):
        result.fill(-1)
        rows, cols = np.where(active & (ranks < eligible_count))
        result[rows, cols] = candidate_order[ranks[rows, cols], rows, cols]
        too_small = np.zeros((height, width), dtype=bool)
        for species_index in range(species_count):
            _, components = label_components(result == species_index)
            for component in components:
                if len(component["flat_indices"]) * cell_area <= min_area_m2:
                    too_small.ravel()[component["flat_indices"]] = True
        if not np.any(too_small):
            break
        ranks[too_small] += 1
        active[ranks >= eligible_count] = False
    result.fill(-1)
    rows, cols = np.where(active & (ranks < eligible_count))
    result[rows, cols] = candidate_order[ranks[rows, cols], rows, cols]
    return result


def proposal_from_assignment(
    assignment: np.ndarray,
    levels_array: np.ndarray,
    top_level_ties: np.ndarray,
    transform,
    cell_area: float,
    min_area_m2: float,
) -> tuple[list[dict], list[dict], np.ndarray, np.ndarray]:
    """Create polygon features, summaries, and rasters for one proposal."""
    height, width = assignment.shape
    species_raster = np.zeros((height, width), dtype=np.uint8)
    suitability_raster = np.zeros((height, width), dtype=np.uint8)
    features: list[dict] = []
    summary: list[dict] = []
    polygon_id = 0
    for species_index, (band, species) in enumerate(SPECIES.items()):
        mask = assignment == species_index
        species_raster[mask] = band
        suitability_raster[mask] = levels_array[species_index, mask].astype(np.uint8)
        labels, components = label_components(mask)
        accepted = {
            int(component["label"])
            for component in components
            if len(component["flat_indices"]) * cell_area > min_area_m2
        }
        geometries = polygonize_labels(labels, accepted, transform)
        retained_cells = 0
        all_values: list[int] = []
        for component in components:
            label = int(component["label"])
            if label not in accepted:
                continue
            polygon_id += 1
            flat = component["flat_indices"]
            values = levels_array[species_index].ravel()[flat]
            ties = top_level_ties.ravel()[flat]
            retained_cells += len(flat)
            all_values.extend(int(value) for value in values)
            properties = {
                "poly_id": polygon_id,
                "band": band,
                "sp_code": band,
                "species": species,
                "cells": int(len(flat)),
                "area_m2": float(len(flat) * cell_area),
                "suit_min": int(values.min()),
                "suit_max": int(values.max()),
                "suit_mean": float(values.mean()),
                "suit_dom": int(dominant_level(values)),
                "tie_pct": float(100.0 * np.mean(ties > 1)),
            }
            features.append(
                {
                    "type": "Feature",
                    "properties": properties,
                    "geometry": geometries[label],
                }
            )
        summary.append(
            {
                "band": band,
                "species": species,
                "proposal_patches": len(accepted),
                "proposal_cells": int(retained_cells),
                "proposal_area_m2": round(float(retained_cells * cell_area), 3),
                "mean_suitability": round(float(np.mean(all_values)), 3) if all_values else "",
            }
        )
    return features, summary, species_raster, suitability_raster


def dominant_level(values: np.ndarray) -> int:
    counts = Counter(int(value) for value in values)
    return max(counts, key=lambda value: (counts[value], value))


def polygonize_labels(
    labels: np.ndarray,
    accepted: set[int],
    transform,
) -> dict[int, dict]:
    mask = np.isin(labels, np.fromiter(accepted, dtype=np.int32)) if accepted else np.zeros_like(labels, dtype=bool)
    result: dict[int, dict] = {}
    if not np.any(mask):
        return result
    for geometry, value in shapes(
        labels.astype(np.int32),
        mask=mask,
        transform=transform,
        connectivity=4,
    ):
        result[int(value)] = geometry
    return result


def write_geojson(path: Path, name: str, crs_epsg: int | None, features: list[dict]) -> None:
    collection: dict = {
        "type": "FeatureCollection",
        "name": name,
        "features": features,
    }
    if crs_epsg is not None:
        collection["crs"] = {
            "type": "name",
            "properties": {"name": f"urn:ogc:def:crs:EPSG::{crs_epsg}"},
        }
    path.write_text(json.dumps(collection, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_font(size: int, bold: bool = False):
    names = ["arialbd.ttf", "DejaVuSans-Bold.ttf"] if bold else ["arial.ttf", "DejaVuSans.ttf"]
    for name in names:
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def add_map_furniture(
    draw: ImageDraw.ImageDraw,
    origin: tuple[int, int],
    map_size: tuple[int, int],
    source_width: int,
    resolution_x: float,
) -> None:
    x0, y0 = origin
    map_width, _ = map_size
    draw.line((x0 + 24, y0 + 64, x0 + 24, y0 + 18), fill=(25, 25, 25), width=4)
    draw.polygon(
        [(x0 + 24, y0 + 8), (x0 + 15, y0 + 26), (x0 + 33, y0 + 26)],
        fill=(25, 25, 25),
    )
    draw.text((x0 + 12, y0 + 67), "N", fill=(25, 25, 25), font=load_font(18, True))
    scale_m = 100.0
    scale_pixels = scale_m / resolution_x * (map_width / source_width)
    sx = x0 + 24
    sy = y0 + map_size[1] - 32
    draw.line((sx, sy, sx + scale_pixels, sy), fill=(25, 25, 25), width=5)
    draw.line((sx, sy - 7, sx, sy + 7), fill=(25, 25, 25), width=3)
    draw.line((sx + scale_pixels, sy - 7, sx + scale_pixels, sy + 7), fill=(25, 25, 25), width=3)
    draw.text((sx, sy + 9), "100 m", fill=(25, 25, 25), font=load_font(16))


def create_preview(
    path: Path,
    candidate_count: np.ndarray,
    highest_species: np.ndarray,
    diversity_species: np.ndarray,
    valid_any: np.ndarray,
    resolution_x: float,
) -> None:
    height, width = candidate_count.shape
    map_height = 690
    map_width = int(round(map_height * width / height))
    gap = 48
    margin = 42
    title_height = 100
    legend_height = 220
    canvas = Image.new(
        "RGB",
        (margin * 2 + map_width * 3 + gap * 2, title_height + map_height + legend_height),
        (255, 255, 255),
    )
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (margin, 25),
        "Species suitability and alternative densification proposals",
        fill=(25, 25, 25),
        font=load_font(27, True),
    )
    origins = [
        (margin, title_height),
        (margin + map_width + gap, title_height),
        (margin + (map_width + gap) * 2, title_height),
    ]
    subtitles = [
        "A  Number of suitable genera (0–9)",
        "B  Highest suitability",
        "C  Maximum feasible diversity",
    ]
    for origin, subtitle in zip(origins, subtitles):
        draw.text((origin[0], title_height - 34), subtitle, fill=(25, 25, 25), font=load_font(18, True))

    count_palette = np.asarray(
        [
            (242, 242, 242),
            (237, 248, 233),
            (199, 233, 192),
            (161, 217, 155),
            (116, 196, 118),
            (65, 171, 93),
            (35, 139, 69),
            (0, 109, 44),
            (0, 83, 34),
            (0, 55, 23),
        ],
        dtype=np.uint8,
    )
    count_rgb = count_palette[np.clip(candidate_count, 0, 9)]
    count_rgb[~valid_any] = (220, 220, 220)
    highest_rgb = np.full((height, width, 3), (242, 242, 242), dtype=np.uint8)
    diversity_rgb = np.full((height, width, 3), (242, 242, 242), dtype=np.uint8)
    highest_rgb[~valid_any] = (220, 220, 220)
    diversity_rgb[~valid_any] = (220, 220, 220)
    for code, color in SPECIES_COLORS.items():
        highest_rgb[highest_species == code] = color
        diversity_rgb[diversity_species == code] = color

    resample = getattr(Image, "Resampling", Image).NEAREST
    for rgb, origin in zip((count_rgb, highest_rgb, diversity_rgb), origins):
        map_image = Image.fromarray(rgb, mode="RGB").resize((map_width, map_height), resample=resample)
        canvas.paste(map_image, origin)
        draw.rectangle(
            (origin[0], origin[1], origin[0] + map_width, origin[1] + map_height),
            outline=(90, 90, 90),
            width=1,
        )
        add_map_furniture(draw, origin, (map_width, map_height), width, resolution_x)

    legend_y = title_height + map_height + 40
    draw.text((origins[0][0], legend_y), "Suitable genera per cell", fill=(25, 25, 25), font=load_font(17, True))
    for index in range(0, 10):
        col = index % 5
        row = index // 5
        xx = origins[0][0] + col * 78
        yy = legend_y + 35 + row * 34
        draw.rectangle((xx, yy, xx + 22, yy + 22), fill=tuple(count_palette[index]))
        draw.text((xx + 29, yy + 1), str(index), fill=(25, 25, 25), font=load_font(15))

    draw.text((origins[1][0], legend_y), "Proposed genus", fill=(25, 25, 25), font=load_font(17, True))
    legend_left = origins[1][0]
    legend_width = origins[2][0] + map_width - legend_left
    for index, (code, species) in enumerate(SPECIES.items()):
        col = index % 5
        row = index // 5
        xx = legend_left + col * (legend_width // 5)
        yy = legend_y + 35 + row * 34
        draw.rectangle((xx, yy, xx + 22, yy + 22), fill=SPECIES_COLORS[code])
        draw.text((xx + 29, yy + 1), species, fill=(25, 25, 25), font=load_font(14))
    canvas.save(path, dpi=(300, 300))


def main() -> None:
    args = parse_args()
    if args.min_area_m2 < 0:
        raise ValueError("--min-area-m2 must be non-negative")
    if not (1 <= args.min_level <= args.max_level <= 7):
        raise ValueError("Suitability levels must satisfy 1 <= min <= max <= 7")

    output = args.output.resolve()
    vector_dir = output / "vector"
    table_dir = output / "tables"
    raster_dir = output / "rasters"
    plot_dir = output / "plots"
    for directory in (output, vector_dir, table_dir, raster_dir, plot_dir):
        directory.mkdir(parents=True, exist_ok=True)

    with rasterio.open(args.input) as source:
        if source.count < max(SPECIES):
            raise ValueError(f"Input has {source.count} bands; band 11 is required")
        if source.crs is None or not source.crs.is_projected:
            raise ValueError("Input must use a projected CRS so patch area is in square metres")
        transform = source.transform
        crs = source.crs
        epsg = crs.to_epsg()
        crs_wkt = crs.to_wkt()
        width, height = source.width, source.height
        bounds = source.bounds
        profile = source.profile.copy()
        levels = []
        valid_masks = []
        for band in SPECIES:
            array = source.read(band).astype(np.int16)
            valid = source.read_masks(band) > 0
            if source.nodatavals[band - 1] is not None:
                valid &= array != int(source.nodatavals[band - 1])
            valid &= (array >= 1) & (array <= 7)
            levels.append(array)
            valid_masks.append(valid)

    levels_array = np.stack(levels)
    valid_array = np.stack(valid_masks)
    cell_area = abs(transform.a * transform.e - transform.b * transform.d)
    if not math.isfinite(cell_area) or cell_area <= 0:
        raise ValueError("Invalid raster transform or cell area")

    candidate_features: list[dict] = []
    candidate_summary: list[dict] = []
    eligible_masks = np.zeros_like(valid_array, dtype=bool)
    polygon_id = 0

    for species_index, (band, species) in enumerate(SPECIES.items()):
        array = levels_array[species_index]
        suitable = (
            valid_array[species_index]
            & (array >= args.min_level)
            & (array <= args.max_level)
        )
        labels, components = label_components(suitable)
        accepted: set[int] = set()
        stats: dict[int, dict] = {}
        for component in components:
            flat = component["flat_indices"]
            area_m2 = len(flat) * cell_area
            if area_m2 <= args.min_area_m2:
                continue
            label = int(component["label"])
            accepted.add(label)
            component_values = array.ravel()[flat]
            stats[label] = {
                "cells": int(len(flat)),
                "area_m2": float(area_m2),
                "suit_min": int(component_values.min()),
                "suit_max": int(component_values.max()),
                "suit_mean": float(component_values.mean()),
                "suit_dom": int(dominant_level(component_values)),
            }
        eligible = np.isin(labels, np.fromiter(accepted, dtype=np.int32)) if accepted else np.zeros_like(suitable)
        eligible_masks[species_index] = eligible
        geometries = polygonize_labels(labels, accepted, transform)
        for label in sorted(accepted):
            polygon_id += 1
            properties = {
                "poly_id": polygon_id,
                "band": band,
                "sp_code": band,
                "species": species,
                **stats[label],
            }
            candidate_features.append(
                {
                    "type": "Feature",
                    "properties": properties,
                    "geometry": geometries[label],
                }
            )
        accepted_cells = int(eligible.sum())
        candidate_summary.append(
            {
                "band": band,
                "species": species,
                "raw_suitable_cells": int(suitable.sum()),
                "raw_suitable_area_m2": round(float(suitable.sum() * cell_area), 3),
                "retained_patches": len(accepted),
                "retained_cells": accepted_cells,
                "retained_area_m2": round(float(accepted_cells * cell_area), 3),
                "excluded_small_cells": int(suitable.sum() - accepted_cells),
                "excluded_small_area_m2": round(float((suitable.sum() - accepted_cells) * cell_area), 3),
            }
        )

    local_support = np.stack(
        [
            box_mean(levels_array[index], valid_array[index], args.tie_radius_cells)
            for index in range(len(SPECIES))
        ]
    )
    species_codes = np.asarray(list(SPECIES), dtype=np.int16)
    scores = np.full(levels_array.shape, -1.0e9, dtype=np.float64)
    for index in range(len(SPECIES)):
        scores[index, eligible_masks[index]] = (
            levels_array[index, eligible_masks[index]] * 1000.0
            + local_support[index, eligible_masks[index]] * 10.0
            - index * 0.001
        )

    # Proposal 1: retain the highest-suitability diagnosis.
    order = np.argsort(-scores, axis=0)
    eligible_count = eligible_masks.sum(axis=0).astype(np.int16)
    highest_index = np.argmax(scores, axis=0).astype(np.int16)
    highest_index[eligible_count == 0] = -1
    highest_index = enforce_minimum_assigned_patches(
        highest_index,
        order,
        eligible_count,
        cell_area,
        args.min_area_m2,
    )

    candidate_count = eligible_masks.sum(axis=0).astype(np.uint8)
    top_level = np.max(np.where(eligible_masks, levels_array, 0), axis=0)
    top_level_ties = np.sum(eligible_masks & (levels_array == top_level), axis=0)
    (
        highest_features,
        highest_summary,
        highest_species,
        highest_suitability,
    ) = proposal_from_assignment(
        highest_index,
        levels_array,
        top_level_ties,
        transform,
        cell_area,
        args.min_area_m2,
    )

    # Proposal 2: maximize feasible genus richness and allocation evenness,
    # with suitability and local support as secondary allocation criteria.
    (
        diversity_initial,
        diversity_order,
        diversity_eligible_count,
        diversity_targets,
        diversity_diagnostics,
    ) = optimize_diversity_assignment(eligible_masks, levels_array, local_support)
    diversity_index = enforce_minimum_assigned_patches(
        diversity_initial,
        diversity_order,
        diversity_eligible_count,
        cell_area,
        args.min_area_m2,
    )
    (
        diversity_features,
        diversity_summary,
        diversity_species,
        diversity_suitability,
    ) = proposal_from_assignment(
        diversity_index,
        levels_array,
        top_level_ties,
        transform,
        cell_area,
        args.min_area_m2,
    )

    candidate_geojson = vector_dir / "species_suitable_areas.geojson"
    highest_geojson = vector_dir / "densification_proposal_highest_suitability.geojson"
    diversity_geojson = vector_dir / "densification_proposal_maximum_diversity.geojson"
    write_geojson(candidate_geojson, "species_suitable_areas", epsg, candidate_features)
    write_geojson(highest_geojson, "densification_proposal_highest_suitability", epsg, highest_features)
    write_geojson(diversity_geojson, "densification_proposal_maximum_diversity", epsg, diversity_features)
    (vector_dir / "crs.wkt").write_text(crs_wkt, encoding="utf-8")

    write_csv(
        table_dir / "species_suitable_area_summary.csv",
        candidate_summary,
        [
            "band",
            "species",
            "raw_suitable_cells",
            "raw_suitable_area_m2",
            "retained_patches",
            "retained_cells",
            "retained_area_m2",
            "excluded_small_cells",
            "excluded_small_area_m2",
        ],
    )
    write_csv(
        table_dir / "densification_proposal_highest_suitability_summary.csv",
        highest_summary,
        [
            "band",
            "species",
            "proposal_patches",
            "proposal_cells",
            "proposal_area_m2",
            "mean_suitability",
        ],
    )
    write_csv(
        table_dir / "densification_proposal_maximum_diversity_summary.csv",
        diversity_summary,
        [
            "band",
            "species",
            "proposal_patches",
            "proposal_cells",
            "proposal_area_m2",
            "mean_suitability",
        ],
    )

    out_profile = profile.copy()
    out_profile.update(
        driver="GTiff",
        count=1,
        dtype="uint8",
        nodata=0,
        compress="deflate",
        predictor=2,
    )
    with rasterio.open(raster_dir / "densification_highest_suitability_species.tif", "w", **out_profile) as destination:
        destination.write(highest_species, 1)
        destination.set_band_description(1, "Highest_suitability_species_code")
        destination.update_tags(
            species_codes=json.dumps(SPECIES),
            selection_rule="Highest level; 5x5 neighbourhood mean tie-break; small-patch fallback",
        )
    with rasterio.open(raster_dir / "densification_highest_suitability_level.tif", "w", **out_profile) as destination:
        destination.write(highest_suitability, 1)
        destination.set_band_description(1, "Highest_suitability_level")
    with rasterio.open(raster_dir / "densification_maximum_diversity_species.tif", "w", **out_profile) as destination:
        destination.write(diversity_species, 1)
        destination.set_band_description(1, "Maximum_diversity_species_code")
        destination.update_tags(
            species_codes=json.dumps(SPECIES),
            selection_rule="Maximum feasible genus richness; constrained area-balancing heuristic; suitability secondary; small-patch fallback",
        )
    with rasterio.open(raster_dir / "densification_maximum_diversity_level.tif", "w", **out_profile) as destination:
        destination.write(diversity_suitability, 1)
        destination.set_band_description(1, "Maximum_diversity_suitability_level")

    valid_any = np.any(valid_array, axis=0)
    create_preview(
        plot_dir / "densification_plan_preview.png",
        candidate_count,
        highest_species,
        diversity_species,
        valid_any,
        abs(transform.a),
    )

    total_unique_candidate_cells = int(np.any(eligible_masks, axis=0).sum())
    total_highest_cells = int((highest_species > 0).sum())
    total_diversity_cells = int((diversity_species > 0).sum())
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input": str(args.input.resolve()),
        "input_band_mapping": SPECIES,
        "input_crs": crs.to_string(),
        "input_epsg": epsg,
        "input_bounds": [bounds.left, bounds.bottom, bounds.right, bounds.top],
        "raster_width": width,
        "raster_height": height,
        "cell_size": [abs(transform.a), abs(transform.e)],
        "cell_area_m2": cell_area,
        "suitable_levels": [args.min_level, args.max_level],
        "minimum_patch_rule": f"Retain four-neighbour connected patches with area > {args.min_area_m2} m2",
        "minimum_patch_area_m2_exclusive": args.min_area_m2,
        "tie_break": {
            "primary": "highest suitability level",
            "secondary": f"highest {(2 * args.tie_radius_cells + 1)}x{(2 * args.tie_radius_cells + 1)}-cell neighbourhood mean suitability",
            "tertiary": "lower source band number",
            "small_patch_fallback": "offer cells to next-best eligible genus; otherwise omit",
        },
        "candidate_polygon_count": len(candidate_features),
        "candidate_count_display_scale": [0, 9],
        "candidate_count_observed_maximum": int(candidate_count.max()),
        "sorbus_retained_suitable_area_m2": 0.0,
        "highest_suitability_proposal_polygon_count": len(highest_features),
        "maximum_diversity_proposal_polygon_count": len(diversity_features),
        "unique_candidate_area_m2": total_unique_candidate_cells * cell_area,
        "highest_suitability_proposal_area_m2": total_highest_cells * cell_area,
        "maximum_diversity_proposal_area_m2": total_diversity_cells * cell_area,
        "maximum_diversity_definition": "Maximize feasible genus richness first; then use constrained area-balancing reference targets, with suitability level and local support as secondary criteria. Exact equal allocation is not generally feasible because suitable masks overlap and some cells have only one eligible genus.",
        "maximum_diversity_balancing_reference_cells": {
            species: float(diversity_targets[index])
            for index, species in enumerate(SPECIES.values())
        },
        "maximum_diversity_optimizer": diversity_diagnostics,
        "scientific_scope": "Model-derived suitability diagnosis; planting feasibility still requires field and infrastructure checks.",
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (output / "RUN_LOG.md").write_text(
        "\n".join(
            [
                "# Species densification plan run log",
                "",
                f"- Input: `{args.input.resolve()}`",
                f"- CRS: `{crs.to_string()}`",
                f"- Raster: {width} × {height}; cell area = {cell_area:.6f} m²",
                f"- Species bands: 3–11 ({', '.join(SPECIES.values())})",
                f"- Suitable levels: {args.min_level}–{args.max_level}",
                f"- Patch rule: four-neighbour connected area > {args.min_area_m2} m²",
                f"- Species-suitable polygons: {len(candidate_features)}",
                f"- Unique retained candidate area: {total_unique_candidate_cells * cell_area:.2f} m²",
                f"- Count-map legend: 0–9 genera; observed maximum = {int(candidate_count.max())} because Sorbus has no retained suitable patch.",
                f"- Highest-suitability proposal polygons: {len(highest_features)}",
                f"- Highest-suitability proposal area: {total_highest_cells * cell_area:.2f} m²",
                f"- Maximum-diversity proposal polygons: {len(diversity_features)}",
                f"- Maximum-diversity proposal area: {total_diversity_cells * cell_area:.2f} m²",
                "- Highest-suitability rule: highest level; 5×5 neighbourhood mean tie-break; stable band-order residual tie-break; next-best fallback for small assigned patches.",
                "- Maximum-diversity rule: represent every genus with retained level-5–7 area, then apply a constrained area-balancing heuristic; suitability and local support are secondary; next-best fallback for small assigned patches.",
                "- Original raster was read only and was not modified.",
                "- Interpretation: planning diagnosis, not a direct planting-feasibility survey.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
