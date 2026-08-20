#!/usr/bin/env python3
"""Export the retained suitable-genus count and map metadata to NPZ."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio

from create_species_densification_plan import SPECIES, label_components


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-area-m2", type=float, default=10.0)
    parser.add_argument("--min-level", type=int, default=5)
    parser.add_argument("--max-level", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with rasterio.open(args.input) as source:
        transform = source.transform
        cell_area = abs(transform.a * transform.e - transform.b * transform.d)
        retained_masks = []
        valid_masks = []
        for band in SPECIES:
            values = source.read(band).astype(np.int16)
            valid = source.read_masks(band) > 0
            nodata = source.nodatavals[band - 1]
            if nodata is not None:
                valid &= values != int(nodata)
            valid &= (values >= 1) & (values <= 7)
            suitable = valid & (values >= args.min_level) & (values <= args.max_level)
            labels, components = label_components(suitable)
            accepted = {
                int(component["label"])
                for component in components
                if len(component["flat_indices"]) * cell_area > args.min_area_m2
            }
            retained = (
                np.isin(labels, np.fromiter(accepted, dtype=np.int32))
                if accepted
                else np.zeros_like(valid)
            )
            retained_masks.append(retained)
            valid_masks.append(valid)
        count = np.stack(retained_masks).sum(axis=0).astype(np.uint8)
        valid_any = np.stack(valid_masks).any(axis=0)
        metadata = {
            "crs": source.crs.to_string(),
            "bounds": list(source.bounds),
            "resolution": [abs(transform.a), abs(transform.e)],
            "cell_area_m2": cell_area,
            "species": SPECIES,
            "display_scale": [0, 9],
            "observed_maximum": int(count.max()),
            "suitable_levels": [args.min_level, args.max_level],
            "minimum_patch_area_m2_exclusive": args.min_area_m2,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        suitable_genus_count=count,
        valid_any=valid_any,
        metadata_json=np.asarray(json.dumps(metadata)),
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
