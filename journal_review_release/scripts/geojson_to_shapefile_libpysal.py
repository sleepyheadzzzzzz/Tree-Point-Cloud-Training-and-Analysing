#!/usr/bin/env python3
"""Convert the densification GeoJSON outputs to ESRI Shapefiles.

This converter uses libpysal's pure-Python SHP/SHX and DBF writers. It is kept
separate from the raster-processing script because the local rasterio and
libpysal installations live in different Python runtimes.
"""

from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path

import libpysal
from libpysal.cg import Polygon


PROPOSAL_SCHEMA = [
    ("POLY_ID", "poly_id", ("N", 10, 0)),
    ("BAND", "band", ("N", 3, 0)),
    ("SP_CODE", "sp_code", ("N", 3, 0)),
    ("SPECIES", "species", ("C", 20, 0)),
    ("CELLS", "cells", ("N", 10, 0)),
    ("AREA_M2", "area_m2", ("N", 18, 2)),
    ("SUIT_MIN", "suit_min", ("N", 3, 0)),
    ("SUIT_MAX", "suit_max", ("N", 3, 0)),
    ("SUIT_MEAN", "suit_mean", ("N", 10, 3)),
    ("SUIT_DOM", "suit_dom", ("N", 3, 0)),
    ("TIE_PCT", "tie_pct", ("N", 8, 2)),
]

SCHEMAS = {
    "species_suitable_areas": [
        ("POLY_ID", "poly_id", ("N", 10, 0)),
        ("BAND", "band", ("N", 3, 0)),
        ("SP_CODE", "sp_code", ("N", 3, 0)),
        ("SPECIES", "species", ("C", 20, 0)),
        ("CELLS", "cells", ("N", 10, 0)),
        ("AREA_M2", "area_m2", ("N", 18, 2)),
        ("SUIT_MIN", "suit_min", ("N", 3, 0)),
        ("SUIT_MAX", "suit_max", ("N", 3, 0)),
        ("SUIT_MEAN", "suit_mean", ("N", 10, 3)),
        ("SUIT_DOM", "suit_dom", ("N", 3, 0)),
    ],
    "densification_proposal_highest_suitability": PROPOSAL_SCHEMA,
    "densification_proposal_maximum_diversity": PROPOSAL_SCHEMA,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--crs-wkt", type=Path, required=True)
    parser.add_argument("--zip-dir", type=Path)
    return parser.parse_args()


def as_polygon(geometry: dict) -> Polygon:
    geometry_type = geometry["type"]
    coordinates = geometry["coordinates"]
    outer_parts: list[list[tuple[float, float]]] = []
    holes: list[list[tuple[float, float]]] = []
    polygons = [coordinates] if geometry_type == "Polygon" else coordinates
    if geometry_type not in {"Polygon", "MultiPolygon"}:
        raise ValueError(f"Unsupported geometry type: {geometry_type}")
    for polygon in polygons:
        if not polygon:
            continue
        outer_parts.append([tuple(point[:2]) for point in polygon[0]])
        holes.extend([tuple(point[:2]) for point in ring] for ring in polygon[1:])
    if not outer_parts:
        raise ValueError("Polygon has no outer ring")
    vertices = outer_parts if len(outer_parts) > 1 else outer_parts[0]
    return Polygon(vertices, holes=holes or None)


def write_shapefile(source: Path, destination: Path, schema: list[tuple], crs_wkt: str) -> int:
    collection = json.loads(source.read_text(encoding="utf-8"))
    features = collection.get("features", [])
    if not features:
        raise ValueError(f"No features in {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".shp", ".shx", ".dbf", ".prj", ".cpg"):
        candidate = destination.with_suffix(suffix)
        if candidate.exists():
            candidate.unlink()

    shp = libpysal.io.open(str(destination.with_suffix(".shp")), "w")
    dbf = libpysal.io.open(str(destination.with_suffix(".dbf")), "w")
    dbf.header = [field[0] for field in schema]
    dbf.field_spec = [field[2] for field in schema]
    try:
        for feature in features:
            shp.write(as_polygon(feature["geometry"]))
            properties = feature.get("properties", {})
            dbf.write([properties.get(field[1]) for field in schema])
    finally:
        shp.close()
        dbf.close()

    destination.with_suffix(".prj").write_text(crs_wkt, encoding="utf-8")
    destination.with_suffix(".cpg").write_text("UTF-8\n", encoding="ascii")
    return len(features)


def zip_shapefile(base: Path, zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for suffix in (".shp", ".shx", ".dbf", ".prj", ".cpg"):
            path = base.with_suffix(suffix)
            archive.write(path, arcname=path.name)


def main() -> None:
    args = parse_args()
    crs_wkt = args.crs_wkt.read_text(encoding="utf-8")
    counts = {}
    for stem, schema in SCHEMAS.items():
        source = args.input_dir / f"{stem}.geojson"
        destination = args.output_dir / stem
        counts[stem] = write_shapefile(source, destination, schema, crs_wkt)
        if args.zip_dir is not None:
            zip_shapefile(destination, args.zip_dir / f"{stem}_shapefile.zip")
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
