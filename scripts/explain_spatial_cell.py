"""Query a spatial package by coordinate; write JSON, CSV and a waterfall SVG."""
import argparse
import csv
import json
from pathlib import Path
from spatial_waterfall_core import SpatialPackage, waterfall_svg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", required=True, type=Path, help="manifest.json")
    parser.add_argument("--x", required=True, type=float)
    parser.add_argument("--y", required=True, type=float)
    parser.add_argument("--species", type=int, default=2, choices=range(1, 12))
    parser.add_argument("--mode", choices=["local", "change"], default="local")
    parser.add_argument("--period", default="21_23")
    parser.add_argument("--output", required=True, type=Path, help="New output file stem")
    args = parser.parse_args()
    paths = [args.output.with_suffix(suffix) for suffix in [".json", ".svg", ".csv"]]
    if any(path.exists() for path in paths):
        raise FileExistsError("Choose a new output stem")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = SpatialPackage(args.package).explain(args.x, args.y, args.species, args.mode, args.period)
    paths[0].write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    paths[1].write_text(waterfall_svg(result), encoding="utf-8")
    if result["status"] == "ok":
        with paths[2].open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Environmental_group", "Contribution_percentage_points", "Contribution_log_SGR"])
            for group, pp, log in zip(result["groups"], result["contributions_pp"], result["contributions_log_sgr"]):
                writer.writerow([group["label"], pp, log])
    print(json.dumps({"status": result["status"], "json": str(paths[0]), "svg": str(paths[1])}))


if __name__ == "__main__":
    main()
