"""Build a QGIS Install-from-ZIP package, including its external-Python worker."""
import argparse
from pathlib import Path
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    root = Path(__file__).resolve().parents[1]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(args.output,"w",compression=zipfile.ZIP_DEFLATED) as archive:
        for path in (root/"qgis/tree_growth_waterfall").rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts:
                archive.write(path,"tree_growth_waterfall/"+path.relative_to(root/"qgis/tree_growth_waterfall").as_posix())
        for name in ["spatial_waterfall_core.py","explain_spatial_cell.py"]:
            archive.write(root/"scripts"/name,"tree_growth_waterfall/worker/"+name)
    print(args.output)


if __name__ == "__main__":
    main()
