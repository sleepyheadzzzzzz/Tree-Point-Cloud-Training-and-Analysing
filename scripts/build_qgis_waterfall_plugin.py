"""Build a QGIS Install-from-ZIP package, including its external-Python worker."""
import argparse
import configparser
from pathlib import Path
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--official", action="store_true", help="Require public author/contact metadata before building for QGIS submission")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    root = Path(__file__).resolve().parents[1]
    metadata=configparser.ConfigParser()
    metadata.read(root/"qgis/tree_growth_waterfall/metadata.txt",encoding="utf-8")
    if args.official:
        general=metadata["general"]
        if not general.get("author","").strip() or "@" not in general.get("email",""):
            raise ValueError("Public author and contact email are required before official submission; see docs/QGIS_PUBLICATION.md")
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(args.output,"w",compression=zipfile.ZIP_DEFLATED) as archive:
        for path in (root/"qgis/tree_growth_waterfall").rglob("*"):
            if path.is_file() and not any(part.startswith(".") or part=="__pycache__" for part in path.relative_to(root/"qgis/tree_growth_waterfall").parts):
                archive.write(path,"tree_growth_waterfall/"+path.relative_to(root/"qgis/tree_growth_waterfall").as_posix())
        for name in ["spatial_waterfall_core.py","explain_spatial_cell.py","tree_growth_workbench.py",
                     "build_clickable_spatial_package.py","validate_clickable_spatial_package.py",
                     "restore_archived_clickable_package.py","run_relative_growth_pipeline_v2.py",
                     "run_spatial_block_validation_three_soil.py", "area_planting.py", "create_species_densification_plan.py"]:
            archive.write(root/"scripts"/name,"tree_growth_waterfall/worker/"+name)
        for relative in ["models/xgb_spatial_deployment_no_period_three_soil.json",
                         "results/spatial_validation/deployment_preprocessing.json",
                         "results/spatial_validation/development_training_domain.csv",
                         "results/suitability/fixed_selected_seven_level_thresholds.csv",
                         "docs/QGIS_WORKBENCH.md","docs/CLICKABLE_SPATIAL_WATERFALL.md", "docs/QGIS_AREA_PLANTING.md", "docs/QGIS_PUBLICATION.md",
                         "requirements.txt","LICENSE"]:
            archive.write(root/relative,"tree_growth_waterfall/"+relative)
    with zipfile.ZipFile(args.output) as archive:
        forbidden={".pyc",".pyo",".dll",".exe",".so",".joblib",".pkl",".tif"}
        assert not any(Path(name).suffix.lower() in forbidden for name in archive.namelist())
    if args.output.stat().st_size > 20_000_000:
        raise ValueError("QGIS package exceeds 20 MB")
    print(args.output)


if __name__ == "__main__":
    main()
