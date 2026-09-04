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
        for name in ["spatial_waterfall_core.py","explain_spatial_cell.py","tree_growth_workbench.py",
                     "build_clickable_spatial_package.py","validate_clickable_spatial_package.py",
                     "restore_archived_clickable_package.py","run_relative_growth_pipeline_v2.py",
                     "run_spatial_block_validation_three_soil.py"]:
            archive.write(root/"scripts"/name,"tree_growth_waterfall/worker/"+name)
        for relative in ["models/xgb_spatial_deployment_no_period_three_soil.json",
                         "models/preprocessing_spatial_deployment.joblib",
                         "results/spatial_validation/development_training_domain.csv",
                         "results/suitability/fixed_selected_seven_level_thresholds.csv",
                         "docs/QGIS_WORKBENCH.md","docs/CLICKABLE_SPATIAL_WATERFALL.md",
                         "requirements.txt","LICENSE"]:
            archive.write(root/relative,"tree_growth_waterfall/"+relative)
    print(args.output)


if __name__ == "__main__":
    main()
