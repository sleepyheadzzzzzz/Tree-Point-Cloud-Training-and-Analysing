# Archived reviewer results

These compact outputs are the numerical audit trail for the manuscript. They allow reviewers to verify reported point estimates without redistributing the full tree-level dataset.

- `model_comparison/`: OLS, RF, XGBoost, and MLP results on the common grouped split, plus clustered bootstrap comparisons.
- `attribution/`: height-and-species baseline versus the full model, and joint-permutation analyses.
- `shap/`: pooled/group SHAP statistics, dependence summaries, and species/category aggregation tables.
- `spatial_validation/`: 500 m spatial-block candidate validation and locked-test diagnostics.
- `suitability/`: fixed seven-level thresholds and tree-level agreement for seven-level detail and the three-zone diagnostic.
- `change_detection/`: temporal daytime-noise summary and Figure 7 metadata/caption.

Large observation-level tables, rasters, shapefiles, and rendered figures are reproducible outputs and are not committed. Generate them under an ignored `outputs/` directory.
