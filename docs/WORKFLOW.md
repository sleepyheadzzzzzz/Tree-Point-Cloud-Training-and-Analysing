# Reproduction workflow

Run commands from the repository root. All output paths below are ignored by Git. Scripts stop if a requested output directory already exists, protecting earlier results.

## 1. Environment and audit

```bash
conda env create -f environment.yml
conda activate urban-tree-growth
python tests/validate_project.py
```

Place the exact author-supplied data at `data/tree_carbon_updated4.csv` before refitting.

## 2. Primary model and SHAP analysis

```bash
python scripts/run_soil_augmented_shap_analysis.py \
  --input data/tree_carbon_updated4.csv \
  --pipeline-script scripts/run_relative_growth_pipeline_v2.py \
  --soil-feature-set three \
  --bootstrap-replicates 1000 \
  --output outputs/soil_augmented_shap

python scripts/run_three_soil_model_comparison.py \
  --input data/tree_carbon_updated4.csv \
  --pipeline-script scripts/run_relative_growth_pipeline_v2.py \
  --soil-script scripts/run_soil_augmented_shap_analysis.py \
  --bootstrap-repetitions 1000 \
  --output outputs/model_comparison
```

The first command is the authoritative three-soil SHAP fit. The second compares OLS, RF, XGBoost, and MLP on the same `OID_`-grouped split and feature specification.

## 3. Incremental attribution and permutation tests

```bash
python scripts/run_three_soil_environmental_attribution.py \
  --input data/tree_carbon_updated4.csv \
  --pipeline-script scripts/run_relative_growth_pipeline_v2.py \
  --soil-script scripts/run_soil_augmented_shap_analysis.py \
  --analysis-dir outputs/soil_augmented_shap \
  --permutation-repetitions 300 \
  --bootstrap-repetitions 1500

python scripts/run_biological_baseline_to_full_attribution.py \
  --input data/tree_carbon_updated4.csv \
  --pipeline-script scripts/run_relative_growth_pipeline_v2.py \
  --soil-script scripts/run_soil_augmented_shap_analysis.py \
  --full-analysis outputs/soil_augmented_shap \
  --bootstrap-repetitions 1500 \
  --output outputs/biological_baseline

python scripts/run_biological_baseline_joint_permutation.py \
  --input data/tree_carbon_updated4.csv \
  --pipeline-script scripts/run_relative_growth_pipeline_v2.py \
  --soil-script scripts/run_soil_augmented_shap_analysis.py \
  --full-analysis outputs/soil_augmented_shap \
  --biological-analysis outputs/biological_baseline \
  --permutation-repetitions 300 \
  --output outputs/joint_permutation
```

The height-and-species baseline excludes period. Consequently, its contrast with the full retrospective model measures the combined added information from monitoring period and environment. The environment-only contrast is separately conditioned on species and period.

## 4. Dependence plots and categorized SHAP profiles

The pooled SHAP run saves observation-level SHAP data used by the plotting scripts. Use the actual path written by that run's log for `--saved-observations`.

```bash
python scripts/run_shap_dependence_interaction_plots.py \
  --input data/tree_carbon_updated4.csv \
  --pipeline-script scripts/run_relative_growth_pipeline_v2.py \
  --soil-script scripts/run_soil_augmented_shap_analysis.py \
  --full-analysis outputs/soil_augmented_shap \
  --saved-observations outputs/soil_augmented_shap/tables/soil_augmented_shap_test_observations.csv \
  --interaction-candidates avg_noise_day Density25 Mono_Rate avg_svf avg_radiation avg_LST lightemiss type_Puisto soil_infill soil_bedrock soil_moraine \
  --target-features avg_noise_day Mono_Rate avg_svf avg_LST lightemiss type_Puisto soil_moraine \
  --output outputs/shap_dependence

python scripts/create_radial_shap_diagram.py \
  --statistics outputs/soil_augmented_shap/tables/soil_augmented_shap_group_statistics.csv \
  --deciles outputs/soil_augmented_shap/tables/soil_augmented_shap_dependence_deciles.csv \
  --output-dir outputs/radial_shap

python scripts/create_species_categorized_shap_profiles.py \
  --statistics outputs/soil_augmented_shap/tables/soil_augmented_shap_group_statistics.csv \
  --deciles outputs/soil_augmented_shap/tables/soil_augmented_shap_dependence_deciles.csv \
  --metrics outputs/model_comparison/tables/model_performance_all_groups_three_soil.csv \
  --rotation-degrees-clockwise 10 \
  --contiguous-wedges \
  --hide-category-background \
  --output-dir outputs/species_profiles
```

## 5. Spatially blocked deployment validation

```bash
python scripts/run_spatial_block_validation_three_soil.py \
  --input data/tree_carbon_updated4.csv \
  --pipeline-script scripts/run_relative_growth_pipeline_v2.py \
  --soil-script scripts/run_soil_augmented_shap_analysis.py \
  --block-size-m 500 \
  --split-search-iterations 20000 \
  --bootstrap-repetitions 1000 \
  --moran-permutations 999 \
  --output outputs/spatial_validation

python scripts/evaluate_suitability_diagnostic_schemes.py \
  --input data/tree_carbon_updated4.csv \
  --pipeline-script scripts/run_relative_growth_pipeline_v2.py \
  --soil-script scripts/run_soil_augmented_shap_analysis.py \
  --split-table outputs/spatial_validation/tables/spatial_block_split.csv \
  --locked-test-predictions outputs/spatial_validation/tables/locked_test_predictions.csv \
  --bootstrap-repetitions 1000 \
  --output outputs/suitability_validation
```

Model selection uses training and spatial validation partitions. The locked 15% spatial test is evaluated once after refitting XGBoost on the combined 85% training-plus-validation data.

## 6. Spatial diagnosis and planning outputs

Prepare the 2 m point-grid input using the column names documented by `run_spatial_relative_growth_diagnosis_v2.py --help`. Missing daytime-noise cells are assigned the 40 dB quiet-floor value.

```bash
python scripts/run_spatial_relative_growth_diagnosis_v2.py \
  --input data/spatial_grid.csv \
  --training-data data/tree_carbon_updated4.csv \
  --pipeline-script scripts/run_relative_growth_pipeline_v2.py \
  --model-dir models \
  --soil-script scripts/run_soil_augmented_shap_analysis.py \
  --fixed-suitability-thresholds results/suitability/fixed_selected_seven_level_thresholds.csv \
  --reference-height-m 10 \
  --park-context 1 \
  --resolution 2 \
  --crs EPSG:3879 \
  --write-csv \
  --output outputs/spatial_diagnosis

python scripts/validate_spatial_relative_growth_outputs.py outputs/spatial_diagnosis

python scripts/export_suitable_genus_count.py \
  --input outputs/spatial_diagnosis/suitability_level_21_23.tif \
  --min-area-m2 10 \
  --min-level 5 \
  --max-level 7 \
  --output outputs/genus_suitability

python scripts/create_species_densification_plan.py \
  --input outputs/spatial_diagnosis/suitability_level_21_23.tif \
  --min-area-m2 10 \
  --min-level 5 \
  --max-level 7 \
  --tie-radius-cells 2 \
  --output outputs/densification_plan
```

The planning scripts remove spots smaller than 10 m2. The proposal prioritizes maximum suitability while resolving ties to improve genus diversity. Sorbus can be excluded at the planning/GIS stage where site knowledge identifies it as unsuitable.

## 7. Change detection

```bash
python scripts/create_spatial_change_detection.py \
  --input-csv data/spatial_grid.csv \
  --spatial-output outputs/spatial_diagnosis \
  --chunk-size 250000

python scripts/create_environment_change_figure_with_noise.py \
  --environment-change outputs/spatial_diagnosis/environmental_change.tif \
  --noise-root data/noise_rasters \
  --output outputs/environment_change
```

All changes are later minus earlier. Figure 7 deliberately reverses the display palette for monoculture rate, solar radiation, LST, and noise so decreases are green and increases are red; the numeric raster signs are unchanged.
