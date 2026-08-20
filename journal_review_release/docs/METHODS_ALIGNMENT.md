# Manuscript-to-code alignment

This index maps the analytical claims in the submitted manuscript to executable scripts and archived reviewer outputs. It separates retrospective interpretation from deployment-oriented spatial diagnosis.

| Manuscript component | Primary script(s) | Archived evidence |
|---|---|---|
| Positive-growth response construction and grouped development/test split | `run_relative_growth_pipeline_v2.py` | model-comparison tables |
| Common OLS, RF, XGBoost, and MLP comparison | `run_three_soil_model_comparison.py` | `results/model_comparison/` |
| Pooled period-controlled XGBoost and environmental SHAP | `run_soil_augmented_shap_analysis.py` | `results/shap/` |
| Height-and-species biological baseline versus full model | `run_biological_baseline_to_full_attribution.py` | `results/attribution/biological_baseline_vs_full_summary.csv` |
| Environment-only and combined-block joint permutation | `run_biological_baseline_joint_permutation.py` | `results/attribution/joint_permutation_summary.csv` |
| Nonlinear SHAP dependence and interaction-coloured scatter plots | `run_shap_dependence_interaction_plots.py`; `run_nonlinear_shap_dependency_analysis.py` | `results/shap/` |
| Categorized pooled and species-specific SHAP profiles | `create_radial_shap_diagram.py`; `create_species_categorized_shap_profiles.py` | SHAP statistics and category tables |
| 500 m blocked 70/15/15 model selection and locked test | `run_spatial_block_validation_three_soil.py` | `results/spatial_validation/` |
| Fixed-threshold seven-level and three-zone suitability agreement | `evaluate_suitability_diagnostic_schemes.py` | `results/suitability/` |
| Period-free 2 m spatial diagnosis | `run_spatial_relative_growth_diagnosis_v2.py`; `validate_spatial_relative_growth_outputs.py` | included deployment model and preprocessing |
| Environmental and modelled-change mapping | `create_spatial_change_detection.py`; `create_environment_change_figure_with_noise.py` | `results/change_detection/` |
| Genus suitability polygons and diversified densification proposal | `export_suitable_genus_count.py`; `create_species_densification_plan.py`; `geojson_to_shapefile_libpysal.py` | generated GIS outputs |

## Guardrails encoded in the workflow

- `OID_` is the grouping identifier, preventing repeated periods from the same tree from crossing a split.
- The target is log annualized specific growth. Log-SGR RMSE/MAE are not labelled as kg C.
- Species are one-hot encoded. Codes 1–11 are observations; “Overall” is an analysis population, not a species code.
- Monitoring period is retained for retrospective inference but omitted from the spatial deployment model.
- The environmental block contains density, monoculture rate, SVF, radiation, LST, daytime noise, nighttime illumination, park context, and infill/bedrock/moraine indicators. Clay and silt/sand are excluded.
- Suitability uses fixed development-derived thresholds, never map-specific quantiles.
- SHAP and environmental contrasts are model associations and scenario diagnostics, not causal effects or measured environmental offsets.
