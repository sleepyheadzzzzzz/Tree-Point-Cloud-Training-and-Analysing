# Included model artifacts

The release includes three XGBoost models and their fitted preprocessing objects:

| Model | Scientific role | Period control |
|---|---|---|
| `xgb_retrospective_period_controlled_three_soil.json` | Retrospective model comparison and SHAP inference | Included |
| `xgb_spatial_deployment_no_period_three_soil.json` | Locked spatial validation and scenario diagnosis | Excluded |
| `xgb_height_species_biological_baseline.json` | Nested biological baseline for incremental attribution | Excluded by definition |

The retrospective model uses height, one-hot species, monitoring-period controls, and the measured environmental block. The deployment model omits categorical period so that maps from different environmental periods are comparable under a common model setting. Soil context is represented by infill, bedrock, and moraine indicators; clay and silt/sand are not used.

The `.json` XGBoost files are portable model structures. The `.joblib` files contain fitted preprocessing objects and should only be loaded from this trusted repository; Python pickle-compatible formats can execute code when loaded from an untrusted source.
