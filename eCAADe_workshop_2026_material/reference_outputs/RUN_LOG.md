# Verified reference run

Date: 2026-08-28  
Workflow: `urban_tree_ml_workshop_2026.py`  
Random seed: 2026  
Model scope: one pooled XGBoost model with one-hot species, site type, and period  
Target: log annualized specific carbon growth (`log-SGR`)

## Data and split

- Source tree rows: 16,665.
- Prepared positive-growth period rows: 37,118.
- Training: 25,965 rows from 11,652 trees.
- Validation: 5,575 rows from 2,495 trees.
- Locked test: 5,578 rows from 2,496 trees.
- Leakage audit: every `Original_Tree_RowID` occurs in exactly one partition.

## Model checkpoint

- Exposed settings: learning rate 0.03, maximum depth 4, row subsample 0.80, column subsample 0.85, L1 0.30, L2 10.0, and 60-round early stopping.
- Validation-selected boosting rounds: 543.
- Validation R²: 0.4268.
- Locked-test R²: 0.4309.
- Locked-test RMSE / MAE: 1.0616 / 0.7792 log-SGR.
- Locked-test MAE: 4.4104 annual percentage points.
- Locked-test MAE: 4.9637 kg C tree⁻¹ yr⁻¹.

## VIF checkpoint

Training-only continuous-predictor VIF values ranged from 1.122 to 2.484. Sky-view factor was highest (2.484), followed by solar radiation (2.090). No numeric predictor reached the conventional VIF = 5 review threshold. Categorical one-hot predictors were excluded from this linear diagnostic.

## SHAP checkpoint

The four environment-only figures were generated from a reproducible sample of 1,200 locked-test observations. The leading environmental predictors by mean absolute SHAP were surrounding-tree density, solar radiation, and monoculture rate. Dependence-plot colour variables are restricted to environmental interactions. The waterfall absorbs height, species, period, and site-type SHAP contributions into its contextual starting value, so only environmental bars are displayed. These are fitted model associations, not causal effects.

## ONNX checkpoint

Python XGBoost and ONNX Runtime were compared on the 12-row example test set.

- Maximum absolute log-SGR difference: 4.7684×10⁻⁶.
- Maximum absolute percentage-point difference: 1.1543×10⁻⁵.
- Maximum absolute kg C difference: 3.2188×10⁻⁵.

Versions used for this verification: Python 3.12.13, NumPy 2.4.6, pandas 3.0.5, scikit-learn 1.7.2, XGBoost 3.0.5, SHAP 0.49.1, ONNX 1.22.0, ONNX Runtime 1.29.0, and onnxmltools 1.16.0.
