# eCAADe Workshop 2026 Material

## Explainable machine learning for urban-tree carbon growth

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sleepyheadzzzzzz/Tree-Point-Cloud-Training-and-Analysing/blob/main/eCAADe_workshop_2026_material/urban_tree_ml_workshop_2026.ipynb)

This exercise turns multi-temporal urban-tree observations into one **pooled** XGBoost model and then explains, maps, and exports that model. Species is retained as a categorical predictor, but the workflow does not fit separate species, conifer, or broadleaf models.

The lesson has five stages:

1. Environment setting and data preparation.
2. Leakage-safe data processing and training-only VIF diagnosis.
3. Model training with visible parameter settings, validation, refitting, and locked testing.
4. Environment-only SHAP explanation: beeswarm, dependence, waterfall, and spatial SHAP.
5. ONNX model export with a reproducible example test set.

## Files

| File or folder | Purpose |
|---|---|
| `urban_tree_ml_workshop_2026.ipynb` | Guided Google Colab exercise. |
| `urban_tree_ml_workshop_2026.py` | The same workflow as a reusable Python module and command-line script. |
| `data/tree_carbon_ml_teaching_sample.csv` | Cleaned tree-level teaching sample used by the exercise. |
| `requirements-colab.txt` | Python dependencies. |
| `reference_outputs/` | Verified model, VIF/settings tables, ONNX examples, metrics, and four environment-only SHAP figures. |

## Scientific target

The model does not use raw kg C growth as its training target. For each tree-period observation:

```text
g = [ln(C_end) - ln(C_start)] / years
y = ln(g)
```

`g` is annualized specific carbon growth in year⁻¹ and `y` is the model target, log-SGR. Only observations with positive, finite `g`, positive starting height, and positive starting carbon are modelled.

The model output is back-transformed explicitly:

```text
annual growth (%) = 100 × [exp(exp(y)) - 1]
annual carbon gain (kg C tree⁻¹ yr⁻¹) = C_start × [exp(exp(y)) - 1]
```

This distinction matters: RMSE or MAE on `y` must not be described as kg C error.

## Data processing and leakage control

The source table has one row per tree and measurements from 2015, 2017, 2021, and 2023. The script first assigns each `Original_Tree_RowID` to a 70% training, 15% validation, or 15% test partition, stratified by the retained species label. It then converts each tree into up to three period records. Consequently, all repeated periods from the same tree remain in one partition.

The predictors are:

- Biological: log starting height and one-hot species.
- Temporal: one-hot monitoring period.
- Site context: street or park.
- Environment: daytime noise, surrounding-tree density, monoculture rate, sky-view factor, solar radiation, land-surface temperature, and nighttime illumination.

The sample does not contain the soil layers used in the full research dataset, so soil predictors are intentionally absent from this teaching exercise.

### VIF diagnostic

Variance inflation factors are calculated from continuous training predictors only. Categorical one-hot variables are excluded to avoid exact dummy-variable dependence. VIF diagnoses linear collinearity; it does not measure nonlinear redundancy and is not used as an automatic XGBoost feature-removal rule. In the verified run, all numeric VIF values were below 2.5, with sky-view factor highest at 2.484.

## Run in Google Colab

1. Click the Colab badge above.
2. Choose **Runtime → Run all**.
3. The setup cell installs only the workshop dependencies and clones this repository if needed.
4. The final cell creates `/content/eCAADe_2026_outputs` and a downloadable ZIP archive.

Every executable cell has a preceding explanation of its purpose, expected output, and interpretation. The notebook calls the functions step by step, so participants can inspect VIF, model settings, the validation decision, locked test, each SHAP view, and ONNX parity separately.

## Run as a Python script

From this folder:

```bash
python -m pip install -r requirements-colab.txt
python urban_tree_ml_workshop_2026.py
```

Custom paths and a smaller SHAP sample can be supplied as follows:

```bash
python urban_tree_ml_workshop_2026.py \
  --input data/tree_carbon_ml_teaching_sample.csv \
  --output outputs \
  --shap-sample 600
```

## What the validation and test sets do

- **Training (70%)** fits an initial XGBoost model.
- **Validation (15%)** selects the number of boosting rounds by early stopping. It is not a second test set.
- **Development refit (85%)** refits the preprocessor and XGBoost using the selected number of trees on training plus validation data.
- **Locked test (15%)** is evaluated once after model selection. It estimates performance on unseen trees under the same sampled-data setting.

Test outcomes are not used to select parameters, thresholds, or stopping time. The small ONNX example is sampled deterministically only after the locked evaluation is complete.

### Verified reference run

The committed reference artifacts were generated with random seed 2026 and 543 validation-selected boosting rounds.

| Evaluation | N | R² log-SGR | RMSE log-SGR | MAE log-SGR | MAE percentage points | MAE kg C tree⁻¹ yr⁻¹ |
|---|---:|---:|---:|---:|---:|---:|
| Validation model selection | 5,575 | 0.427 | 1.060 | 0.787 | 4.42 | 5.16 |
| Locked test after 85% refit | 5,578 | 0.431 | 1.062 | 0.779 | 4.41 | 4.96 |

The percentage-point test RMSE is larger than its MAE because back-transformation magnifies a small number of high-growth observations. The log-SGR metrics remain the primary model-scale comparison.

### XGBoost settings shown in the lesson

The notebook displays the executed settings before fitting: learning rate 0.03, maximum depth 4, minimum child weight 8, row subsample 0.80, column subsample 0.85, gamma 0.02, L1 regularization 0.30, L2 regularization 10.0, a 1,500-tree selection cap, and 60-round validation early stopping. The verified run selected 543 trees before the 85% refit.

## Reading the environment-only SHAP outputs

- **Beeswarm:** summarizes only the seven environmental contribution distributions across the pooled test sample. Height, species, period, and site type are omitted from display.
- **Dependence:** x is the observed value of an environmental predictor, y is its SHAP contribution to log-SGR, and colour denotes its strongest approximate environmental interaction. Non-environment variables cannot appear on the colour bars.
- **Waterfall:** starts from a contextual value that already contains height, species, period, and site-type SHAP contributions; the visible bars show only how the environmental block moves that value to the complete prediction.
- **Spatial SHAP:** maps the local contribution of the highest-ranked environmental predictor at sampled test-tree coordinates. It is not an interpolated surface, a causal effect map, or a direct map of measured carbon gain.

Positive SHAP values raise the fitted prediction relative to the model baseline; negative values lower it. SHAP describes fitted associations and interactions, not causal environmental effects.

## ONNX input and output guidance

The exported model is `reference_outputs/models/pooled_urban_tree_growth.onnx`. The ONNX graph has two inputs:

| Input | Type and shape | Meaning |
|---|---|---|
| `features` | `float32 [N, F]` | Preprocessed numeric and one-hot features in the exact order stored in `feature_schema.json`. |
| `initial_carbon_kg` | `float32 [N, 1]` | Starting carbon stock used only to convert relative growth into kg C tree⁻¹ yr⁻¹. |

It returns four outputs:

| Output | Meaning |
|---|---|
| `annual_growth_percent` | Back-transformed one-year carbon growth in percent. |
| `carbon_gain_kg_tree_year` | Back-transformed annual carbon gain using `initial_carbon_kg`. |
| `log_sgr` | Raw XGBoost target prediction. |
| `specific_growth_rate` | `exp(log_sgr)`, in year⁻¹. |

ONNX receives engineered values rather than raw strings. Use `pooled_preprocessor.joblib` in Python, or reproduce the exact one-hot transformation described by `feature_schema.json`. The folder includes:

- `example_test_set_raw.csv`: human-readable raw test cases;
- `example_test_set_engineered.csv`: the exact ONNX input matrix plus initial carbon;
- `example_onnx_predictions.csv`: verified expected outputs;
- `onnx_parity_report.json`: maximum numerical difference between Python XGBoost and ONNX Runtime.

In the verified run, maximum absolute ONNX differences were 4.77×10⁻⁶ log-SGR, 1.15×10⁻⁵ percentage points, and 3.22×10⁻⁵ kg C.

## Teaching prompts

1. Why is the split assigned before converting trees into period rows?
2. What information can the validation set influence, and what must remain locked?
3. Which numeric predictor has the largest VIF, and does it exceed a conventional review threshold?
4. Compare the environmental beeswarm with the three dependence panels. Can a globally important predictor still have a nonlinear local association?
5. In the environmental waterfall, does the environmental block move the representative prediction above or below its contextual starting value?
6. Why must a spatial SHAP map be labelled as model contribution rather than environmental effect?
7. Change one engineered value in the ONNX example and predict again. Does the change agree with the dependence plot locally?

## Reproducibility and interpretation boundaries

- Random seeds are fixed in the configuration.
- The included test metrics are a teaching checkpoint, not a replacement for spatially blocked external deployment validation.
- The pooled model retains period indicators for retrospective learning. A future-scenario or planning deployment model should remove period or fix every prediction to one common reference period.
- Test performance applies to this filtered teaching sample and target definition.
- Do not interpret full model R² as the percentage impact of the environment.
