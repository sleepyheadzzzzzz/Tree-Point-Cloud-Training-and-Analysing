# Tree Growth Workbench for QGIS

An experimental research plugin for connected steps: **train and compare →
finalize and test → spatial diagnosis → click-based interpretation → area planting**. It does
not change the archived manuscript results or turn model associations into
validated planting prescriptions. QGIS 3.40.11 is the tested desktop version.

## Install

```bash
python scripts/build_qgis_waterfall_plugin.py --output outputs/tree_growth_workbench.zip
```

Install through **Plugins → Manage and Install Plugins → Install from ZIP**.
Restart QGIS when upgrading an already loaded version. Open **Raster → Tree
Growth → Tree Growth Workbench**. The same internal plugin ID is retained so
this upgrades the earlier Tree Growth Waterfall plugin rather than creating a
second unrelated plugin.

See [publication and environment setup](QGIS_PUBLICATION.md). Version 2.1.0 is
an experimental ZIP candidate; official repository submission/approval is separate.

Select the external scientific Python executable under **Interpretation**.
It must satisfy the repository `requirements.txt`; QGIS's own Python does not
need the machine-learning dependencies. Leave extra module paths blank when
using a normal configured virtual/Conda environment. The installer contains
the workflow workers and current frozen journal deployment model, but **not
private training observations or full-area study rasters**.

The locally delivered launcher preselects the working Python and full-area
package. Run it in **Plugins → Python Console → Show Editor**, not Processing's
algorithm Script Editor. Reopen the file after updating it.

## 1. Training and model comparison

Choose a tree CSV and a **new** output directory. Select a projected coordinate
system in metres, block size (default 500 m), and model settings. The editable
JSON includes XGBoost learning rate, depth, subsampling, column subsampling,
regularization and tree count; RF depth, tree count, leaf size and feature
sampling; MLP hidden layers, learning rate, regularization and iteration count.

The backend compares OLS, RF, XGBoost and MLP on the same approximately 70/15/15
spatial-block partition. Every OID_ and all trees in a block stay in one split.
Block assignment is balanced using counts/species, never held-out growth values.
The selected model is the highest validation R², with RMSE as the tie-breaker.
MLP internal row-level early stopping is disabled to avoid separating periods
from the same training tree. Non-convergence warnings remain in the run report.

Training-only medians supply missing predictors, and OLS/MLP scaling is fitted
on training only. Environmental VIF is diagnostic: correlated/constant features
are reported, not automatically removed. Training and validation metrics are
saved; no test performance is reported at this stage.

### Input data

Either supply the journal's original wide-format table (including soil), or
one row per tree-period with:

```text
OID_, Period, X, Y, Species, Height, Initial_Carbon, End_Carbon, Years,
avg_noise_day, Density25, Mono_Rate, avg_svf, avg_radiation, avg_LST,
lightemiss, type_Puisto, soil_infill, soil_bedrock, soil_moraine
```

Height is metres; carbon stocks are kg C; Years is the observation interval;
Species uses the existing codes 1–11. Missing noise/sentinel values receive
40 dB under this project's quiet-noise convention. Other missing environmental
values use training-fitted medians. A raw `soil` column can replace the three
indicators using the existing Helsinki code mapping (`t`, `ka`, `mr`); this is
not a general-purpose soil classifier. Missing soil codes remain missing, not
silently encoded as three absences. Tree location and species must not change
between periods; duplicate OID_/Period records are rejected.

The target is `ln((ln(C_end)-ln(C_start))/Years)`. Positive height/stocks/years,
increasing carbon and known species are required. This model is for the retained
positive-growth population, not mortality or negative net carbon balance.

**Difference from the historical manuscript pipeline:** new plugin training
does not apply the old whole-dataset P05–P95 absolute-growth filter before
splitting. No categorical period enters this deployment specification. These
are explicitly new experiments; their sample counts and performance need not
match the retrospective or archived deployment analyses.

## 2. Finalization and locked validation

After reviewing validation performance, select **Refit selected model on 85%
and open locked test ONCE**. The model and preprocessing are refitted on
training plus validation; test blocks are then evaluated. A persistent
test-access marker prevents silent repeat finalization. Failed attempts also
leave the marker for audit. Changing/tuning experiments after seeing those same
held-out trees' outcomes invalidates a genuinely independent test claim; a file
marker cannot prevent scientific misuse across copied/new directories.

Outputs include R²/RMSE/MAE in log-SGR, annual-growth percentage-point errors,
and interval-consistent kg C/tree/year errors using each tree's initial stock.
They also include calibration, fixed seven-level agreement and weighted kappa,
tree-cluster bootstrap R² intervals, and residual Moran's I. The intervals are
tree-bootstrap intervals, not spatial-block bootstrap intervals. Spatial blocks
may share boundaries; this is not buffered spatial validation.

New-run thresholds are development-outcome septiles, frozen for subsequent
maps and scenarios. They are not the historical standardized-height thresholds
or an automatic reproduction of the journal's separately selected thresholds.
No claim of improved suitability classification follows from adding the plugin.

`finalized/deployment.json` ties the selected model, its preprocessing, domain,
and thresholds together. All four selected algorithms can be deployed and
explained. Only species actually represented in training are offered; the
eleven-band numbering is retained, with unsupported species bands left NoData.

## 3. Spatial diagnosis

Choose an aligned environmental point-grid CSV and a new map-package folder.
Use a finalized training run or leave it blank for the bundled, frozen current
journal model. Select fixed height, park fallback, coordinate system and pixel
size. An optional GeoTIFF template sets geometry only; it cannot supply missing
predictors. Grid CRS must match the training coordinate system.

Current-model packages need all eleven environmental indicators, including
soil. See [column mapping and outputs](CLICKABLE_SPATIAL_WATERFALL.md). Source
values may vary across the three periods or be shared explicitly; a shared
variable has no measured temporal change. New exports retain the 40 dB rule,
but other missing grid inputs produce NoData rather than unexplained
median-filled predictions. No grid-wide SHAP images are stored: explanations
are calculated on demand from the matched input rasters.

## 4. Interpretation

Open `manifest.json`, choose a species, period or change mode, and load the map.
Click a coloured cell. The waterfall shows either reference-to-local predicted
growth or earlier-to-later predicted growth. Contributions use percentage
points and sum to the continuous mapped difference. Suitability classes are
shown separately and are not additive SHAP quantities. Save SVG/CSV/JSON for a
figure or audit. The model checksum, feature order, raster grid and endpoint
predictions are checked before displaying a result.

This is exact **reference-based grouped Shapley attribution of back-transformed
growth**, not raw log-SGR TreeSHAP with a percentage label. Soil moves jointly
where present. It explains an explicitly chosen model contrast, not the
population-expected prediction or a causal environmental effect. Correlated
predictors can make hybrid scenarios unrealistic. Reliability 1 (in-domain)
does not establish joint support, causal validity or calibrated confidence.
See the [SHAP background-to-prediction explanation](https://shap.readthedocs.io/en/stable/example_notebooks/api_examples/plots/waterfall.html)
for the general waterfall concept and the package's method notes for this
reference-specific attribution game.

Models generated by the workbench use joblib to preserve selected estimators
and scalers. Joblib can execute code when loaded: use only packages you created
or trust. The QGIS interface asks before loading an externally opened joblib
package. JSON XGBoost packages remain supported.

## 5. Area planting

The **4. Area planting** workflow tab converts a single period's fixed suitability
into suitable-genus counts, suitable-area polygons and two exclusive allocation
alternatives. Set boundary/exclusion polygons, optional aligned reliability,
minimum patch area and the permitted diversity/suitability trade-off. QGIS exports
GeoPackage and Shapefile and loads styled layers. See the full
[inputs, rules, outputs and synthetic exercise](QGIS_AREA_PLANTING.md).

## Restoring the existing full-area study maps

The July-28 eight-input model can be restored separately:

```bash
python scripts/restore_archived_clickable_package.py \
  --archive path/to/spatial_diagnosis_relative_growth_noise40_v2_20260728 \
  --input path/to/original/prediction_area/tree.csv \
  --model-dir path/to/relative_growth_pipeline_v2_20260728_final/models \
  --output outputs/restored_full_area
```

The adapter copies the original growth, deviation, suitability and change
rasters **without changing their values**, restores original inputs and cleaning,
and verifies sampled predictions before publishing its manifest. The legacy
model has eight environmental indicators and no soil. It must not be described
as the later three-soil model. Static noise, illumination and park settings
cannot be interpreted as temporal changes.

Original median-imputed cells retain predictions, but get reliability code 0
and an explicit imputation warning; their waterfall explains the imputed model
inputs, not known site conditions. Source-footprint gaps remain NoData. The
restored study package contains 3,988,736 populated cells and 58 imputed cells
per period. Those private rasters stay local rather than being uploaded here.

## Tests and limits

```bash
python tests/test_spatial_waterfall.py
python tests/test_tree_growth_workbench.py
python scripts/validate_clickable_spatial_package.py --package outputs/restored_full_area/manifest.json --output outputs/audit.json
```

The installed ZIP is tested in real headless QGIS, including asynchronous jobs
and map-click SVG rendering. Technical reconstruction/additivity checks do not
increase predictive accuracy. Site feasibility, available planting space,
utilities, species survival and field validation still require independent
assessment. The workbench is research decision support, not a ready-to-use
causal environmental remediation or planting approval system.
