# Clickable suitability and change diagnosis

The QGIS **TreeSuit XAI** workbench connects a matched raster package to a
cell-specific environmental explanation. The GeoTIFFs also open in ArcGIS;
the interactive side panel is implemented for QGIS 3.x.

See [QGIS_WORKBENCH.md](QGIS_WORKBENCH.md) for the added training, locked
validation and diagnosis interfaces, and restoration of the original full-area
eight-input map package. The default export instructions below use the newer
eleven-input model; archived packages retain their own model and feature groups.

## What is explained

- **Local mode:** reference-environment predicted annual growth to local predicted
  annual growth, holding the selected species and reference height fixed.
- **Change mode:** earlier to later predicted annual growth at the same cell,
  holding species, height, and the period-free model fixed.

Both waterfalls use **annual percentage points**, not raw log-SGR SHAP relabelled
as percent. The sum of the contribution bars equals the continuous mapped
deviation/change. Seven-level suitability and its integer transition are shown
separately: class numbers are not additive environmental contributions.

The model, target, and fixed thresholds are not refitted or selected here.
Predictions retain the existing transformation:

```text
annual growth (%) = 100 * expm1(exp(predicted log-SGR))
environmental deviation (pp) = local growth (%) - reference growth (%)
growth change (pp) = later growth (%) - earlier growth (%)
```

### Exact attribution method

For each environmental group, all subsets of the other changed groups are
evaluated. A group's marginal difference is weighted by the standard Shapley
factor `|S|! (m-|S|-1)! / m!`. The function being explained is the **complete
back-transformed model**, using one matched endpoint as the background.
The method is reference-dependent grouped Shapley attribution. It is not the
same attribution game as the manuscript's population-background TreeSHAP.

All eleven environmental input indicators are retained. The seven continuous
environmental predictors and park context have separate bars. Infill, bedrock,
and moraine move **jointly as one soil-context group**, preserving each
endpoint's joint soil coding when constructing hybrid scenarios. Thus the
default waterfall has nine environmental bars, not eleven independent bars.
Unchanged groups have zero contribution. Interactions are allocated among
changed groups through Shapley weighting.

With nine groups, at most 512 coalition evaluations are needed for a click;
unchanged groups reduce that count. This does not require calculating a whole
city's SHAP values or saving a separate image for every cell.

The decomposition is additive to numerical precision. Its interpretation is
still associative: hybrid environmental combinations can be unrealistic when
predictors are correlated, and endpoint min-max checks do not establish joint
distribution support, causality, or calibrated predictive confidence.

## Build a matched raster package

Use a Python environment satisfying the repository requirements. The QGIS click
worker itself needs only NumPy, XGBoost, and rasterio. Export additionally uses
pandas and joblib; load only trusted preprocessing artifacts.

```bash
python scripts/build_clickable_spatial_package.py \
  --input data/spatial_grid_with_soil.csv \
  --output outputs/clickable_spatial \
  --height 10 \
  --park-context 1 \
  --resolution 2 \
  --crs EPSG:3879
```

Defaults select the frozen spatially validated three-soil model, the archived
development domain, and the six fixed validation-selected overall thresholds.
Use `--template-raster data/site_suitability.tif` to use a prior site's exact
extent and grid; cells outside that extent are excluded. The template is used
for **geometry only**, not as a substitute for the original predictors.
Input coordinates must already match the cell centres and CRS. Misaligned or
duplicate cells raise an error instead of silently resampling/averaging data.

The output directory must be new. Original maps and source data are preserved.
Other than the 40 dB quiet-noise convention, missing environmental values are
**not median-imputed into an apparently valid diagnosis**: predictions for such
cells are NoData, and reliability is 0. This is intentionally stricter than
the legacy raster exporter and must be reported when comparing coverage.

### Input columns

`X` and `Y` are cell-centre coordinates. For each feature and period, the
preferred name is `FEATURE_PERIOD`, for example `avg_noise_day_15_17` or
`soil_infill_21_23`. A shared canonical column such as `soil_infill` is also
accepted and then stays constant over time. The existing legacy grid aliases
are accepted:

| Feature | 2015–2017 | 2017–2021 | 2021–2023 |
|---|---|---|---|
| `Density25` | `Density_15` | `Density_17` | `Density_21` |
| `Mono_Rate` | `Mono_Rate_` | `Mono_Rate1` | `Mono_Rat_1` |
| `avg_svf` | `svf15_17` | `svf17_21` | `svf21_23` |
| `avg_radiation` | `RA15_17` | `RA17_21` | `RA21_23` |
| `avg_LST` | `LST15_17` | `LST17_21` | `LST21_23` |

Other canonical names are `avg_noise_day`, `lightemiss`, `type_Puisto`,
`soil_infill`, `soil_bedrock`, and `soil_moraine`. Shared noise alias `noise`
is accepted. Soil and park indicators must be encoded 0/1. If no park column
exists, the explicitly selected `--park-context` supplies a common setting.

Period-specific canonical names take precedence over legacy names and shared
columns. The actual mapping is saved in `manifest.json`. A shared noise field
does **not** explain temporal noise change: supply period-specific noise values
if that change is intended. The same rule applies to density, illumination,
park context, and soil. Do not reinterpret a static layer as a measured change.

### Output rasters

The package contains `model.json`, `manifest.json`, and:

- `environment_PERIOD.tif`: eleven environmental input bands in documented order.
- `growth_PERIOD.tif`: eleven species/category bands, annual growth (%).
- `deviation_PERIOD.tif`: eleven species/category bands, local minus reference (pp).
- `suitability_PERIOD.tif`: fixed levels 1–7; 0 is NoData.
- `reliability_PERIOD.tif`: 0 missing, 1 inside development min-max, 2 outside range.
- `within_p01_p99_PERIOD.tif`: conservative feature-range flag, not confidence.
- `growth_change_pp.tif`: later minus earlier growth, with joint valid coverage.
- `suitability_level_change.tif`: later minus earlier level; -128 is NoData.
- `change_reliability.tif`: both endpoints must be valid; either out-of-range
  endpoint makes the pair code 2.

Species-band order remains 1 General Conifer, 2 General Broadleaf, 3 Acer,
4 Alnus, 5 Betula, 6 Pinus, 7 Prunus, 8 Quercus, 9 Sorbus, 10 Tilia, 11 Ulmus.
No tree-level species code 0 is invented. Selection of a genus is a standardized
scenario, not identification of the tree currently occupying a cell.

## QGIS installation and use

```bash
python scripts/build_qgis_waterfall_plugin.py --output outputs/tree_growth_waterfall.zip
```

1. In QGIS, open **Plugins → Manage and Install Plugins → Install from ZIP**.
2. Install the ZIP and open **Plugins → Tree Growth → TreeSuit XAI**.
3. Browse to the matched package's `manifest.json`.
4. Select the Python executable in the ML environment used to generate the
   package. QGIS's bundled Python does not need XGBoost installed.
5. Leave extra module paths blank for an ordinary virtual/Conda environment.
6. Select a species, local/reference or temporal-change mode, and map type.
7. Click **Load / update map**, then **Activate click explanation**.
8. Click a valid cell. **Save last result** exports its SVG waterfall, JSON
   metadata/endpoints, and CSV contribution table.

The worker is asynchronous and checks frozen-model checksum, feature order,
species bands, endpoint prediction agreement, and deviation/change agreement.
An explanation is rejected if it does not match the raster package. Missing
cells display no diagnosis; extrapolations display a warning. The continuous
map display uses a red–white–green -5 to +5 pp reference colour scale (values
beyond it saturate); the underlying raster numbers are not clipped.

For ArcGIS users, the same cell can be queried by coordinate using the CLI:

```bash
python scripts/explain_spatial_cell.py \
  --package outputs/clickable_spatial/manifest.json \
  --x 25496993 --y 6675075 --species 10 --mode change \
  --output outputs/click_examples/tilia_change
```

### Launcher troubleshooting

A local `START_IN_QGIS.py` launcher belongs in **Plugins → Python Console →
Show Editor**, not the Processing Toolbox's Script Editor. Processing executes
code in an empty namespace and expects a processing algorithm, whereas this
launcher opens an interactive dock. A launcher must explicitly obtain the
desktop interface with `from qgis.utils import iface`; it must not rely on the
console's injected `iface` variable. Keep the plugin instance alive beyond the
editor namespace. If Processing reports “No script found”, use the Python
Console editor or install the plugin ZIP instead. Reopen the launcher after
updating it so QGIS does not run a stale editor copy.

## Verification

```bash
python tests/test_spatial_waterfall.py
python scripts/validate_clickable_spatial_package.py \
  --package outputs/clickable_spatial/manifest.json \
  --output outputs/clickable_spatial_validation.json
```

Tests cover exact interaction allocation, grouped soil-style masking, dummy
features, reversed contrasts, all species, missing input, the 40 dB floor,
out-of-domain flags, and detection of tampered rasters. Test fixtures are
synthetic; no private grid data are included in the repository.

### Status of the local study application

The first version used a 225-observed-cell demonstration while the original
source drive was unavailable. After that drive was reconnected, the full
3,988,736-cell July-28 area was restored using the original inputs, eight-input
model, preprocessing and thresholds. Original map values remain unchanged.
The restored package supports true local and temporal waterfalls over its
populated footprint; it is not relabelled as the later three-soil model.
See the workbench guide for the restoration command and missing-input flags.
Generating new full-area maps with the current model still requires aligned
soil predictors, and temporal noise changes require period-specific noise.

## Methods wording for a future manuscript update

> Cell-specific environmental explanations were calculated for the frozen,
> period-free deployment model using reference-based grouped Shapley values of
> back-transformed annual percentage growth. Species and reference height were
> held constant. Contributions summed to either local-minus-reference growth
> or later-minus-earlier growth in percentage points. Soil indicators were
> treated as a joint group. Fixed suitability classes were reported separately
> from the continuous decomposition. These explanations describe
> reference-dependent model associations rather than causal effects.

Use this wording for the **implemented and verified scope**; do not claim
study-wide application until the full input grid has been regenerated and audited.

References: [SHAP exact explanations](https://shap.readthedocs.io/en/latest/generated/shap.ExactExplainer.html),
[QGIS map tools](https://docs.qgis.org/3.44/en/docs/pyqgis_developer_cookbook/canvas.html),
[QGIS raster queries](https://docs.qgis.org/3.44/en/docs/pyqgis_developer_cookbook/raster.html).
