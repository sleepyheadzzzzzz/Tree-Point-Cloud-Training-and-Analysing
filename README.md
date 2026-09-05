# Tree Point Cloud Training and Analysing

## Urban-tree carbon growth modelling and spatial diagnosis

[![Project audit](https://img.shields.io/badge/project%20audit-passing-brightgreen)](tests/MANIFEST.sha256)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)

This repository contains the analysis underlying **“Towards Carbon Responsive Landscape Planning: Multi-Temporal LiDAR and Explainable Machine Learning for Urban Tree Growth Diagnosis.”** It provides an integrated workflow for relative-growth modelling, explanation, spatial validation, diagnosis, change detection, and planning outputs. Earlier exploratory LiDAR/carbon scripts are retained under `legacy_pipeline/` for provenance.

Teaching material for the pooled train–validation–test, SHAP, spatial-SHAP, and ONNX workflow is available in [eCAADe Workshop 2026 Material](eCAADe_workshop_2026_material/README.md).

New: [TreeSuit XAI for QGIS](docs/QGIS_WORKBENCH.md) is an explainable-AI
workbench for diagnosing urban-tree growth suitability. It combines editable
OLS/RF/XGBoost/MLP training, spatial validation, diagnosis-raster export and
[clickable environmental waterfalls](docs/CLICKABLE_SPATIAL_WATERFALL.md).
The local demo now restores all 3,988,736 cells of the archived diagnosis grid,
with matching original inputs/model rather than observed-tree coverage only.
New model experiments and the archived eight-input maps remain clearly separated.

Version 2.1 adds [area planting](docs/QGIS_AREA_PLANTING.md): a 0–9 suitable-genus
count map, suitable-area polygons and two alternatives—highest suitability and
diversity-oriented allocation—with GeoPackage/Shapefile export. See
[installation and official QGIS publication](docs/QGIS_PUBLICATION.md).
The plugin is experimental. Its QGIS repository submission is undergoing
automated and manual review; it is not yet an approved public release.

## What is reproducible here

- A common `OID_`-grouped comparison of OLS, random forest, XGBoost, and MLP.
- Pooled period-controlled XGBoost SHAP analysis using infill, bedrock, and moraine soil indicators; clay and silt/sand are excluded.
- A height-and-species biological baseline, nested incremental R2, partial R2, clustered uncertainty, and joint permutation tests.
- Pooled and species-specific SHAP statistics, dependence plots, nonlinear diagnostics, and categorized radial profiles.
- A 500 m blocked 70/15/15 deployment experiment with validation-only model selection and a locked spatial test.
- Fixed seven-level suitability and a more decision-ready three-zone diagnosis.
- Period-free 2 m mapping for a standardized 10 m tree, environmental contrasts, reliability flags, temporal change, suitable-genus polygons, and diversified densification proposals.

## Archived numerical checkpoints

The included compact tables reproduce the manuscript's principal results:

| Checkpoint | Archived value |
|---|---:|
| Retrospective XGBoost independent-test R2 | 0.611 |
| Height-and-species baseline R2 | 0.559 |
| Full-model R2 | 0.611 |
| Combined period + environment delta R2 | 0.052 |
| Combined period + environment partial R2 | 0.118 |
| Environment-only delta R2, conditional on species and period | 0.022 |
| Environment-only joint-permutation R2 loss | 0.0448 |
| Locked spatial-test XGBoost R2 | 0.564 |
| Seven-level agreement, exact / within one | 36.5% / 77.3% |
| Three-zone exact agreement | 68.7% |

Run `python tests/validate_project.py` to compare every checkpoint against the archived CSVs and to compile all Python scripts.

## Quick start

```bash
conda env create -f environment.yml
conda activate urban-tree-growth
python tests/validate_project.py
```

The exact assembled tree-level data are not publicly redistributed here. After obtaining the file under the manuscript's data-availability conditions, place it at `data/tree_carbon_updated4.csv`. Then follow [docs/WORKFLOW.md](docs/WORKFLOW.md). Principal columns and transformations are documented in [data/README.md](data/README.md) and [data/required_columns.csv](data/required_columns.csv).

## Directory guide

```text
Tree-Point-Cloud-Training-and-Analysing/
├── data/             input contract; exact study data intentionally absent
├── docs/             ordered commands and manuscript-to-code alignment
├── models/           frozen biological, retrospective, and deployment artifacts
├── results/          compact numerical audit trail
├── scripts/          analysis, validation, visualization, and GIS exporters
├── tests/            project integrity and numerical checkpoint audit
├── qgis/             training, validation, diagnosis and interpretation plugin
├── eCAADe_workshop_2026_material/  pooled Colab/Python teaching exercise
└── legacy_pipeline/  archived preliminary scripts and sample data
```

## Interpretation boundaries

- The full R2 is not attributed to the environment. Overall predictive fit, direct nested increments, partial R2, and permutation loss are reported separately.
- SHAP values and local-versus-reference environmental contrasts are fitted associations. They do not establish causal effects or measured environmental offsets.
- The seven-level raster retains useful detail but should not be presented as exact tree-level prediction. The three-zone output is the primary planning diagnostic.
- Residual spatial clustering remains after validation, so reliability masks and local field verification should accompany design decisions.
- The no-period deployment model fixes model structure across dates; only intended environmental layers change in temporal comparisons.

## Reproducibility notes

Random seeds and bootstrap/permutation counts are defined in the scripts. Output directories are never overwritten. Large derived rasters and observation-level SHAP tables are intentionally regenerated rather than versioned. Included `.joblib` artifacts should only be loaded from a trusted checkout.

See [docs/METHODS_ALIGNMENT.md](docs/METHODS_ALIGNMENT.md) for the manuscript-to-script index and [results/README.md](results/README.md) for the archived evidence map.

## License and citation

Code is provided under the repository's GPL-3.0 license. Citation metadata are in [CITATION.cff](CITATION.cff). Please cite the associated manuscript when using the scientific workflow.
