# Area planting: two alternative densification proposals

Open **Raster → Tree Growth → Tree Growth Workbench**, then the workflow tab
**4. Area planting**. This extends model diagnosis into explicit, inspectable
planning alternatives. It does not retrain the model or validate a planting design.

## Inputs

| Control | Meaning |
|---|---|
| Suitability GeoTIFF | A single period's fixed integer levels 1–7. Not a growth, deviation or change raster. |
| Genus bands | Default 3–11: Acer, Alnus, Betula, Pinus, Prunus, Quercus, Sorbus, Tilia, Ulmus. Verify this order. General conifer/broadleaf bands are not genera. |
| Boundary polygons | Optional planting-site boundary, in any correctly declared CRS; QGIS transforms it to the raster CRS. |
| Exclusion polygons | Optional buildings, water, infrastructure setbacks or existing-canopy exclusion areas. Buffer point/line features to polygons in GIS first. |
| Reliability | Optional exactly aligned raster: retain only code 1. One common band or the same genus-band layout as suitability. Code 1 is a domain flag, not a probability of success. |
| Minimum level | Default 5, accepting levels 5–7. |
| Minimum patch area | Default **strictly greater than 10 m²**, four-neighbour connectivity; diagonal-only contact does not connect a patch. |
| Diversity level-loss limit | Default 2 levels below the best eligible genus; set 0 for top-level ties only or 1 for a narrower compromise. The minimum suitability level still applies. |
| Output folder | Must not exist. The source raster and earlier results are preserved. |

**Use current diagnosis period** copies the matching suitability/reliability paths
from the interpretation tab. A change map is rejected. Clip large citywide rasters
to a site first: this in-memory allocation is limited to 2 million grid cells.
Clipping must preserve exact pixel alignment if using a reliability raster.

Polygon masks use **pixel centres**. Exported boundaries follow the raster grid;
they are not exact vector clipping. A cell may extend across a boundary by part
of a pixel. For strict legal/property/setback boundaries, clip final polygons and
recheck their areas in GIS. The CRS's linear-unit conversion is used for m².

## Rules and outputs

1. Require valid observations for all nine configured genus bands at a cell.
   Any missing band, masked/excluded cell or failed reliability flag gives NoData.
   If a new model lacks a supported genus, this conservative nine-genus workflow
   must not claim a complete suitability count for that cell.
2. Retain each genus's suitable four-connected patches above the minimum area.
   Count retained suitable genera at each cell. The legend remains **0–9**;
   **0 = known but no retained suitable genus; 255/NoData = unknown or excluded**.
   Sorbus is not globally disabled: it receives zero area only when the input
   diagnosis contains no retained suitable Sorbus patch.
3. **Highest suitability:** choose the highest level among retained eligible
   genera; break level ties by the 5×5-cell neighbourhood mean, then source order.
   Drop small assigned fragments rather than replacing them with a lower-level
   genus. Every retained assignment therefore remains a local highest-level
   eligible choice. Some candidate land can remain unassigned.
4. **Diversity-oriented:** balance assigned area among feasible genera with a
   constrained target/bias heuristic, with suitability and local support as
   secondary criteria. Restrict each choice by the minimum level and maximum
   allowed loss, and remove small assigned fragments. This does **not** guarantee
   globally maximum richness or evenness. Report actual richness, area shares,
   area-based Shannon index, lost suitability levels and unassigned area.

Both alternatives are exclusive allocations within themselves. They are
**alternatives**, not two layers to combine into one plan. Candidate genus
polygons overlap deliberately to show multiple options at the same site.

The worker writes five GeoTIFFs and three projected GeoJSON layers:

- `suitable_genus_count.tif` (0–9, NoData 255).
- `highest_suitability_genus.tif` and `highest_suitability_level.tif`.
- `diversity_oriented_genus.tif` and `diversity_oriented_level.tif`.
- `species_suitable_areas.geojson`, `highest_suitability.geojson`,
  `diversity_oriented.geojson`.

Proposal rasters use **0 = unassigned, 255 = unknown/excluded**; genus codes are
3–11. QGIS then exports nonempty polygon layers as **GeoPackage and Shapefile**,
and saves QML styles. Keep `.shp`, `.shx`, `.dbf`, `.prj`, `.cpg` together.
GeoPackage is convenient for sharing one file. Empty alternatives remain valid
empty GeoJSON with an explicit report, rather than fabricated polygons.

Polygon attributes: `sp_code`, `genus`, `cells`, `area_m2`, `suit_mean`,
`poly_id`; proposals also have `loss_mean` relative to the best eligible level.
`planting_report.json` records input hash, settings, masks, area summaries and
warnings; `GIS_EXPORT.json` records native vector exports. Toggle **one proposal
at a time** in the new layer group. All outputs are local.

## Minimal public exercise

No study data are bundled. Generate a deterministic **synthetic** input:

```bash
python scripts/area_planting.py --demo outputs/synthetic_planting
```

Choose its `suitability.tif` in the plugin, retain default band order, and choose
a new output folder. This tests the interface without biological observations.
The generator is also in the installed plugin's `worker/area_planting.py`.
It deliberately includes an unsuitable Sorbus scenario and missing border cells.

Command-line use takes a JSON configuration:

```json
{
  "input": "outputs/synthetic_planting/suitability.tif",
  "output": "outputs/synthetic_proposals",
  "bands": [3,4,5,6,7,8,9,10,11],
  "min_area_m2": 10,
  "min_level": 5,
  "diversity_max_level_loss": 2
}
```

Run `python scripts/area_planting.py --config planting.json`. CLI outputs are
GeoTIFF/GeoJSON/report; QGIS performs native GPKG/SHP conversion. Advanced CLI
polygon masks must be GeoJSON geometry dictionaries in the **raster CRS** under
`boundary_geometries` and/or `exclusion_geometries`; the GUI handles reprojection.

## Interpretation and caption

Suggested caption: **Application of model-based growth suitability to area
planting scenarios.** The suitable-genus count shows the number of genus
scenarios meeting the fixed suitability threshold in retained connected patches.
Two alternatives allocate candidate land either to the locally highest-suitability
genus or through diversity-oriented area balancing subject to suitability and
minimum-area constraints. Unassigned and unknown areas are distinguished. These
outputs support comparative screening and early-stage design, and require
independent checks of planting space, infrastructure, soils, survival and management.

Genus-area balance is not a count of planted trees or a measured biodiversity
outcome. The workflow does not simulate feedback from densification on density,
monoculture, microclimate or later growth. Recalculate those environmental inputs
and re-run diagnosis to evaluate a changed design. Do not interpret an association
as the carbon gain caused by planting, or a seven-level score as survival probability.
