# Data access and placement

The exact analysis dataset is intentionally not committed to this public repository. The manuscript's data-availability statement identifies the municipal and open geospatial sources and states that the assembled tree-level dataset is available from the authors subject to the applicable permissions. Keeping it out of Git avoids redistributing derived records without confirming those permissions.

To reproduce the numerical analyses, place the author-supplied file at:

```text
journal_review_release/data/tree_carbon_updated4.csv
```

The analytical unit is a positive-growth tree-period observation. Repeated monitoring periods from the same tree are linked by `OID_` and must remain in the same split. The response is constructed as:

```text
g = [ln(C_end) - ln(C_start)] / years
y = ln(g)
```

`g` is annualized specific carbon growth (year^-1); `y` is the log-transformed modelling target. Errors measured on `y` are log-SGR errors, not kg C. Percentage-point errors and kg C tree^-1 year^-1 errors are reported only after explicit back-transformation.

For spatial diagnosis, separately place the point-grid CSV and municipal raster layers in a user-selected input directory. The spatial grid must use a projected metric coordinate reference system; the manuscript analysis used EPSG:3879 and 2 m cells.

See [required_columns.csv](required_columns.csv) for the principal tree-table fields. Several upstream fields may have aliases handled by `scripts/run_relative_growth_pipeline_v2.py`; the script performs the authoritative validation.
