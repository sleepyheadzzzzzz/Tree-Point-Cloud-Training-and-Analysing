# Publishing TreeSuit XAI

## Status

Version 2.1.0 was uploaded to the official QGIS repository but blocked by the
initial automated scan. Version 2.1.1 corrected its security/quality findings,
passed that scan with zero issues, and adopted the public name **TreeSuit XAI**;
the separate Qt6 checker then reported 26 legacy enum aliases. Version 2.1.2 uses
Qt6-style scoped enums that were also verified in QGIS 3.40.11. It remains an
experimental submission candidate until it passes the repository checks and
maintainer review. Public author/contact metadata is set to
**Yao Chaowen — chaowen.yao@aalto.fi**. Installing a ZIP, submitting it, or
pushing source code to GitHub is not the same as QGIS approval. The live
[QGIS plugin record](https://plugins.qgis.org/plugins/tree_growth_waterfall/)
is the authoritative publication status.

## Local installation and dependencies

1. Obtain the source from this public repository.
2. Create a separate scientific Python environment (Python 3.12 is the tested
   local runtime). Do not replace QGIS's Python or install packages into it.

   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\python -m pip install -r requirements.txt
   # Linux/macOS equivalent:
   .venv/bin/python -m pip install -r requirements.txt
   ```

3. Build the installer with the repository root as the working directory:

   ```bash
   python scripts/build_qgis_waterfall_plugin.py --output outputs/TreeSuit_XAI_2.1.2.zip
   ```

4. In QGIS choose **Plugins → Manage and Install Plugins → Install from ZIP**.
   Select the ZIP, install, enable it, then restart QGIS after upgrading.
5. Open **Raster → Tree Growth → TreeSuit XAI**. In Interpretation,
   select the separate environment's Python executable. Extra module paths are
   normally blank. Follow the [user guide](QGIS_WORKBENCH.md) and
   [area planting exercise](QGIS_AREA_PLANTING.md).

The installer contains Python source, text JSON XGBoost/preprocessing, SVG icon,
CSV domain/threshold tables, documentation and license. It contains no compiled
libraries, joblib/pickle files, private observations or study rasters. Scientific
libraries are installed separately. The synthetic demonstration is generated
locally from source. No login, automatic download or telemetry is required by
the plugin; the documentation button opens the public guide only when clicked.

Models created by users can still use **trusted local joblib** bundles to retain
OLS/RF/MLP estimators and scalers. These can execute code when loaded: never open
untrusted training runs/model packages. They are not bundled in the installer.
Reviewers should examine the external-process and trust-warning code paths.

## Official repository submission

The [official publishing requirements](https://plugins.qgis.org/publish/) require
documentation, public code/issue links, a GPL-compatible license, declared external
dependencies, and a package below 20 MB without binaries. The current package
structure is designed around those requirements; only QGIS maintainers can
decide approval. The specialized workflow connects blocked carbon-growth model
comparison, reference-matched spatial explanations and genus-area allocation.
Search the QGIS catalogue for overlapping tools before submission and explain
that specific scope in the submission notes.

1. Verify `qgis/tree_growth_waterfall/metadata.txt`: the user-approved public
   author is **Yao Chaowen** and contact is **chaowen.yao@aalto.fi**. These details
   will be public. Confirm publication rights to all bundled model/code assets
   before release.
2. Review version, description, license, dependencies, changelog, icon, links and
   limits. QGIS 3.40.11 on Windows is tested; QGIS 4 is **not supported**. Linux/
   macOS are unverified, not claimed tested. Additional platform testing is advised.
3. Run the tests and inspect the generated ZIP:

   ```bash
   python tests/validate_project.py
   python tests/test_spatial_waterfall.py
   python tests/test_tree_growth_workbench.py
   python tests/test_area_planting.py
   python scripts/build_qgis_waterfall_plugin.py --official --output outputs/TreeSuit_XAI_2.1.2.zip
   ```

   `--official` rejects missing author/contact fields. Archive names must start
   with `tree_growth_waterfall/`; no `.git`, caches, private data or bundled
   executables. Test installation from that exact ZIP with a fresh QGIS profile.
   Use synthetic input to check both planning alternatives and GIS exports;
   verify training/finalization and map-click diagnosis with a trusted test package.
4. Commit the exact source used for the ZIP. A GitHub release can attach the ZIP
   and link these guides; release assets are an optional distribution route, not
   official QGIS publication. Keep this broad research repository: a second
   repository is not necessary.
5. Sign in to [QGIS Plugins](https://plugins.qgis.org/accounts/login/). The current
   login page offers **GitHub, GitLab, Google or OSGeo ID**; complete your chosen
   sign-in yourself. Then choose **Share a plugin** and upload the verified ZIP. Enter the requested
   metadata and retain the experimental designation. Do not share your password
   or access token in chat, source files or documentation.
6. Wait for the repository's security scan and manual review, and respond to
   reviewer feedback. Submission does not guarantee approval or a fixed deadline.
   See the [official approval process](https://plugins.qgis.org/docs/approval).
7. After approval, users can search for TreeSuit XAI in QGIS's plugin
   manager. Because it is experimental, they may need to enable experimental
   plugins. For updates, increment the version and changelog, test and upload a
   new version under the same plugin/package identity.

## Suggested submission description

TreeSuit XAI is an explainable-AI research tool for diagnosing urban-tree carbon
growth. It compares OLS, RF, XGBoost and MLP on grouped spatial training/validation
partitions, evaluates a locked spatial test, exports fixed-threshold growth
suitability and environmental contrasts, and explains map cells through exact
reference-matched grouped Shapley waterfalls. An area-planning function exports
suitable-genus counts and polygons plus highest-suitability and diversity-oriented
allocation alternatives. It uses a separate scientific Python environment.
The supplied frozen model is Helsinki-specific; outputs support research and
comparative site screening, not causal inference or planting approval.

## Scientific and security release boundaries

No original model is retrained by packaging, no manuscript metric is revised,
and no private full-area map is uploaded. JSON conversion preserves the frozen
unscaled model's feature order and numeric medians; old artifacts remain intact.
The new planting function may leave more land unassigned than the historical
figure because highest-suitability fragments are dropped rather than reassigned
to a lower-level genus. This is a new versioned planning rule, not a silent
replacement of the published illustration. See the run-specific report.
