"""Synthetic planting constraints; no study data or manuscript refit."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import joblib
import numpy as np
import rasterio
from scipy.ndimage import label

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
from area_planting import allocate, retain_patches, run, make_demo


class PlantingTests(unittest.TestCase):
    def test_strict_area_and_diagonal(self):
        diagonal = np.eye(4, dtype=bool)
        self.assertFalse(retain_patches(diagonal, 4, 4).any())
        self.assertEqual(retain_patches(np.ones((2,2), bool), 4, 10).sum(), 4)
        self.assertFalse(retain_patches(np.ones((2,2), bool), 4, 16).any())

    def test_highest_and_diversity_constraints(self):
        rng = np.random.default_rng(34)
        levels = rng.integers(4, 8, (9,18,20)).astype(float)
        valid = np.ones_like(levels, bool)
        eligible, count, highest, diversity, top, _ = allocate(levels, valid, 4, 10, 5, 1)
        for assignment in [highest, diversity]:
            for i in range(9):
                mask = assignment == i
                self.assertTrue(eligible[i, mask].all())
                self.assertTrue((levels[i, mask] >= 5).all())
                self.assertTrue((top[mask]-levels[i, mask] <= (0 if assignment is highest else 1)).all())
                labels, _ = label(mask)
                self.assertTrue((np.bincount(labels.ravel())[1:]*4 > 10).all())
        self.assertTrue((count <= 9).all())

    def test_all_unsuitable_and_unknown(self):
        levels = np.full((9,5,5), 3.)
        valid = np.ones_like(levels, bool)
        valid[2, 0, 0] = False
        _, count, highest, diversity, _, _ = allocate(levels, valid, 4)
        self.assertEqual(count[0,0], 255)
        self.assertEqual(count[1,1], 0)
        self.assertTrue((highest == -1).all())
        self.assertTrue((diversity == -1).all())

    def test_single_feasible_genus(self):
        levels = np.full((9,6,6), 3.)
        levels[4] = 7
        _, _, highest, diversity, _, _ = allocate(levels, np.ones_like(levels,bool), 4)
        self.assertTrue((highest == 4).all())
        self.assertTrue((diversity == 4).all())

    def test_raster_roundtrip_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); make_demo(root/"demo")
            config = dict(input=str(root/"demo/suitability.tif"), output=str(root/"plan"))
            report = run(config)
            self.assertEqual(report["count_legend"], [0,9])
            for strategy in ["highest_suitability", "diversity_oriented"]:
                self.assertEqual(report["summaries"][strategy]["area_by_genus_m2"]["Sorbus"], 0)
                geo = json.loads((root/"plan"/(strategy+".geojson")).read_text())
                area = sum(f["properties"]["area_m2"] for f in geo["features"])
                self.assertAlmostEqual(area, report["summaries"][strategy]["assigned_area_m2"])
                self.assertTrue(all(f["properties"]["area_m2"] > 10 for f in geo["features"]))
            with rasterio.open(root/"plan/suitable_genus_count.tif") as src:
                self.assertEqual(src.nodata, 255)
                self.assertTrue(src.read(1, masked=True).mask[:2].all())
            with self.assertRaises(FileExistsError): run(config)

    def test_boundary_reliability_and_grid_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); make_demo(root/"demo")
            source=root/"demo/suitability.tif"
            with rasterio.open(source) as src: profile=src.profile.copy()
            profile.update(count=1)
            reliability=np.ones((40,50), "uint8"); reliability[:,10]=2
            with rasterio.open(root/"reliability.tif", "w", **profile) as dst: dst.write(reliability,1)
            geometry=dict(type="Polygon",coordinates=[[[25496000,6674000],[25496050,6674000],
                [25496050,6673920],[25496000,6673920],[25496000,6674000]]])
            config=dict(input=str(source),output=str(root/"plan"),reliability=str(root/"reliability.tif"),boundary_geometries=[geometry])
            run(config)
            with rasterio.open(root/"plan/suitable_genus_count.tif") as src:
                array=src.read(1)
                self.assertTrue((array[:,25:]==255).all())
                self.assertTrue((array[:,10]==255).all())
                self.assertTrue((array[3:,:10]!=255).any())
            profile["transform"] = rasterio.transform.from_origin(0,0,2,2)
            with rasterio.open(root/"bad.tif", "w", **profile) as dst: dst.write(reliability,1)
            config.update(output=str(root/"badplan"),reliability=str(root/"bad.tif"))
            with self.assertRaisesRegex(ValueError,"same CRS/grid"): run(config)

    def test_reject_change_values_and_wrong_bands(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); make_demo(root/"demo")
            source=root/"demo/suitability.tif"
            config=dict(input=str(source),output=str(root/"plan"))
            config["bands"]=[4,3,5,6,7,8,9,10,11]
            with self.assertRaisesRegex(ValueError,"does not match"): run(config)
            config.pop("bands")
            with rasterio.open(source,"r+") as dst:
                array=dst.read(3); array[5,5]=8; dst.write(array,3)
            with self.assertRaisesRegex(ValueError,"integer suitability"): run(config)

    def test_json_preprocessing_unchanged(self):
        original=joblib.load(ROOT/"models/preprocessing_spatial_deployment.joblib")
        text=json.loads((ROOT/"results/spatial_validation/deployment_preprocessing.json").read_text())
        self.assertFalse(original["use_scaled"])
        self.assertEqual(original["feature_columns"],text["feature_columns"])
        self.assertEqual(original["feature_medians"].to_dict(),text["feature_medians"])


if __name__=="__main__": unittest.main()
