"""Numerical and raster-contract tests; uses synthetic inputs, no private data."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
import xml.etree.ElementTree as ET

import joblib
import numpy as np
import pandas as pd
import rasterio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from spatial_waterfall_core import ENVIRONMENT, exact_grouped_contrast, SpatialPackage, waterfall_svg
from build_clickable_spatial_package import build, expected_model_hash


class ShapleyTests(unittest.TestCase):
    def test_explicit_digest_integrity_and_validation(self):
        import hashlib
        raw=hashlib.sha256(b"test model").digest()
        prep=dict(model_digest=dict(algorithm="sha256",bytes=list(raw)))
        self.assertEqual(expected_model_hash(prep),raw.hex())
        prep["model_digest"]["bytes"][0]=256
        with self.assertRaises(ValueError): expected_model_hash(prep)
        from validate_clickable_spatial_package import require
        with self.assertRaises(ValueError): require(False,"must always fail")

    @staticmethod
    def model(x):
        x = np.asarray(x, dtype=np.float64)
        growth = 2 + 3*x[:,0] + 4*x[:,1] + 6*x[:,0]*x[:,1]
        return np.log(np.log1p(growth/100))

    def test_interaction_allocation_and_dummy(self):
        result = exact_grouped_contrast(self.model,[0,0,3],[1,1,3],[[0],[1],[2]])
        np.testing.assert_allclose(result["contributions_pp"],[6,7,0],atol=1e-10)
        self.assertAlmostEqual(result["delta_pp"],13)
        self.assertEqual(result["coalitions_evaluated"],4)

    def test_joint_group(self):
        result = exact_grouped_contrast(self.model,[0,0],[1,1],[[0,1]])
        np.testing.assert_allclose(result["contributions_pp"],[13],atol=1e-10)

    def test_reversal_and_zero(self):
        forward = exact_grouped_contrast(self.model,[0,0],[1,1],[[0],[1]])
        reverse = exact_grouped_contrast(self.model,[1,1],[0,0],[[0],[1]])
        np.testing.assert_allclose(forward["contributions_pp"],-np.array(reverse["contributions_pp"]),atol=1e-10)
        same = exact_grouped_contrast(self.model,[1,1],[1,1],[[0],[1]])
        self.assertEqual(same["delta_pp"],0)
        self.assertEqual(same["coalitions_evaluated"],1)

    def test_reject_invalid_grouping(self):
        with self.assertRaises(ValueError):
            exact_grouped_contrast(self.model,[0,0],[1,1],[[0]])
        with self.assertRaises(ValueError):
            exact_grouped_contrast(self.model,[0,0],[1,1],[[0,1],[1]])


class RasterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.path = Path(cls.temp.name)
        prep = joblib.load(ROOT/"models/preprocessing_spatial_deployment.joblib")
        rows = []
        for i,(x,y) in enumerate([(1,1),(3,1),(5,1),(1,3),(5,3)]):
            row = dict(X=x,Y=y)
            for feature in ENVIRONMENT:
                for period in ["15_17","21_23"]:
                    row[f"{feature}_{period}"] = float(prep["feature_medians"][feature])
            if i == 1:
                row["Density25_15_17"],row["Density25_21_23"] = 5,35
                row["avg_noise_day_15_17"] = -9999
            if i == 2:
                row["avg_radiation_21_23"] = np.nan
            if i == 3:
                row["Density25_21_23"] = 999
            rows.append(row)
        pd.DataFrame(rows).to_csv(cls.path/"grid.csv",index=False)
        cls.args = SimpleNamespace(input=cls.path/"grid.csv",output=cls.path/"package",
            model=ROOT/"models/xgb_spatial_deployment_no_period_three_soil.json",
            preprocessing=ROOT/"models/preprocessing_spatial_deployment.joblib",
            domain=ROOT/"results/spatial_validation/development_training_domain.csv",
            thresholds=ROOT/"results/suitability/fixed_selected_seven_level_thresholds.csv",
            template_raster=None,resolution=2,crs="EPSG:3879",height=10,park_context=1,
            periods=["15_17","21_23"],earlier="15_17",later="21_23",chunk_size=2,scope="observed_cells_demo")
        build(cls.args)
        cls.package = SpatialPackage(cls.args.output/"manifest.json")

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_all_species_local_and_change_closure(self):
        for species in range(1,12):
            for mode in ["local","change"]:
                result = self.package.explain(3,1,species,mode,"21_23")
                self.assertEqual(result["status"],"ok")
                self.assertLess(abs(result["additivity_error_pp"]),1e-8)
                self.assertLess(result["max_raster_parity_error_pp"],3e-5)
                self.assertEqual(len(result["contributions_pp"]),9)
                ET.fromstring(waterfall_svg(result))

    def test_text_preprocessing_produces_identical_rasters(self):
        args=SimpleNamespace(**vars(self.args))
        args.preprocessing=ROOT/"results/spatial_validation/deployment_preprocessing.json"
        args.output=self.path/"text_package"
        build(args)
        for original in (self.args.output/"rasters").glob("*.tif"):
            with rasterio.open(original) as a, rasterio.open(args.output/"rasters"/original.name) as b:
                np.testing.assert_array_equal(a.read(),b.read())
                self.assertEqual(a.transform,b.transform)

    def test_changed_integrity_digest_blocks_export(self):
        args=SimpleNamespace(**vars(self.args))
        prep=json.loads((ROOT/"results/spatial_validation/deployment_preprocessing.json").read_text())
        prep["model_digest"]["bytes"][0] ^= 1
        args.preprocessing=self.path/"changed_digest.json"
        args.preprocessing.write_text(json.dumps(prep))
        args.output=self.path/"must_not_build"
        with self.assertRaisesRegex(ValueError,"frozen training/preprocessing"):
            build(args)
        self.assertFalse(args.output.exists())

    def test_missing_and_off_extent(self):
        for x,y in [(5,1),(3,3),(100,100)]:
            self.assertEqual(self.package.explain(x,y)["status"],"missing")

    def test_domain_warning_and_quiet_floor(self):
        result = self.package.explain(1,3)
        self.assertEqual(result["reliability_code"],2)
        result = self.package.explain(3,1,mode="change")
        self.assertEqual(result["start_environment"]["avg_noise_day"],40)

    def test_no_change_baseline(self):
        result = self.package.explain(1,1,mode="change")
        self.assertAlmostEqual(result["delta_pp"],0)
        np.testing.assert_allclose(result["contributions_pp"],np.zeros(9))

    def test_tampered_raster_rejected(self):
        path = self.args.output/"rasters/growth_21_23.tif"
        with rasterio.open(path,"r+") as raster:
            data = raster.read(2)
            old = data[0,2]
            data[0,2] += 1
            raster.write(data,2)
        try:
            with self.assertRaisesRegex(ValueError,"prediction mismatch"):
                self.package.explain(5,3)
        finally:
            with rasterio.open(path,"r+") as raster:
                data[0,2] = old
                raster.write(data,2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
