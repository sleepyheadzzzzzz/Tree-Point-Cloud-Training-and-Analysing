"""End-to-end scientific backend smoke tests using synthetic, non-private data."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import joblib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"scripts"))
from tree_growth_workbench import train,finalize,diagnose,normalized_data,make_model,FEATURES
from spatial_waterfall_core import ENVIRONMENT,SpatialPackage,ModelPredictor,exact_grouped_contrast


def synthetic_data():
    rng=np.random.default_rng(937);rows=[]
    for i in range(660):
        species=i%11+1;block=i//11;height=rng.uniform(6,22)
        for k,p in enumerate(["15_17","17_21","21_23"]):
            e=[rng.uniform(40,70),rng.uniform(3,35),rng.uniform(0,1),rng.uniform(.5,1),rng.uniform(400,1100),rng.uniform(20,30),rng.uniform(40,150),float(i%2),float(i%3==0),float(i%3==1),float(i%3==2)]
            y=-3.4+.08*np.log(height)+.022*e[1]-.003*e[0]+.009*species+rng.normal(0,.08)
            initial=rng.uniform(30,350);years=4 if k==1 else 2
            rows.append(dict(OID_=i,Period=p,Species=species,X=300000+(block%10)*500+10,Y=6600000+(block//10)*500+10,
                Height=height,Years=years,Initial_Carbon=initial,End_Carbon=initial*np.exp(np.exp(y)*years),**dict(zip(ENVIRONMENT,e))))
    return pd.DataFrame(rows)


class WorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();cls.path=Path(cls.temp.name)
        cls.data=synthetic_data();cls.data.to_csv(cls.path/"trees.csv",index=False)
        config=dict(input=str(cls.path/"trees.csv"),output=str(cls.path/"run"),split_search_iterations=30,
            bootstrap_repetitions=15,moran_permutations=9,models={"RF":{"n_estimators":12,"max_depth":6},
            "XGB":{"n_estimators":20,"max_depth":3},"MLP":{"max_iter":15,"hidden_layer_sizes":[12],"batch_size":128}})
        cls.report=train(config);cls.final=finalize(cls.path/"run")

    @classmethod
    def tearDownClass(cls): cls.temp.cleanup()

    def test_split_and_selection(self):
        self.assertTrue(self.report["no_tree_leakage"]);self.assertTrue(self.report["no_block_leakage"])
        metrics=pd.read_csv(self.path/"run/tables/candidate_metrics.csv")
        self.assertEqual(set(metrics.Split),{"Training","Validation"})
        self.assertEqual(len(metrics),8)
        best=metrics.loc[metrics.Split=="Validation"].sort_values(["R2_LogSGR","RMSE_LogSGR"],ascending=[False,True]).iloc[0].Model
        self.assertEqual(best,self.report["selected_model"])
        self.assertTrue((self.path/"run/tables/training_environment_vif.csv").exists())

    def test_refuse_second_test_access(self):
        with self.assertRaises(FileExistsError): finalize(self.path/"run")

    def test_refuse_changed_tree_location(self):
        broken=self.data.copy();broken.loc[0,"X"]+=1
        with self.assertRaisesRegex(ValueError,"changes location"): normalized_data(broken)

    def test_no_outcome_quantile_trimming(self):
        normalized,meta=normalized_data(self.data)
        self.assertEqual(len(normalized),len(self.data))

    def test_new_model_rasters_and_explanation(self):
        prep=joblib.load(self.path/"run/finalized/preprocessing.joblib")
        rows=[]
        for i in range(3):
            row=dict(X=1+i*2,Y=1)
            for f in ENVIRONMENT: row[f]=float(prep["feature_medians"][f])
            # Use actual valid binary indicators, not fractional medians.
            for f in ["type_Puisto",*ENVIRONMENT[8:]]: row[f]=0
            for p in ["15_17","17_21","21_23"]: row["Density25_"+p]=5+i*5+(4 if p=="21_23" else 0)
            rows.append(row)
        pd.DataFrame(rows).to_csv(self.path/"grid.csv",index=False)
        diagnose(dict(input=str(self.path/"grid.csv"),output=str(self.path/"maps"),training_run=str(self.path/"run")))
        package=SpatialPackage(self.path/"maps/manifest.json")
        for code in range(1,12):
            for mode in ["local","change"]:
                r=package.explain(1,1,code,mode)
                self.assertEqual(r["status"],"ok")
                self.assertLess(abs(sum(r["contributions_pp"])-r["delta_pp"]),1e-8)

    def test_all_estimator_adapters(self):
        from sklearn.preprocessing import StandardScaler
        rng=np.random.default_rng(42);x=rng.normal(size=(100,len(FEATURES))).astype("float32");y=-3+x[:,0]*.1
        scaler=StandardScaler().fit(x)
        for name,parameters in [("OLS",{}),("RF",{"n_estimators":3,"random_state":1}),
                                ("XGB",{"n_estimators":3,"n_jobs":1}),("MLP",{"hidden_layer_sizes":[4],"max_iter":10,"early_stopping":False,"random_state":1})]:
            scaled=name in ["OLS","MLP"];values=scaler.transform(x) if scaled else x
            model=make_model(name,parameters);model.fit(values,y)
            path=self.path/(name+".joblib")
            joblib.dump(dict(model=model,feature_columns=FEATURES,scaler=scaler,use_scaled=scaled),path)
            predictor=ModelPredictor(path,"trusted_joblib")
            np.testing.assert_allclose(predictor.inplace_predict(x[:3]),model.predict(values[:3]),rtol=1e-6)


if __name__=="__main__": unittest.main(verbosity=2)
