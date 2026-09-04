"""QGIS workbench backend: grouped spatial training, locked validation and diagnosis.

New runs are deployment experiments, not reproductions of manuscript statistics.
Test outcomes are not used for model choice or preprocessing. All serialized
artifacts are local/trusted; never load joblib from an untrusted source.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,cohen_kappa_score,confusion_matrix
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from spatial_waterfall_core import ENVIRONMENT,SPECIES,percentage,sha256
from run_relative_growth_pipeline_v2 import XGB_PARAMETERS,RF_PARAMETERS,MLP_PARAMETERS,metric_record
from run_spatial_block_validation_three_soil import choose_block_split,morans_i

ROOT=Path(__file__).resolve().parents[1]
FEATURES=["Log_Height"]+sorted("Species_"+s for s in SPECIES.values())+ENVIRONMENT


def write_json(path,data):
    path.write_text(json.dumps(data,indent=2,allow_nan=False,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)),encoding="utf-8")


def defaults():
    return dict(seed=42,block_size_m=500,split_search_iterations=2000,bootstrap_repetitions=250,moran_permutations=199,
                crs="EPSG:3879",models={"OLS":{},"RF":{**RF_PARAMETERS,"n_jobs":4},
                "XGB":{**XGB_PARAMETERS,"n_jobs":4},"MLP":{**MLP_PARAMETERS,"early_stopping":False}})


def normalized_data(raw):
    """Accept journal-wide or canonical tree-period data; never trim on test outcomes."""
    if {"Initial_Carbon","End_Carbon","Height","Years"}.issubset(raw.columns):
        frame=raw.copy()
        if "Period" not in frame: raise ValueError("Canonical data needs Period to identify repeated observations")
    else:
        needed={"OID_","X","Y","Species","type","noise17d","noise22d","lightemiss","Density25","Mono_Rate"}
        needed|={f"CS_{p}" for p in ["15","17","21","23"]}|{f"H{p}" for p in ["15","17","21"]}
        needed|={f"svf{p}" for p in ["15","17","21","23"]}|{"ra15_17","ra17_21","ra21_23","LST_1516","LST_1720","LST_2122"}
        if needed-set(raw): raise ValueError("Missing journal-wide columns: "+", ".join(sorted(needed-set(raw))))
        raw=raw.loc[raw.type.isin(["Katu","Puisto"])].copy()
        raw["avg_noise_day"]=raw[["noise17d","noise22d"]].apply(pd.to_numeric,errors="coerce").mask(lambda x:x<=-9990).mean(axis=1)
        raw["type_Puisto"]=(raw.type=="Puisto").astype(float)
        rows=[]
        for p,start,end,years,ra,lst in [("15_17","15","17",2,"ra15_17","LST_1516"),
                                       ("17_21","17","21",4,"ra17_21","LST_1720"),("21_23","21","23",2,"ra21_23","LST_2122")]:
            f=raw.copy();f["Period"]=p;f["Years"]=years
            for name,source in [("Initial_Carbon",f"CS_{start}"),("End_Carbon",f"CS_{end}"),("Height",f"H{start}"),("avg_radiation",ra),("avg_LST",lst)]:
                f[name]=pd.to_numeric(f[source],errors="coerce")
            f["avg_svf"]=f[[f"svf{start}",f"svf{end}"]].apply(pd.to_numeric,errors="coerce").mask(lambda x:x<=-9990).mean(axis=1)
            rows.append(f)
        frame=pd.concat(rows,ignore_index=True)
    if not set(ENVIRONMENT[8:]).issubset(frame.columns):
        if "soil" not in frame: raise ValueError("Provide soil or the three soil indicator columns")
        s=frame.soil.astype("string").str.lower().str.replace(" ","",regex=False)
        frame["soil_infill"]=s.str.startswith("t").astype(float)
        frame["soil_bedrock"]=s.str.contains("ka",regex=False).astype(float)
        frame["soil_moraine"]=s.str.contains("mr",regex=False).astype(float)
        frame.loc[s.isna()|s.eq(""),ENVIRONMENT[8:]]=np.nan
    need={"OID_","X","Y","Species","Period","Height","Initial_Carbon","End_Carbon","Years",*ENVIRONMENT}
    if need-set(frame): raise ValueError("Missing canonical columns: "+", ".join(sorted(need-set(frame))))
    before=len(frame)
    for name in ["X","Y","Species","Height","Initial_Carbon","End_Carbon","Years",*ENVIRONMENT]:
        frame[name]=pd.to_numeric(frame[name],errors="coerce")
    biological=frame[["Height","Initial_Carbon","End_Carbon","Years"]]
    valid=np.isfinite(biological).all(axis=1)&(biological>0).all(axis=1)&(frame.End_Carbon>frame.Initial_Carbon)&frame.Species.isin(SPECIES)
    frame=frame.loc[valid,sorted(need)].copy().reset_index(drop=True)
    if frame.empty: raise ValueError("No valid positive-growth observations")
    if frame.OID_.isna().any() or frame.Period.isna().any() or not np.isfinite(frame[["X","Y"]]).all().all():
        raise ValueError("OID_, Period and projected coordinates must be present")
    if frame.duplicated(["OID_","Period"]).any(): raise ValueError("Duplicate OID_ / Period observations")
    if (frame.groupby("OID_")[["X","Y","Species"]].nunique()>1).any().any():
        raise ValueError("A tree changes location or species across periods; resolve before splitting")
    frame[ENVIRONMENT]=frame[ENVIRONMENT].mask(~np.isfinite(frame[ENVIRONMENT])|(frame[ENVIRONMENT]<=-9990))
    frame["avg_noise_day"]=frame.avg_noise_day.fillna(40)
    frame.loc[frame.lightemiss<0,"lightemiss"]=np.nan
    for name in ["type_Puisto",*ENVIRONMENT[8:]]:
        frame.loc[~frame[name].isin([0,1]),name]=np.nan
    frame["Species_Name_Model"]=frame.Species.map(SPECIES)
    frame["Log_Height"]=np.log(frame.Height)
    frame["Log_Specific_Growth_Rate"]=np.log((np.log(frame.End_Carbon)-np.log(frame.Initial_Carbon))/frame.Years)
    for code,name in SPECIES.items(): frame["Species_"+name]=(frame.Species==code).astype(float)
    return frame,dict(input_rows=before,retained_rows=len(frame),trees=int(frame.OID_.nunique()),
        filter="Positive stocks, increasing carbon, positive height/years, known species; no outcome-quantile trimming.",
        target="ln((ln(C_end)-ln(C_start))/years)",period="Period omitted from deployment predictors",
        difference_from_journal="New training does not repeat the archived whole-dataset P05-P95 absolute-growth trim.")


def make_model(name,parameters):
    factories={"OLS":LinearRegression,"RF":RandomForestRegressor,"XGB":XGBRegressor,"MLP":MLPRegressor}
    if name=="MLP" and parameters.get("early_stopping",False):
        raise ValueError("MLP early_stopping must be false: its internal row split can leak repeated trees")
    return factories[name](**parameters)


def train(config):
    output=Path(config["output"])
    if output.exists(): raise FileExistsError("Choose a new training output folder")
    settings=defaults()
    settings.update({k:v for k,v in config.items() if k not in ["models","input","output"]})
    for name,parameters in config.get("models",{}).items():
        if name not in settings["models"]: raise ValueError("Unknown candidate model: "+name)
        settings["models"][name].update(parameters)
    if settings["block_size_m"]<=0 or settings["split_search_iterations"]<1: raise ValueError("Invalid block / search size")
    if settings["bootstrap_repetitions"]<1 or settings["moran_permutations"]<1: raise ValueError("Bootstrap and Moran repetitions must be positive")
    import rasterio
    crs=rasterio.crs.CRS.from_user_input(settings["crs"])
    if not crs.is_projected or abs(crs.linear_units_factor[1]-1)>1e-8: raise ValueError("Spatial blocking requires projected metre coordinates")
    data,construction=normalized_data(pd.read_csv(config["input"]))
    data["Spatial_Block"]=(np.floor(data.X/settings["block_size_m"]).astype(int).astype(str)+"_"+np.floor(data.Y/settings["block_size_m"]).astype(int).astype(str))
    tree=data.groupby("OID_",as_index=False).first()[["OID_","Spatial_Block","Species_Name_Model","X","Y"]]
    if tree.Spatial_Block.nunique()<7: raise ValueError("At least seven spatial blocks are required; reduce block size or expand data")
    split,diag=choose_block_split(tree,data.groupby("OID_").size(),int(settings["split_search_iterations"]),int(settings["seed"]))
    data["Spatial_Split"]=data.Spatial_Block.map(split)
    training=data.Spatial_Split.eq("Training");validation=data.Spatial_Split.eq("Validation")
    if any(int((data.Spatial_Split==p).sum())<3 for p in ["Training","Validation","Test"]): raise ValueError("Too few observations per split")
    supported=sorted(data.loc[training,"Species"].astype(int).unique().tolist())
    if not set(data.Species).issubset(supported): raise ValueError("A species has no training observations; revise split seed/block size before any test is opened")
    medians=data.loc[training,FEATURES].median()
    if medians.isna().any(): raise ValueError("All training values missing for: "+", ".join(medians.index[medians.isna()]))
    x=data[FEATURES].fillna(medians).to_numpy(np.float32);y=data.Log_Specific_Growth_Rate.to_numpy(np.float32)
    scaler=StandardScaler().fit(x[training]);rows=[];fit_warnings={}
    output.mkdir(parents=True);(output/"tables").mkdir()
    print(f"Training {training.sum():,}; validation {validation.sum():,}; test outcomes locked",flush=True)
    for name in ["OLS","RF","XGB","MLP"]:
        print("Fitting "+name,flush=True)
        model=make_model(name,settings["models"][name]);scaled=name in ["OLS","MLP"]
        values=scaler.transform(x) if scaled else x
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always");model.fit(values[training],y[training])
        fit_warnings[name]=list(dict.fromkeys(str(w.message) for w in caught))
        for label,mask in [("Training",training),("Validation",validation)]:
            predicted=model.predict(values[mask])
            metrics=metric_record(y[mask],predicted,data.loc[mask,"Initial_Carbon"].to_numpy(),data.loc[mask,"Years"].to_numpy())
            rows.append(dict(Model=name,Split=label,N_Rows=int(mask.sum()),**metrics))
    comparison=pd.DataFrame(rows)
    selected=comparison.loc[comparison.Split=="Validation"].sort_values(["R2_LogSGR","RMSE_LogSGR"],ascending=[False,True]).iloc[0].Model
    comparison.to_csv(output/"tables/candidate_metrics.csv",index=False)
    data[["OID_","Period","Spatial_Block","Spatial_Split","Species"]].to_csv(output/"tables/split_assignments.csv",index=False)
    # VIF is diagnostic only. No automatic feature removal and no access to test rows.
    vif=[];env=x[training][:,[FEATURES.index(f) for f in ENVIRONMENT]]
    for j,f in enumerate(ENVIRONMENT):
        yy=env[:,j];others=np.delete(env,j,axis=1)
        if np.ptp(yy)==0: value=None;note="Constant in training"
        else:
            rr=r2_score(yy,LinearRegression().fit(others,yy).predict(others))
            value=None if 1-rr<=1e-10 else float(1/(1-rr));note="Perfect collinearity" if value is None else ""
        vif.append(dict(Feature=f,VIF=value,Note=note))
    pd.DataFrame(vif).to_csv(output/"tables/training_environment_vif.csv",index=False)
    state=dict(data=data,features=FEATURES,settings=settings,selected=str(selected),supported_species=supported)
    joblib.dump(state,output/"training_state.joblib")
    report=dict(status="validation_complete_test_locked",selected_model=str(selected),settings=settings,construction=construction,
                split_diagnostics=diag,split_rows=data.Spatial_Split.value_counts().to_dict(),
                no_tree_leakage=bool(data.groupby("OID_").Spatial_Split.nunique().max()==1),
                no_block_leakage=bool(data.groupby("Spatial_Block").Spatial_Split.nunique().max()==1),
                source_sha256=sha256(config["input"]),training_state_sha256=sha256(output/"training_state.joblib"),warnings=fit_warnings)
    write_json(output/"training_report.json",report)
    print("Training/validation complete. Selected by validation only:",selected,flush=True)
    return report


def finalize(folder):
    folder=Path(folder)
    if not (folder/"training_report.json").is_file(): raise ValueError("Training has not completed")
    training_report=json.loads((folder/"training_report.json").read_text())
    if sha256(folder/"training_state.joblib")!=training_report["training_state_sha256"]:
        raise ValueError("Frozen training state was modified; cannot open the locked test")
    # A failed test-opening attempt also remains visible; never silently reopen it.
    with (folder/"TEST_ACCESS_STARTED.txt").open("x",encoding="utf-8") as f:
        f.write("Locked test was opened. Do not tune from these outcomes or silently repeat finalization.\n")
    state=joblib.load(folder/"training_state.joblib");data=state["data"];settings=state["settings"]
    dev=data.Spatial_Split.ne("Test");test=~dev
    medians=data.loc[dev,FEATURES].median();x=data[FEATURES].fillna(medians).to_numpy(np.float32)
    y=data.Log_Specific_Growth_Rate.to_numpy(np.float32);scaler=StandardScaler().fit(x[dev])
    selected=state["selected"];scaled=selected in ["OLS","MLP"]
    values=scaler.transform(x) if scaled else x
    model=make_model(selected,settings["models"][selected])
    print("Refitting validation-selected model on development:",selected,flush=True)
    model.fit(values[dev],y[dev]);pred=model.predict(values[test]);actual=y[test]
    metrics=metric_record(actual,pred,data.loc[test,"Initial_Carbon"].to_numpy(),data.loc[test,"Years"].to_numpy())
    thresholds=np.quantile(percentage(y[dev]),np.arange(1,7)/7)
    if not np.all(np.diff(thresholds)>0): raise ValueError("Development outcomes cannot form seven distinct thresholds")
    a,b=np.searchsorted(thresholds,percentage(actual))+1,np.searchsorted(thresholds,percentage(pred))+1
    slope,intercept=np.polyfit(pred,actual,1)
    matrix=confusion_matrix(a,b,labels=range(1,8))
    diagnostics=dict(calibration_intercept=float(intercept),calibration_slope=float(slope),exact_level_agreement=float(np.mean(a==b)),
                     within_one_level=float(np.mean(abs(a-b)<=1)),quadratic_weighted_kappa=float(cohen_kappa_score(a,b,weights="quadratic")))
    # Tree-cluster uncertainty; repeated periods travel together.
    groups=data.loc[test].reset_index(drop=True).groupby("OID_").indices
    indices=list(groups.values());rng=np.random.default_rng(settings["seed"]+1);draws=[]
    for _ in range(int(settings["bootstrap_repetitions"])):
        ii=np.concatenate([indices[j] for j in rng.integers(0,len(indices),len(indices))])
        draws.append(r2_score(actual[ii],pred[ii]))
    if draws: diagnostics["test_R2_tree_bootstrap_95_interval"]=np.quantile(draws,[.025,.975]).tolist()
    residuals=data.loc[test,["OID_","X","Y"]].copy();residuals["Residual"]=actual-pred
    tree_residuals=residuals.groupby("OID_",as_index=False).agg(X=("X","first"),Y=("Y","first"),Residual=("Residual","mean"))
    if len(tree_residuals)>10 and np.var(tree_residuals.Residual)>1e-20:
        diagnostics["residual_spatial_clustering"]=morans_i(tree_residuals[["X","Y"]].to_numpy(),tree_residuals.Residual.to_numpy(),int(settings["moran_permutations"]),settings["seed"]+2)
    final=folder/"finalized";final.mkdir();(final/"tables").mkdir();(final/"plots").mkdir()
    bundle=dict(model=model,feature_columns=FEATURES,scaler=scaler,use_scaled=scaled)
    joblib.dump(bundle,final/"model.joblib")
    joblib.dump(dict(feature_columns=FEATURES,feature_medians=medians,scaler=scaler,use_scaled=scaled,
                     model_format="trusted_joblib",model_sha256=sha256(final/"model.joblib"),supported_species=state["supported_species"]),final/"preprocessing.joblib")
    domain=[]
    for f in ["Log_Height",*ENVIRONMENT]:
        column=x[dev,FEATURES.index(f)]
        domain.append(dict(Feature=f,Minimum=float(column.min()),Maximum=float(column.max()),P01=float(np.quantile(column,.01)),P99=float(np.quantile(column,.99))))
    pd.DataFrame(domain).to_csv(final/"tables/domain.csv",index=False)
    pd.DataFrame(dict(Threshold_Number=range(1,7),Threshold=thresholds)).to_csv(final/"tables/thresholds.csv",index=False)
    pd.DataFrame([dict(Model=selected,**metrics)]).to_csv(final/"tables/locked_test_metrics.csv",index=False)
    pd.DataFrame(matrix,index=range(1,8),columns=range(1,8)).to_csv(final/"tables/level_confusion.csv")
    heldout=data.loc[test,["OID_","Period","X","Y","Species","Initial_Carbon","Years"]].copy()
    heldout["Actual_LogSGR"]=actual;heldout["Predicted_LogSGR"]=pred
    heldout["Actual_Annual_Percent"]=percentage(actual);heldout["Predicted_Annual_Percent"]=percentage(pred)
    heldout.to_csv(final/"tables/locked_test_predictions.csv",index=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout="constrained")
    hb=axes[0].hexbin(pred,actual,gridsize=40,mincnt=1,bins="log",cmap="viridis")
    fig.colorbar(hb,ax=axes[0],label="Count (log colour)")
    limits=[min(pred.min(),actual.min()),max(pred.max(),actual.max())]
    axes[0].plot(limits,limits,color="black",ls="--");axes[0].set(xlabel="Predicted log-SGR",ylabel="Observed log-SGR",title=f"Locked test | {selected} | R²={metrics['R2_LogSGR']:.3f}")
    proportions=matrix/np.maximum(matrix.sum(axis=1,keepdims=True),1)
    image=axes[1].imshow(proportions,vmin=0,vmax=1,cmap="Blues")
    axes[1].set(xticks=range(7),xticklabels=range(1,8),yticks=range(7),yticklabels=range(1,8),xlabel="Predicted level",ylabel="Observed level",title="Fixed-threshold agreement")
    fig.colorbar(image,ax=axes[1],label="Within observed-level proportion")
    fig.savefig(final/"plots/locked_test_validation.png",dpi=180);plt.close(fig)
    report=dict(status="finalized",selected_model=selected,metrics=metrics,diagnostics=diagnostics,
        threshold_method="Seven levels: fixed septiles of development outcomes, never map-specific quantiles",
        interpretation="Predictive validation, not validation of causal environmental contributions. Repeated tuning after opening test invalidates the locked-test claim.")
    write_json(final/"validation_report.json",report)
    write_json(final/"deployment.json",dict(model="model.joblib",preprocessing="preprocessing.joblib",domain="tables/domain.csv",thresholds="tables/thresholds.csv",crs=settings["crs"]))
    print("Finalized model and locked-test report:",final,flush=True)
    return report


def diagnose(config):
    from build_clickable_spatial_package import build
    args=dict(input=Path(config["input"]),output=Path(config["output"]),height=float(config.get("height",10)),
              park_context=int(config.get("park_context",1)),resolution=float(config.get("resolution",2)),crs=config.get("crs","EPSG:3879"),
              template_raster=Path(config["template_raster"]) if config.get("template_raster") else None,
              periods=["15_17","17_21","21_23"],earlier="15_17",later="21_23",chunk_size=100000,scope="wall_to_wall_input_grid")
    if config.get("training_run"):
        final=Path(config["training_run"])/"finalized"
        deployment=json.loads((final/"deployment.json").read_text())
        if args["crs"]!=deployment["crs"]: raise ValueError("Diagnosis CRS differs from training coordinates")
        args.update({key:final/deployment[key] for key in ["model","preprocessing","domain","thresholds"]})
    else:
        text_preprocessing=ROOT/"results/spatial_validation/deployment_preprocessing.json"
        args.update(model=ROOT/"models/xgb_spatial_deployment_no_period_three_soil.json",preprocessing=text_preprocessing if text_preprocessing.exists() else ROOT/"models/preprocessing_spatial_deployment.joblib",
                    domain=ROOT/"results/spatial_validation/development_training_domain.csv",thresholds=ROOT/"results/suitability/fixed_selected_seven_level_thresholds.csv")
    build(SimpleNamespace(**args))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("action",choices=["train","finalize","diagnose"])
    p.add_argument("--config",type=Path,help="JSON configuration for train or diagnose")
    p.add_argument("--run",type=Path,help="Existing training run to finalize")
    a=p.parse_args()
    if a.action=="finalize": finalize(a.run)
    else:
        config=json.loads(a.config.read_text(encoding="utf-8"))
        (train if a.action=="train" else diagnose)(config)
