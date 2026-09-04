"""Attach original predictor rasters to archived maps without replacing predictions.

Restores the July-28 no-soil workflow, its exact preprocessing/reference and fixed
thresholds. Original maps are copied, never relabelled as the newer journal model.
The recovered inputs are checked against sampled archived predictions before a
manifest is published. No training occurs.
"""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import shutil
import tempfile
import time

import joblib
import numpy as np
import pandas as pd
import rasterio

from build_clickable_spatial_package import profile, windows, set_descriptions, LEGACY
from spatial_waterfall_core import (ENVIRONMENT, GROUPS, SPECIES, NODATA, ModelPredictor,
                                    SpatialPackage, matrix_from_environment, percentage, sha256)


def required_input_columns():
    return sorted({"X","Y","noise","lightemiss"}|{c for mapping in LEGACY.values() for c in mapping.values()})


def clean_period_environment(chunk,period,prep,park):
    """Frozen July-28 cleaning; later exporters must not change this restoration."""
    sources={**LEGACY[period],"avg_noise_day":"noise","lightemiss":"lightemiss"}
    cleaned={};missing={}
    for feature in ENVIRONMENT[:7]:
        values=pd.to_numeric(chunk[sources[feature]],errors="coerce").to_numpy(float,copy=True)
        invalid=~np.isfinite(values)|(values<=-9990)
        if feature=="avg_noise_day":
            values[invalid]=40;invalid[:]=False
        if feature=="lightemiss": invalid|=values<0
        missing[feature]=invalid
        values[invalid]=float(prep["no_period_medians"][feature])
        cleaned[feature]=values.astype("float32")
    cleaned["type_Puisto"]=np.full(len(chunk),park,np.float32)
    missing["type_Puisto"]=np.zeros(len(chunk),bool)
    return cleaned,missing


def restore(archive, input_csv, model_dir, output, chunk_size=200000):
    archive,input_csv,model_dir,output = map(Path,[archive,input_csv,model_dir,output])
    if output.exists():
        raise FileExistsError("Choose a new output folder; existing maps are never changed")
    started = time.perf_counter()
    old = json.loads((archive/"run_metadata.json").read_text())
    prep = joblib.load(model_dir/"preprocessing.joblib")
    predictor = ModelPredictor(model_dir/"xgb_no_period.json")
    env_names = ENVIRONMENT[:8]
    if prep["environment_features"] != env_names:
        raise ValueError("This adapter is exclusively for the archived eight-input model")
    reference = {f:float(prep["no_period_medians"][f]) for f in env_names}
    height = float(old["arguments"]["reference_height_m"])
    park = float(old["arguments"]["park_context"])
    reference["type_Puisto"] = park
    refs = pd.read_csv(archive/"reference_predictions.csv").set_index("Species_Code")
    predicted_refs = percentage(predictor.inplace_predict(np.concatenate([
        matrix_from_environment(predictor,list(reference.values()),height,s) for s in SPECIES])))
    np.testing.assert_allclose(predicted_refs,refs.loc[list(SPECIES),"Reference_Annual_Growth_Percent"],atol=2e-5,rtol=2e-6)
    with rasterio.open(archive/"relative_growth_pct_21_23.tif") as template:
        width,h,transform,crs = template.width,template.height,template.transform,template.crs
        labels = dict(zip(SPECIES,template.descriptions))
    domain_table = pd.read_csv(archive/"training_domain.csv").set_index("Feature")
    domain = {f:{out:float(domain_table.loc[f,src]) for out,src in
                 [("Minimum","Min"),("Maximum","Max"),("P01","P01"),("P99","P99")]}
              for f in ["Log_Height",*env_names]}
    output.mkdir(parents=True)
    (output/"rasters").mkdir()
    (output/"provenance").mkdir()
    for name in ["run_metadata.json","grid_metadata.json","reference_predictions.csv","training_domain.csv",
                 "suitability_thresholds.csv","OUTPUT_GUIDE.md","VALIDATION.json","reliability_summary.csv"]:
        shutil.copy2(archive/name,output/"provenance"/name)
    shutil.copy2(model_dir/"xgb_no_period.json",output/"model.json")
    periods = {}
    for p in ["15_17","17_21","21_23"]:
        record = {kind:f"rasters/{prefix}_{p}.tif" for kind,prefix in
                  [("growth","relative_growth_pct"),("deviation","environmental_deviation_pp"),("suitability","suitability_level")]}
        for relative in record.values():
            print("Copying archived raster:",Path(relative).name,flush=True)
            shutil.copy2(archive/Path(relative).name,output/relative)
        record.update(environment=f"rasters/environment_{p}.tif",reliability=f"rasters/reliability_{p}.tif",
                      within_p01_p99=f"rasters/within_p01_p99_{p}.tif",imputed_inputs=f"rasters/imputed_inputs_{p}.tif")
        periods[p] = record
    change = dict(earlier="15_17",later="21_23",growth_change="rasters/relative_growth_change_pp_2015_2023.tif",
                  suitability_change="rasters/suitability_level_change_2015_2023.tif",reliability="rasters/change_reliability.tif")
    for key in ["growth_change","suitability_change"]:
        shutil.copy2(archive/Path(change[key]).name,output/change[key])
    stats = {}; samples=[]; count=0
    with tempfile.TemporaryDirectory(dir=output,prefix="inputs_") as temp,ExitStack() as stack:
        envs={}; flags={}
        for p in periods:
            envs[p]=np.memmap(Path(temp)/f"env_{p}",mode="w+",dtype="float32",shape=(8,h,width))
            flags[p]=np.memmap(Path(temp)/f"missing_{p}",mode="w+",dtype="uint8",shape=(8,h,width))
            for data in [envs[p],flags[p]]: stack.callback(data._mmap.close)
            envs[p][:]=NODATA; flags[p][:]=255
            stats[p]=dict(populated=0,imputed_cells=0)
        seen=np.memmap(Path(temp)/"seen",mode="w+",dtype="uint8",shape=(h,width))
        stack.callback(seen._mmap.close); seen[:]=0
        for chunk in pd.read_csv(input_csv,usecols=required_input_columns(),chunksize=chunk_size):
            cf=(chunk.X.to_numpy()-transform.c)/transform.a-0.5
            rf=(chunk.Y.to_numpy()-transform.f)/transform.e-0.5
            if not np.isfinite([cf,rf]).all(): raise ValueError("Non-finite grid coordinates")
            cols,rows=np.rint(cf).astype(int),np.rint(rf).astype(int)
            if np.any(np.maximum(abs(cf-cols),abs(rf-rows))>1e-4): raise ValueError("Grid is not aligned")
            if np.any((rows<0)|(rows>=h)|(cols<0)|(cols>=width)): raise ValueError("Source outside archived grid")
            if len(set(zip(rows,cols)))!=len(rows) or np.any(seen[rows,cols]): raise ValueError("Duplicate grid cell")
            seen[rows,cols]=1
            for p in periods:
                cleaned,missing=clean_period_environment(chunk,p,prep,park)
                for j,f in enumerate(env_names):
                    envs[p][j,rows,cols]=cleaned[f]; flags[p][j,rows,cols]=missing[f]
                stats[p]["populated"]+=len(chunk)
                stats[p]["imputed_cells"]+=int(np.any(np.stack(list(missing.values())),axis=0).sum())
            for i in np.linspace(0,len(chunk)-1,min(4,len(chunk)),dtype=int):
                samples.append((float(chunk.X.iloc[i]),float(chunk.Y.iloc[i])))
            count+=len(chunk)
            print(f"Recovered original inputs: {count:,} cells",flush=True)
        if count != old["run"]["rows_processed"]: raise ValueError("Source row count differs from archived run")
        for p,record in periods.items():
            with rasterio.open(archive/f"reliability_{p}.tif") as oldrel, ExitStack() as files:
                outputs={}
                for kind,n,dtype,nodata in [("environment",8,"float32",NODATA),("imputed_inputs",8,"uint8",255),
                                            ("reliability",1,"uint8",0),("within_p01_p99",1,"uint8",0)]:
                    outputs[kind]=files.enter_context(rasterio.open(output/record[kind],"w",**profile(width,h,transform,crs,n,dtype,nodata)))
                    set_descriptions(outputs[kind],env_names if n==8 else [kind],"source_inputs" if n==8 else "code")
                for win in windows(width,h):
                    r,c,hh,ww=map(int,[win.row_off,win.col_off,win.height,win.width])
                    source=seen[r:r+hh,c:c+ww]>0
                    archived=oldrel.read(window=win)
                    if not np.array_equal(source,archived[0]!=255): raise ValueError("Source coverage differs from archived map")
                    missing=np.array(flags[p][:,r:r+hh,c:c+ww])
                    code=np.where(~source|np.any(missing==1,axis=0),0,np.where(archived[0]==1,1,2)).astype("uint8")
                    outputs["environment"].write(np.array(envs[p][:,r:r+hh,c:c+ww]),window=win)
                    outputs["imputed_inputs"].write(missing,window=win)
                    outputs["reliability"].write(code,1,window=win)
                    outputs["within_p01_p99"].write(np.where(source,archived[1],0).astype("uint8"),1,window=win)
            print("Recovered raster inputs:",p,stats[p],flush=True)
    with rasterio.open(output/periods["15_17"]["reliability"]) as a,rasterio.open(output/periods["21_23"]["reliability"]) as b, \
         rasterio.open(output/change["reliability"],"w",**profile(width,h,transform,crs,1,"uint8",0)) as dst:
        for win in windows(width,h):
            x,y=a.read(1,window=win),b.read(1,window=win)
            dst.write(np.where((x>0)&(y>0),np.maximum(x,y),0).astype("uint8"),1,window=win)
    meta=dict(schema_version=1,scope="wall_to_wall_input_grid",scope_note="Full archived diagnosis area; original July-28 predictions and thresholds retained.",
              crs=crs.to_string(),grid=dict(width=width,height=h,transform=list(transform)[:6]),
              model=dict(file="model.json",sha256=sha256(output/"model.json"),feature_names=predictor.feature_names),
              model_vintage_note="Archived July-28 model: eight environmental inputs; no soil indicators. Not the final three-soil model.",
              environment_features=env_names,groups=GROUPS[:8],species=SPECIES,species_band_labels=labels,
              reference_height_m=height,reference_environment=reference,reference_growth_percent=dict(zip(SPECIES,map(float,predicted_refs))),
              thresholds_annual_growth_percent=old["suitability_thresholds"],domain=domain,periods=periods,change=change,
              statistics=stats,source_input_sha256=sha256(input_csv),input_rows=count,included_cells=count,
              missing_policy="Archived preprocessing retained: noise missing/sentinel=40 dB; other missing inputs median-imputed, flagged reliability 0 and warned on click.")
    candidate=output/"manifest.pending.json"
    candidate.write_text(json.dumps(meta,indent=2,allow_nan=False),encoding="utf-8")
    package=SpatialPackage(candidate); checks=[]
    for x,y in samples:
        for species in [2,3,10]:
            for mode,p in [("local","15_17"),("local","17_21"),("local","21_23"),("change","21_23")]:
                result=package.explain(x,y,species,mode,p)
                if result["status"]!="ok": raise AssertionError(result)
                checks.append(result["max_raster_parity_error_pp"])
    report=dict(passed=True,input_cells=count,sampled_explanations=len(checks),
                max_raster_parity_error_pp=max(checks),elapsed_seconds=time.perf_counter()-started,
                archived_predictions_copied_unchanged=True,model_retrained=False,thresholds_changed=False)
    (output/"RESTORATION_VALIDATION.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    candidate.rename(output/"manifest.json")
    (output/"RUN_LOG.md").write_text("# Restored full-area diagnosis\n\nOriginal predictions, species bands and fixed thresholds copied unchanged. "
        "Original eight-input model and source CSV restored for reference-matched grouped Shapley explanations. "
        "Source missing-input masks are retained. This is not a new model or an improvement in predictive accuracy.\n",encoding="utf-8")
    print(json.dumps(report,indent=2),flush=True)
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ["archive","input","model-dir","output"]: parser.add_argument("--"+name,type=Path,required=True)
    a=parser.parse_args(); restore(a.archive,a.input,a.model_dir,a.output)
