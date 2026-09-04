"""Audit raster alignment, missing masks, classes and sampled waterfall closure."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import rasterio
from spatial_waterfall_core import SpatialPackage, NODATA


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--package",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--sample-cells",type=int,default=10)
    p.add_argument("--species",type=int,nargs="+",default=[2,3,10])
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    started = time.perf_counter()
    package = SpatialPackage(args.package)
    meta = package.meta
    shape, transform, crs = None, None, None
    rasters_checked, valid = 0, {}
    for period, record in meta["periods"].items():
        for kind,path in record.items():
            with rasterio.open(package.file(path)) as src:
                current = (src.height,src.width)
                if shape is None:
                    shape,transform,crs = current,src.transform,src.crs
                assert (current,src.transform,src.crs) == (shape,transform,crs), "Misaligned raster"
                rasters_checked += 1
        with rasterio.open(package.file(record["reliability"])) as src:
            mask = src.read(1)
            assert np.isin(mask,[0,1,2]).all()
            valid[period] = mask > 0
        with rasterio.open(package.file(record["growth"])) as growth, rasterio.open(package.file(record["suitability"])) as suitability:
            for species in range(1,12):
                g,s = growth.read(species),suitability.read(species)
                assert np.all(g[~valid[period]] == NODATA)
                assert np.all(s[~valid[period]] == 0)
                expected = np.searchsorted(package.thresholds,g[valid[period]],side="right")+1
                # Stored float32 growth may round at a threshold: flag rather than silently change classes.
                assert np.all(s[valid[period]] == expected), "Suitability thresholds do not reproduce stored classes"
    records = []
    rng = np.random.default_rng(2026)
    modes = [("local",period,mask) for period,mask in valid.items()]
    change = meta["change"]
    pair_valid = valid[change["earlier"]]&valid[change["later"]]
    for key in ["growth_change","suitability_change","reliability"]:
        with rasterio.open(package.file(change[key])) as src:
            package.check_grid(src)
            rasters_checked += 1
    early,late = meta["periods"][change["earlier"]],meta["periods"][change["later"]]
    for input_key,output_key,nodata in [("growth","growth_change",NODATA),("suitability","suitability_change",-128)]:
        with rasterio.open(package.file(early[input_key])) as a,rasterio.open(package.file(late[input_key])) as b,rasterio.open(package.file(change[output_key])) as out:
            for species in range(1,12):
                difference = b.read(species).astype(float)-a.read(species).astype(float)
                actual = out.read(species)
                assert np.all(actual[~pair_valid] == nodata)
                np.testing.assert_allclose(actual[pair_valid],difference[pair_valid],rtol=1e-6,atol=1e-6)
    modes += [("change",change["later"],valid[change["earlier"]]&valid[change["later"]])]
    for mode,period,mask in modes:
        positions = np.argwhere(mask)
        chosen = rng.choice(len(positions),min(len(positions),args.sample_cells),replace=False)
        for position in chosen:
            row,col = positions[position]
            x,y = rasterio.transform.xy(transform,row,col)
            for species in args.species:
                result = package.explain(x,y,species,mode,period)
                assert result["status"] == "ok"
                assert abs(sum(result["contributions_pp"])-result["delta_pp"]) < 1e-8
                records.append({k:result[k] for k in ["mode","period","species_code","row","column","delta_pp","additivity_error_pp","max_raster_parity_error_pp"]})
    assert records, "No valid explanations tested"
    report = dict(passed=True,rasters_checked=rasters_checked,explanations_checked=len(records),
                  valid_cells_by_period={k:int(v.sum()) for k,v in valid.items()},
                  max_additivity_error_pp=max(abs(r["additivity_error_pp"]) for r in records),
                  max_raster_parity_error_pp=max(r["max_raster_parity_error_pp"] for r in records),
                  elapsed_seconds=time.perf_counter()-started,checks=records)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k != "checks"},indent=2))


if __name__ == "__main__":
    main()
