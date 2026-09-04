"""Make a clearly labelled sparse observed-cell demo, not a wall-to-wall map."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",type=Path,required=True)
    p.add_argument("--spatial-split",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--center-oid",type=int,default=23718)
    p.add_argument("--radius-m",type=float,default=400)
    p.add_argument("--resolution",type=float,default=2)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    raw = pd.read_csv(args.input)
    split = pd.read_csv(args.spatial_split)
    origin = raw.loc[raw.OID_ == args.center_oid].iloc[0]
    use = (raw.X-origin.X).abs().le(args.radius_m) & (raw.Y-origin.Y).abs().le(args.radius_m)
    use &= raw.OID_.isin(split.loc[split.Spatial_Split == "Test","OID_"])
    selected = raw.loc[use].sort_values("OID_").copy()
    selected["X"] = np.floor(selected.X/args.resolution)*args.resolution+args.resolution/2
    selected["Y"] = np.floor(selected.Y/args.resolution)*args.resolution+args.resolution/2
    before = len(selected)
    selected = selected.drop_duplicates(["X","Y"],keep="first")
    result = selected[["OID_","X","Y"]].copy()
    for feature in ["Density25","Mono_Rate","lightemiss"]:
        result[feature] = selected[feature]
    result["avg_noise_day"] = selected[["noise17d","noise22d"]].mean(axis=1)
    result["type_Puisto"] = selected.type.eq("Puisto").astype(float)
    soil = selected.soil.fillna("").astype(str).str.strip().str.casefold().str.replace(" ","",regex=False)
    result["soil_infill"] = soil.str.startswith("t").astype(float)
    result["soil_bedrock"] = soil.str.contains("ka",regex=False).astype(float)
    result["soil_moraine"] = soil.str.contains("mr",regex=False).astype(float)
    # Missing soil labels remain missing rather than being guessed as absence.
    result.loc[soil == "",["soil_infill","soil_bedrock","soil_moraine"]] = np.nan
    for period,start,end,lst in [("15_17","15","17","LST_1516"),("17_21","17","21","LST_1720"),("21_23","21","23","LST_2122")]:
        result[f"avg_svf_{period}"] = selected[["svf"+start,"svf"+end]].mean(axis=1)
        result[f"avg_LST_{period}"] = selected[lst]
        result[f"avg_radiation_{period}"] = selected["ra"+period]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    result.to_csv(args.output,index=False)
    note = dict(scope="observed_cells_demo",observed_rows=len(result),duplicate_cells_dropped=before-len(result),
                spatial_partition="Test",center_oid=args.center_oid,radius_m=args.radius_m,
                coordinate_rule="Observed locations snapped to containing cell; smallest OID retained on collision.",
                temporal_note="SVF, radiation and LST vary by period. Other supplied demo environmental fields are shared across periods; no noise/density change is fabricated.",
                target_note="No target fitting or accuracy claims. Standardized 10m species scenarios use observed environmental inputs only.")
    args.output.with_suffix(".provenance.json").write_text(json.dumps(note,indent=2),encoding="utf-8")
    print(json.dumps(note,indent=2))


if __name__ == "__main__":
    main()
