"""Convert the repository's trusted, unscaled XGBoost preprocessing to text JSON.

No model is refitted. Do not use this converter with untrusted joblib files.
"""
import hashlib
import json
from pathlib import Path
import joblib


def export(root):
    root = Path(root)
    original = root/"models/preprocessing_spatial_deployment.joblib"
    model = root/"models/xgb_spatial_deployment_no_period_three_soil.json"
    prep = joblib.load(original)
    if prep["use_scaled"]:
        raise ValueError("This export supports only the frozen unscaled XGBoost")
    result = dict(feature_columns=prep["feature_columns"],
        feature_medians={k:float(v) for k,v in prep["feature_medians"].items()},
        use_scaled=False, model_format="xgboost_json", supported_species=list(range(1,12)),
        model_digest=dict(algorithm="sha256", bytes=list(hashlib.sha256(model.read_bytes()).digest())),
        source_preprocessing_digest=dict(algorithm="sha256", bytes=list(hashlib.sha256(original.read_bytes()).digest())))
    # These are public file-integrity digests, not credentials. Explicit algorithm
    # and byte-array fields avoid mistaking opaque hexadecimal strings for keys.
    target=root/"results/spatial_validation/deployment_preprocessing.json"
    target.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(target)


if __name__ == "__main__":
    export(Path(__file__).resolve().parents[1])
