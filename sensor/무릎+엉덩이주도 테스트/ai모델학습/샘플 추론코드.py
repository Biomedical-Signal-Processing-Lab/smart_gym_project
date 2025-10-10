# infer_rep_classifier.py
import json, joblib, pandas as pd
from pathlib import Path

ART = Path("artifacts")
model = joblib.load(ART / "best_model.joblib")
FEATURES = json.loads((ART / "feature_order.json").read_text(encoding="utf-8"))

def predict_rep(feature_row: dict):
    # feature_row: {"tempo_desc_ms":..., "tempo_asc_ms":..., ...}
    df = pd.DataFrame([feature_row], columns=FEATURES)
    proba = None
    if hasattr(model.named_steps["clf"], "predict_proba"):
        proba = model.predict_proba(df)[0]
        classes = model.named_steps["clf"].classes_.tolist()
    y_hat = model.predict(df)[0]
    return y_hat, (classes, proba) if proba is not None else None

# 예시
if __name__ == "__main__":
    sample = {
        "tempo_desc_ms": 820, "tempo_asc_ms": 710, "rom_deg": 88.0,
        "env_int_desc": 1.25, "env_int_asc": 1.05,
        "env_peak_desc": 0.62, "env_peak_asc": 0.55,
        "env_init_mean": 0.21
    }
    y, proba = predict_rep(sample)
    print("Pred:", y)
    if proba:
        print("Proba:", dict(zip(proba[0], proba[1])))
