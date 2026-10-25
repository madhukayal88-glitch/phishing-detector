"""Inference - used by Streamlit app."""
import joblib
import pandas as pd
from src.config import MODEL_PATH
from src.features import extract_features
from src.preprocessing import load_scaler_and_features


def predict_url(url: str) -> dict:
    model = joblib.load(MODEL_PATH)
    scaler, feature_names = load_scaler_and_features()
    feats = extract_features(url)

    row = {k: v for k, v in feats.items() if k != "tld"}
    for name in feature_names:
        if name.startswith("tld_"):
            row[name] = int(name == f"tld_{feats['tld']}")
    X = pd.DataFrame([row]).reindex(columns=feature_names, fill_value=0)
    X_scaled = scaler.transform(X)
    proba = model.predict_proba(X_scaled)[0, 1]
    return {
        "label": "Phishing" if proba >= 0.5 else "Legit",
        "confidence": float(proba),
        "features": feats,
    }
