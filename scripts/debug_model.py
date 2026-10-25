import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import pandas as pd
from src.config import MODEL_PATH
from src.features import extract_features
from src.preprocessing import load_scaler_and_features

model = joblib.load(MODEL_PATH)
scaler, feature_names = load_scaler_and_features()

urls = [
    "http://paypal-secure-verify.xyz/login",
    "https://www.google.com/",
    "https://github.com/torvalds/linux",
    "http://192.168.1.1/admin",
    "http://apple-id-confirm.tk/signin",
    "https://www.sbi.co.in/",
]

for url in urls:
    feats = extract_features(url)
    row = {k: v for k, v in feats.items() if k != "tld"}
    for name in feature_names:
        if name.startswith("tld_"):
            row[name] = int(name == f"tld_{feats['tld']}")
    X = pd.DataFrame([row]).reindex(columns=feature_names, fill_value=0)
    X_scaled = scaler.transform(X)
    proba = model.predict_proba(X_scaled)[0, 1]
    label = "PHISHING" if proba >= 0.5 else "LEGIT"
    print(f"{label:9s} | proba={proba:.4f} | {url}")
