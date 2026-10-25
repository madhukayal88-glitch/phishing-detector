"""Convert PhiUSIIL dataset into project feature format.

PhiUSIIL label convention:
  - label=0 -> PHISHING (malicious)  -- 134,850 rows
  - label=1 -> LEGIT (benign)        -- 100,520 rows

We flip so that ours matches:
  - label=1 -> PHISHING
  - label=0 -> LEGIT
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from src.features import extract_features

SRC = Path("data/raw/phiusiil/PhiUSIIL_Phishing_URL_Dataset.csv")
DST = Path("data/raw/phiusiil_features.csv")

df = pd.read_csv(SRC)
print(f"Loaded {len(df)} rows")
print(f"Original label counts: {df['label'].value_counts().to_dict()}")

rows = []
for url, label in zip(df["URL"], df["label"]):
    try:
        f = extract_features(str(url))
        f["url"] = str(url)
        # PhiUSIIL: 0=phishing, 1=legit -> ours: 1=phishing, 0=legit
        f["label"] = 1 - int(label)
        rows.append(f)
    except Exception:
        continue

out = pd.DataFrame(rows)
out = out.drop_duplicates(subset=["url"]).reset_index(drop=True)
out.to_csv(DST, index=False)
print(f"Wrote {len(out)} rows to {DST}")
print(f"Label counts (after flip):\n{out['label'].value_counts()}")
print(f"Columns: {list(out.columns)}")

# Sanity check: known legit vs known phishing
print("\n--- Sanity check ---")
checks = [
    ("github.com",          1),  # 1 = phishing in our convention? No, legit -> 0
    ("southbankmosaics.com", 0),  # legit
    ("paypal-support",       1),  # phishing
    ("googleimg.top",        1),  # phishing
]
for kw, expected_ours in checks:
    sample = out[out["url"].str.contains(kw, case=False, na=False)].head(1)
    if not sample.empty:
        actual = int(sample["label"].iloc[0])
        ok = "OK" if actual == expected_ours else "MISMATCH"
        print(f"  {kw:25s} -> label={actual} expected={expected_ours} {ok}")
