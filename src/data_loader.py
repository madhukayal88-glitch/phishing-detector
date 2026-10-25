"""Load and validate the raw phishing CSV."""
import pandas as pd
from src.config import DATA_RAW, TARGET_COLUMN


def load_data(path=DATA_RAW) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found at {path}")
    df = pd.read_csv(path)
    print(f"[data_loader] Loaded {len(df)} rows x {len(df.columns)} cols")
    df = df.dropna()
    df = df.drop_duplicates(subset=["url"])
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Missing target column '{TARGET_COLUMN}'")
    print(f"[data_loader] Label distribution:\n{df[TARGET_COLUMN].value_counts()}")
    return df


if __name__ == "__main__":
    df = load_data()
    print(df.head())



