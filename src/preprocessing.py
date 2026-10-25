"""Feature engineering, scaling, train/test split."""
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.config import (
    FEATURE_COLUMNS, TARGET_COLUMN, SEED, TEST_SIZE,
    SCALER_PATH, FEATURES_PATH,
)


def prepare_features(df: pd.DataFrame):
    df = df.copy()
    tld_dummies = pd.get_dummies(df["tld"], prefix="tld")
    X = pd.concat(
        [df[FEATURE_COLUMNS].reset_index(drop=True),
         tld_dummies.reset_index(drop=True)], axis=1)
    y = df[TARGET_COLUMN].astype(int)
    return X, y


def split_and_scale(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    SCALER_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(list(X.columns), FEATURES_PATH)
    print(f"[preprocessing] Train: {X_train_scaled.shape}, Test: {X_test_scaled.shape}")
    return X_train_scaled, X_test_scaled, y_train, y_test


def load_scaler_and_features():
    scaler = joblib.load(SCALER_PATH)
    feature_names = joblib.load(FEATURES_PATH)
    return scaler, feature_names
