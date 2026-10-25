"""Train the phishing classifier."""
import joblib
from sklearn.ensemble import RandomForestClassifier
from src.config import MODEL_PATH, RF_PARAMS
from src.data_loader import load_data
from src.preprocessing import prepare_features, split_and_scale
from src.evaluate import evaluate_model


def train():
    print("=" * 60)
    print("TRAINING PHISHING URL CLASSIFIER")
    print("=" * 60)
    df = load_data()
    X, y = prepare_features(df)
    X_train, X_test, y_train, y_test = split_and_scale(X, y)
    model = RandomForestClassifier(**RF_PARAMS)
    print(f"[train] Fitting {model.n_estimators} trees...")
    model.fit(X_train, y_train)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"[train] Model saved to {MODEL_PATH}")
    evaluate_model(model, X_test, y_test)
    return model


if __name__ == "__main__":
    train()
