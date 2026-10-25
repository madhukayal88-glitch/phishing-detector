from src.preprocessing import prepare_features
from src.data_loader import load_data


def test_prepare_features_runs():
    df = load_data()
    X, y = prepare_features(df)
    assert len(X) == len(y) > 0
    assert "url_length" in X.columns
