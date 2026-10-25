from src.predict import extract_features


def test_extract_features_keys():
    feats = extract_features("https://example.com/login?user=1")
    for k in ["url_length", "num_dots", "has_https", "entropy", "tld"]:
        assert k in feats


def test_has_ip_detection():
    feats = extract_features("http://192.168.1.1/admin")
    assert feats["has_ip"] == 1
