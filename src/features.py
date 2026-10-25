"""Feature extraction (no model dependency)."""
import math
import re
from collections import Counter
from urllib.parse import urlparse


def _entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = Counter(s)
    length = len(s)
    return -sum((c / length) * math.log2(c / length) for c in counts.values())


def extract_features(url: str) -> dict:
    parsed = urlparse(url if "://" in url else f"http://{url}")
    host = parsed.netloc or ""
    path = parsed.path or ""
    query = parsed.query or ""
    suspicious_terms = [
        "login", "secure", "account", "update", "verify", "bank",
        "wallet", "signin", "confirm", "password", "authenticate",
    ]
    
    # IP check
    is_ip = bool(re.match(r"^\d{1,3}(\.\d{1,3}){3}", host))
    
    # TLD extraction: IP-க்கு "ip" கொடு, இல்லைனா TLD
    if is_ip:
        tld = "ip"
    elif "." in host:
        tld = host.split(".")[-1]
    else:
        tld = "unknown"
    
    return {
        "url_length": len(url),
        "num_dots": url.count("."),
        "has_https": int(parsed.scheme == "https"),
        "has_ip": int(is_ip),
        "num_subdirs": len([p for p in path.split("/") if p]),
        "num_params": len(query.split("&")) if query else 0,
        "suspicious_words": sum(1 for t in suspicious_terms if t in url.lower()),
        "special_char_count": sum(1 for c in url if not c.isalnum()),
        "digits_count": sum(1 for c in url if c.isdigit()),
        "entropy": _entropy(url),
        "tld": tld,
    }
