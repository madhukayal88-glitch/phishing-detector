import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.predict import predict_url

urls = [
    "http://paypal-secure-verify.xyz/login",
    "https://www.google.com/",
    "https://github.com/torvalds/linux",
    "https://www.sbi.co.in/",
    "http://apple-id-confirm.tk/signin",
    "http://192.168.1.1/admin",
]

for u in urls:
    r = predict_url(u)
    print(f"{r['label']:9s} | {r['confidence']*100:5.1f}% | {u}")
