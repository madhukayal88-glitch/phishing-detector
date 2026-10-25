"""Add a few well-known legit URLs to the training data."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from src.features import extract_features

DST = Path("data/raw/phiusiil_features.csv")
df = pd.read_csv(DST)

legit_urls = [
    "https://github.com/torvalds/linux",
    "https://github.com/microsoft/vscode",
    "https://stackoverflow.com/questions/12345",
    "https://en.wikipedia.org/wiki/Phishing",
    "https://www.wikipedia.org/",
    "https://docs.python.org/3/",
    "https://developer.mozilla.org/en-US/",
    "https://www.reddit.com/r/programming/",
    "https://twitter.com/home",
    "https://www.linkedin.com/feed/",
    "https://www.amazon.com/dp/B08N5WRWNW",
    "https://www.netflix.com/browse",
    "https://www.apple.com/iphone/",
    "https://www.microsoft.com/en-us/",
    "https://mail.google.com/mail/u/0/",
    "https://drive.google.com/drive/my-drive",
    "https://www.paypal.com/us/home",
    "https://www.chase.com/",
    "https://www.bankofamerica.com/",
    "https://www.hdfcbank.com/",
    "https://www.icicibank.com/",
    "https://www.airbnb.com/",
    "https://www.booking.com/",
    "https://www.uber.com/",
    "https://www.imdb.com/",
    "https://www.espn.com/",
    "https://www.bbc.com/news",
    "https://www.cnn.com/",
    "https://www.nytimes.com/",
    "https://www.coursera.org/",
    "https://www.khanacademy.org/",
    "https://www.quora.com/",
    "https://medium.com/",
    "https://www.duolingo.com/",
    "https://www.spotify.com/",
]

new_rows = []
for url in legit_urls:
    f = extract_features(url)
    f["url"] = url
    f["label"] = 0   # 0 = legit
    new_rows.append(f)

combined = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
combined = combined.drop_duplicates(subset=["url"]).reset_index(drop=True)
combined.to_csv(DST, index=False)
print(f"Total rows: {len(combined)}")
print(f"Label counts:\n{combined['label'].value_counts()}")
