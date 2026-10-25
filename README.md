Phishing URL Detector
=====================

A machine learning model that classifies URLs as Phishing or Legit.

Model
-----
- Algorithm: RandomForest (300 trees, max_depth=20)
- Dataset: PhiUSIIL Phishing URL Dataset (235,405 rows, 57:43 balanced)
- Features: URL length, dots, https, IP, subdirs, params, suspicious words, special chars, digits, entropy, TLD (one-hot)

Performance (test set, 47,081 samples)
--------------------------------------
- Accuracy: 98.72%
- Precision: 98.97%
- Recall: 98.02%
- F1: 98.49%
- ROC-AUC: 99.78%

Quick Start
-----------
1. python -m venv .venv
2. .venv\Scripts\Activate.ps1
3. python -m pip install -r requirements.txt
4. python scripts\prepare_phiusiil.py
5. python -m src.train
6. streamlit run app/streamlit_app.py

Test
----
- python -m pytest tests\ -v
- python scripts\test_predictions.py

Sample Predictions
------------------
- http://paypal-secure-verify.xyz/login -> PHISHING (98.8%)
- https://www.google.com/                -> LEGIT (77.6%)
- http://apple-id-confirm.tk/signin      -> PHISHING (97.9%)
- https://www.sbi.co.in/                 -> LEGIT (62.4%)

Known Limitations
-----------------
- Model trained on PhiUSIIL dataset. Legit URLs not in training data
  (e.g. github.com/torvalds/linux) may be classified as phishing.
- Private IP URLs (192.168.x.x) classified as phishing due to dataset bias.
