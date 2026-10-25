"""Streamlit UI - run: streamlit run app/streamlit_app.py"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import streamlit as st
from src.predict import predict_url

st.set_page_config(page_title="Phishing URL Detector", layout="centered")
st.title("🛡️ Phishing URL Detector")
st.caption("Paste a URL — the model will classify it as Phishing or Legit.")

url = st.text_input("Enter URL", placeholder="https://example.com/login")

if st.button("Analyze", type="primary"):
    if not url.strip():
        st.warning("⚠️ Please enter a URL first.")
    else:
        with st.spinner("Analyzing..."):
            try:
                result = predict_url(url.strip())
            except Exception as e:
                st.error(f"Error: {type(e).__name__}: {e}")
                st.exception(e)
                st.stop()

        if result["label"] == "Phishing":
            st.error(f"🚨 **PHISHING** — confidence {result['confidence']:.1%}")
        else:
            st.success(f"✅ **LEGIT** — confidence {(1 - result['confidence']):.1%}")

        with st.expander("Feature breakdown", expanded=True):
            st.json(result["features"])

st.divider()
st.caption("Feature-based RandomForest classifier trained on PhiUSIIL dataset.")
