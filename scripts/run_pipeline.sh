#!/usr/bin/env bash
set -e
echo "==> Training model"
python -m src.train
echo "==> Running tests"
pytest tests/ -v
echo "==> Launching app"
streamlit run app/streamlit_app.py
