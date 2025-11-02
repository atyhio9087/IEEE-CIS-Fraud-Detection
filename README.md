# AI-Driven Fraud Detection & Insights (IEEE-CIS)

End-to-end fraud risk scoring app (CPU-friendly). Upload transactions (+ optional identity), get risk scores, KPIs, histogram, and export flagged rows.

## Live Demo
- Open Streamlit app: https://ieee-cis-fraud-detection-3pwswxjxlgvrm4cxu45iow.streamlit.app/
- Source code: this repo

## Features
- Consistent feature engineering & categorical encoding (from training artifacts)
- Risk scoring with threshold selector
- KPIs, score distribution chart
- Export flagged transactions
- Optional permutation importance (interpretability)
- Handles .csv and .csv.gz (large file friendly)
- Upload limit set to 1 GB

## Tech
Python, pandas, scikit-learn (HistGradientBoosting), Plotly, Streamlit, joblib

## Quickstart (local)
```bash
pip install -r requirements.txt
streamlit run app.py
