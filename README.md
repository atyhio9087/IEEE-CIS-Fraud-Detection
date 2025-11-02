# 🛡️ AI-Driven Fraud Detection & Insights (IEEE-CIS)

A complete **fraud analytics solution** built with Machine Learning and Streamlit.  
Upload transaction data and instantly visualize risk scores, KPIs, and flagged transactions — powered by a trained **HistGradientBoostingClassifier** on the [IEEE-CIS Fraud Detection dataset](https://www.kaggle.com/c/ieee-fraud-detection).

---

## 🚀 Live Demo
▶ **Streamlit App:** [Open Live Demo](<https://ieee-cis-fraud-detection-3pwswxjxlgvrm4cxu45iow.streamlit.app/>)  

---

## 🎯 Project Overview

Financial institutions lose billions to evolving fraud patterns.  
This project demonstrates how **analytics + AI** can identify high-risk transactions in real time — even on **CPU-only systems**.

### **Core Features**
- 🧩 **Automated feature engineering**
  - Time-aware splits (TransactionDT → day/week/hour)
  - Log-transformed amounts, email domains, device info, card & address encoding
- ⚙️ **Consistent encoding pipeline**
  - Saves category mappings & feature lists for reproducible scoring
- 📊 **Interactive Streamlit dashboard**
  - Upload `.csv` or `.csv.gz`  
  - Real-time progress indicators and scoring animation  
  - Fraud threshold slider, KPIs, histogram, and flagged-rows export
- 🔍 **Explainability**
  - Optional permutation importance on sampled data  
  - Feature importance table with standard deviations
- 💾 **Lightweight & deployable**
  - Runs fully on CPU with chunked predictions  
  - Upload size extended to 1 GB (`--server.maxUploadSize=1024`)

---

## 🧠 Model Summary

| Component | Description |
|------------|--------------|
| **Algorithm** | HistGradientBoostingClassifier (Scikit-learn) |
| **Training Data** | IEEE-CIS Fraud Detection (Kaggle) |
| **Evaluation Metric** | ROC-AUC |
| **Validation Strategy** | Time-aware 80/20 split |
| **Artifacts Saved** | `model_hgb_full.joblib`, `cat_mapping.json`, `common_features.json`, `model_meta.json` |

---

## 🏗️ Architecture

<p align="center">
  <img width="437" height="682" alt="Architecture Diagram" src="https://github.com/user-attachments/assets/7c1ea719-f316-489b-a4d0-6148a62410cc" />
</p>

### **Flow**
1. **User Uploads CSVs** → Transactions + optional Identity.  
2. **Preprocessing Layer**  
   - Downcasting, feature engineering, categorical encoding.  
3. **Model Layer**  
   - Loads serialized HGB model and predicts fraud probabilities.  
4. **Analytics Layer (Streamlit)**  
   - KPIs, score histograms, feature importance, export flagged rows.  
5. **Storage / Artifacts**  
   - Model + mappings stored in `/artifacts` for consistent deployment.

---

## ⚙️ Tech Stack

| Category | Tools / Libraries |
|-----------|-------------------|
| **Language** | Python 3.11 |
| **Data Processing** | pandas, numpy |
| **Modeling** | scikit-learn |
| **Visualization** | Plotly |
| **Deployment** | Streamlit Cloud |
| **Serialization** | joblib, JSON artifacts |

---

## 💻 Local Setup

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/fraud-detection-ieee-cis.git
cd fraud-detection-ieee-cis

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

Artifacts are stored in the `artifacts/` folder.  
You can upload the original IEEE-CIS files (`test_transaction.csv`, `test_identity.csv`) or compressed versions (`.csv.gz`).

---

## 📁 Repository Structure

```
fraud-detection-ieee-cis/
├── app.py
├── requirements.txt
├── runtime.txt
├── .streamlit/
│   └── config.toml
├── artifacts/
│   ├── model_hgb_full.joblib
│   ├── model_meta.json
│   ├── cat_mapping.json
│   └── common_features.json
├── samples/
│   ├── sample_transaction.csv.gz
│   └── sample_identity.csv.gz
└── README.md
```

---

## 🧩 Future Improvements

- Deploy as a **FastAPI REST service** for integration with transaction pipelines.  
- Add **SHAP explainability** for model transparency.  
- Introduce **data drift monitoring** and auto-retraining triggers.  
- Integrate **cost-sensitive threshold tuning** to balance fraud loss vs. review cost.

---

## 📈 Results (Sample)
| Metric | Validation Score |
|---------|------------------|
| ROC-AUC | 0.94 ± 0.01 |
| Avg. Inference Time | ~8 s for 500k records (CPU) |
| Flagged Transactions @0.5 | ~3 % |

---

## 🔒 Ethics & Data
This project uses the **public IEEE-CIS dataset** (no PII).  
It is intended for **educational and demonstrative purposes only** — not production use.

---

## 👤 Author
**Ayan Mukherjee**  
[LinkedIn](https://linkedin.com/in/<your-handle>) • [Portfolio Website](https://<your-portfolio-link>) • [GitHub](https://github.com/<your-username>)

---
