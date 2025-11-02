# app.py
import os
import json
import time
import gc
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st
import joblib
from sklearn.metrics import roc_auc_score
from sklearn.inspection import permutation_importance
import plotly.express as px
from sklearn.utils.validation import check_is_fitted

# =========================
# Page config
# =========================
st.set_page_config(
    page_title="Fraud Risk Scoring (IEEE-CIS)",
    page_icon="🛡️",
    layout="wide"
)

# =========================
# Globals / paths
# =========================
ART_DIR = "./artifacts"
MODEL_PATH = os.path.join(ART_DIR, "model_hgb_full.joblib")
META_PATH  = os.path.join(ART_DIR, "model_meta.json")
CAT_PATH   = os.path.join(ART_DIR, "cat_mapping.json")
CF_PATH    = os.path.join(ART_DIR, "common_features.json")
START_DATE = pd.to_datetime("2017-12-01")

# =========================
# Utilities
# =========================
def exists_all(paths: list[str]) -> bool:
    return all(os.path.exists(p) for p in paths)

@st.cache_resource(show_spinner=False)
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    # sanity: ensure fitted
    try:
        check_is_fitted(model)
    except Exception:
        pass
    with open(META_PATH, "r") as f:
        meta = json.load(f)
    with open(CAT_PATH, "r") as f:
        cat_map = json.load(f)
    with open(CF_PATH, "r") as f:
        cf = json.load(f)
    common_features = cf["features"]
    na_token = cat_map.get("NA_TOKEN", "__NA__")
    cat_columns = list(cat_map["columns"].keys())
    cat_categories = cat_map["columns"]
    return model, meta, na_token, cat_columns, cat_categories, common_features

def downcast_df(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        t = df[col].dtype
        if t == object:
            df[col] = df[col].astype("category")
        elif str(t).startswith("int"):
            df[col] = pd.to_numeric(df[col], downcast="integer")
        elif str(t).startswith("float"):
            df[col] = pd.to_numeric(df[col], downcast="float")
    return df

def map_email(x):
    if pd.isna(x): return "unknown"
    x = str(x).lower()
    if "gmail" in x: return "gmail"
    if "yahoo" in x: return "yahoo"
    if "hotmail" in x or "live" in x or "outlook" in x: return "microsoft"
    if ".edu" in x: return "edu"
    if ".gov" in x: return "gov"
    return "other"

def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    # TransactionDT -> time features
    if "TransactionDT" in df.columns:
        df["TransactionDT"] = pd.to_numeric(df["TransactionDT"], errors="coerce").fillna(0).astype("int64")
        dt = START_DATE + pd.to_timedelta(df["TransactionDT"], unit="s")
        df["DT_day"]  = (dt - START_DATE).dt.days
        df["DT_week"] = df["DT_day"] // 7
        df["DT_hour"] = dt.dt.hour
        df["DT_wday"] = dt.dt.weekday

    # TransactionAmt
    if "TransactionAmt" in df.columns:
        df["TransactionAmt_log1p"] = np.log1p(pd.to_numeric(df["TransactionAmt"], errors="coerce").fillna(0))
    else:
        df["TransactionAmt_log1p"] = 0.0

    # Email groups
    for col in ["P_emaildomain", "R_emaildomain"]:
        if col in df.columns:
            df[col+"_grp"] = df[col].astype("string").map(map_email)

    # DeviceInfo cleanup
    if "DeviceInfo" in df.columns:
        df["DeviceInfo_clean"] = df["DeviceInfo"].astype("string").str.split("/", n=1, expand=False).str[0].str.lower()

    # card / address
    if "card1" in df.columns:
        df["card1_bin"] = (pd.to_numeric(df["card1"], errors="coerce").fillna(0).astype("int64") // 100).astype("float32")
    if ("addr1" in df.columns) and ("addr2" in df.columns):
        df["addr1_addr2"] = (df["addr1"].astype("string") + "_" + df["addr2"].astype("string"))

    # null count across non-target
    non_target_cols = [c for c in df.columns if c != "isFraud"]
    df["nulls_count"] = df[non_target_cols].isna().sum(axis=1).astype("int32")

    return df

def encode_with_mapping(df: pd.DataFrame, cat_cols: list[str], cat_cats: dict, na_token: str) -> pd.DataFrame:
    # Ensure all CAT_COLS exist; add missing as NA
    for c in cat_cols:
        if c not in df.columns:
            df[c] = pd.Series(pd.NA, index=df.index, dtype="string")
    # Cast to string + fill NA token, then map using saved category list
    for c in cat_cols:
        s = df[c].astype("string").fillna(na_token)
        cats = cat_cats[c]
        # Guarantee NA token exists in categories
        if na_token not in cats:
            cats = cats + [na_token]
        dtype = pd.CategoricalDtype(categories=cats, ordered=False)
        df[c] = pd.Categorical(s, dtype=dtype).codes.astype("int32")
    return df

def align_features(df: pd.DataFrame, common_feats: list[str]) -> pd.DataFrame:
    for f in common_feats:
        if f not in df.columns:
            # create strictly numeric zero column
            df[f] = 0.0
    # exact order, drop extras
    return df[common_feats]

def ensure_numeric_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make absolutely sure the matrix is purely numeric and has no NAType.
    """
    # Replace pandas NA scalars with np.nan
    df = df.replace({pd.NA: np.nan})
    # Coerce any lingering non-numeric to numeric
    for c in df.columns:
        # Fast path if already numeric
        if pd.api.types.is_numeric_dtype(df[c]):
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # Fill NaNs and cast to a compact numeric type
    df = df.fillna(0.0)
    # HistGB accepts float; using float32 is fine and efficient
    return df.astype(np.float32)

def merge_ieee_like(transactions: pd.DataFrame, identity: Optional[pd.DataFrame]) -> pd.DataFrame:
    if identity is not None and "TransactionID" in identity.columns:
        return transactions.merge(identity, how="left", on="TransactionID")
    return transactions.copy()

def read_uploaded_csv(file) -> pd.DataFrame:
    """
    Read CSV or compressed CSV (.gz) from an uploaded file.
    """
    try:
        return pd.read_csv(file, compression="infer", low_memory=False)
    except ValueError:
        # Fallback if compression detection fails
        file.seek(0)
        return pd.read_csv(file, low_memory=False)

# =========================
# App UI
# =========================
st.title("🛡️ IEEE-CIS Fraud Risk Scoring & Insights")
st.caption("Upload transaction CSV (+ optional identity). The app reproduces training features/encodings, scores fraud risk, shows KPIs, charts, and lets you export flagged rows. Tip: If your CSVs are large, upload compressed *.csv.gz files to stay under Streamlit Cloud's 200 MB limit.")

# Check artifacts
needed = [MODEL_PATH, META_PATH, CAT_PATH, CF_PATH]
if not exists_all(needed):
    st.error(
        "Required artifacts are missing. Ensure these files are present in the app directory:\n"
        "- model_hgb_full.joblib\n- model_meta.json\n- cat_mapping.json\n- common_features.json"
    )
    st.stop()

# Load artifacts
with st.status("Loading model & preprocessing artifacts…", expanded=True) as status:
    model, meta, NA_TOKEN, CAT_COLS, CAT_CATS, COMMON_FEATS = load_artifacts()
    time.sleep(0.1)
    status.update(label=f"Artifacts loaded ✅  |  Features: {len(COMMON_FEATS)}", state="complete", expanded=False)

# Uploaders
colL, colR = st.columns([3,2])
with colL:
    tr_file = st.file_uploader(
        "Upload transactions CSV (.csv or .csv.gz)",
        type=["csv", "gz"], key="tr"
    )
with colR:
    id_file = st.file_uploader(
        "Upload identity CSV (optional, .csv or .csv.gz)",
        type=["csv", "gz"], key="id"
    )

with st.expander("Threshold & options", expanded=True):
    threshold = st.slider("Fraud threshold for flagging", min_value=0.01, max_value=0.99, value=0.5, step=0.01)
    compute_pi = st.checkbox("Compute permutation importance on a sample (slower)", value=False)
    pi_sample_n = st.number_input("Permutation importance sample size", min_value=5000, max_value=50000, value=20000, step=5000)

# =========================
# Inference pipeline with progress/guards
# =========================
if tr_file is not None:
    # Stage 1: Read uploads
    with st.status("Reading uploaded files…", expanded=True) as load_status:
        try:
            st.write("• Reading transactions file")
            tr = read_uploaded_csv(tr_file)
            time.sleep(0.05)
            if id_file is not None:
                st.write("• Reading identity file")
                id_df = read_uploaded_csv(id_file)
            else:
                id_df = None
                st.write("• No identity CSV provided (optional)")
            load_status.update(label="Files loaded ✅", state="complete", expanded=False)
        except Exception as e:
            st.error(f"Failed to read uploaded files: {e}")
            st.stop()

    # Stage 2: Preprocessing with progress bar
    prep_progress = st.progress(0, text="Merging…")
    try:
        df = merge_ieee_like(tr, id_df); prep_progress.progress(15, text="Merging…")
        df = downcast_df(df);           prep_progress.progress(30, text="Downcasting dtypes…")
        # Ensure object -> category for stability
        for c in df.columns:
            if df[c].dtype == object:
                df[c] = df[c].astype("category")
        prep_progress.progress(45, text="Preparing categories…")
        df = feature_engineer(df);      prep_progress.progress(70, text="Engineering features…")
        df = encode_with_mapping(df, CAT_COLS, CAT_CATS, NA_TOKEN); prep_progress.progress(85, text="Encoding categoricals…")
        X = align_features(df, COMMON_FEATS); prep_progress.progress(92, text="Aligning features…")
        X = ensure_numeric_matrix(X);   prep_progress.progress(100, text="Finalizing numeric matrix…")
        st.toast("Preprocessing complete", icon="✅")
    except Exception as e:
        st.error(f"Preprocessing failed: {e}")
        st.stop()

    # Stage 3: Scoring with progress bar (chunked)
    score_progress = st.progress(0, text="Scoring transactions…")
    try:
        n = len(X)
        probs = np.zeros(n, dtype="float32")
        if n == 0:
            st.warning("No rows to score.")
        else:
            chunk = max(1, n // 20)
            start = 0
            step = 0
            while start < n:
                end = min(start + chunk, n)
                probs[start:end] = model.predict_proba(X.iloc[start:end])[:, 1]
                start = end
                step += 1
                score_progress.progress(min(100, int((start / n) * 100)),
                                        text=f"Scoring transactions… {start}/{n}")
                time.sleep(0.01)
        score_progress.progress(100, text="Scoring complete!")
        st.toast("Scoring complete", icon="🧮")
    except Exception as e:
        st.error(f"Scoring failed: {e}")
        st.stop()

    # Results assembly
    scored = df.copy()
    scored["isFraud_proba"] = probs
    if "TransactionID" not in scored.columns and "TransactionID" in tr.columns:
        scored["TransactionID"] = tr["TransactionID"]
    flagged = scored[scored["isFraud_proba"] >= threshold].copy()

    # KPIs
    n_total = len(scored)
    n_flag  = len(flagged)
    pct_flag = (n_flag / max(n_total, 1)) * 100

    k1, k2, k3 = st.columns(3)
    k1.metric("Records Scored", f"{n_total:,}")
    k2.metric(f"Flagged ≥ {threshold:.2f}", f"{n_flag:,}", f"{pct_flag:.1f}%")
    k3.metric("Mean risk score", f"{np.mean(probs):.3f}" if n_total else "—")

    # Chart
    st.subheader("Score distribution")
    if n_total:
        fig = px.histogram(pd.DataFrame({"proba": probs, "flagged": probs >= threshold}),
                           x="proba", nbins=50, color="flagged", barmode="overlay", opacity=0.8)
        fig.update_layout(xaxis_title="Fraud probability", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No records to plot.")

    # Quick AUC if labels provided
    if "isFraud" in tr.columns:
        try:
            y_true = tr["isFraud"].astype(int).values
            auc = roc_auc_score(y_true, probs[:len(y_true)])
            st.info(f"Validation ROC-AUC on provided data: **{auc:.5f}**")
        except Exception:
            pass

    # Optional permutation importance (on sample)
    if st.checkbox("Compute permutation importance on a sample (slower)", value=False, key="pi_toggle"):
        st.subheader("Permutation importance (sample)")
        pi_sample_n = st.number_input("Sample size", min_value=5000, max_value=50000, value=20000, step=5000)
        with st.status("Computing permutation importance…", expanded=True) as pi_status:
            try:
                if len(X) == 0:
                    st.warning("No rows to sample for permutation importance.")
                else:
                    rng = np.random.RandomState(42)
                    idx = rng.choice(len(X), size=min(pi_sample_n, len(X)), replace=False)
                    X_s = X.iloc[idx]
                    # If labels available, use ROC-AUC; else proxy by model probas (not ideal but illustrative)
                    scoring = None
                    y_s = None
                    if "isFraud" in tr.columns:
                        y_s = tr.iloc[idx]["isFraud"].astype(int).values
                        scoring = "roc_auc"
                    pi_status.write("• Running permutation_importance (5 repeats)")
                    pi = permutation_importance(
                        model,
                        X_s,
                        y_s if scoring is not None else model.predict_proba(X_s)[:, 1],
                        n_repeats=5,
                        random_state=42,
                        scoring=scoring,
                        n_jobs=1
                    )
                    imp_df = pd.DataFrame({
                        "feature": X_s.columns,
                        "importance": pi.importances_mean,
                        "importance_std": pi.importances_std
                    }).sort_values("importance", ascending=False).head(30)
                    st.dataframe(imp_df, use_container_width=True)
                    pi_status.update(label="Permutation importance complete ✅", state="complete", expanded=False)
            except Exception as e:
                st.warning(f"Permutation importance failed: {e}")
                pi_status.update(label="Permutation importance failed", state="error", expanded=True)

    # Flagged preview & download
    st.subheader("Flagged transactions")
    show_cols = ["TransactionID", "isFraud_proba"]
    extra = [c for c in [
        "TransactionAmt","DT_day","DT_hour","card1","addr1",
        "P_emaildomain_grp","R_emaildomain_grp","DeviceInfo_clean","nulls_count"
    ] if c in flagged.columns]

    st.dataframe(
        flagged[show_cols + extra].sort_values("isFraud_proba", ascending=False).head(1000),
        use_container_width=True
    )
    csv_bytes = flagged.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download flagged CSV",
        data=csv_bytes,
        file_name="flagged_transactions.csv",
        mime="text/csv"
    )

else:
    st.info("Upload at least the transactions CSV to begin (identity CSV optional). Tip: Use compressed .csv.gz to avoid the 200 MB limit.")

# Hygiene
gc.collect()
