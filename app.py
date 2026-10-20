import io, json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, classification_report, ConfusionMatrixDisplay)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

st.set_page_config(page_title="Industrial AI Copilot", page_icon="🏭", layout="wide")
st.title("🏭 Industrial AI Copilot")
st.caption("Predictive maintenance • Explainable ML • What-if analysis • Reproducible evaluation")
st.warning("Educational prototype, not a safety-critical diagnostic tool. Validate on documented real data before making operational claims.")

def demo_data(n=2200, seed=17):
    rng = np.random.default_rng(seed)
    typ = rng.choice(["L", "M", "H"], n, p=[.5, .3, .2])
    air = rng.normal(300, 2, n)
    process = air + rng.normal(10, 1.5, n)
    speed = rng.normal(1530, 160, n).clip(900, 2200)
    torque = rng.normal(40, 9, n).clip(5, 80)
    wear = rng.uniform(0, 250, n)
    effect = pd.Series(typ).map({"L": -.15, "M": 0, "H": .2}).to_numpy()
    z = -5 + .025*(process-310) + .075*(torque-40) + .010*(wear-100) + .0015*(1500-speed) + effect
    prob = 1/(1+np.exp(-z))
    failure = (rng.random(n) < prob).astype(int)
    return pd.DataFrame({"Type":typ, "Air temperature [K]":air,
        "Process temperature [K]":process, "Rotational speed [rpm]":speed,
        "Torque [Nm]":torque, "Tool wear [min]":wear, "Machine failure":failure})

def target_to_binary(s):
    if pd.api.types.is_numeric_dtype(s):
        if set(s.dropna().unique()).issubset({0, 1, 0.0, 1.0}) and s.nunique() == 2:
            return s.astype("Int64")
        return None
    mapping = {"0":0, "normal":0, "no":0, "false":0, "1":1, "failure":1, "failed":1, "yes":1, "true":1}
    out = s.astype(str).str.strip().str.lower().map(mapping)
    return out.astype("Int64") if out.notna().sum() else None

def build_model(X, kind):
    nums = X.select_dtypes(include=np.number).columns.tolist()
    cats = [c for c in X.columns if c not in nums]
    transformers = []
    if nums:
        transformers.append(("numeric", Pipeline([("imputer", SimpleImputer(strategy="median"))]), nums))
    if cats:
        transformers.append(("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), cats))
    prep = ColumnTransformer(transformers=transformers)
    if kind == "Random Forest":
        clf = RandomForestClassifier(n_estimators=350, min_samples_leaf=2,
            class_weight="balanced_subsample", random_state=42, n_jobs=-1)
    else:
        clf = HistGradientBoostingClassifier(max_iter=180, learning_rate=.08,
            l2_regularization=1.0, random_state=42)
    return Pipeline([("preprocess", prep), ("model", clf)])

def positive_scores(model, X):
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
        classes = list(model.named_steps["model"].classes_)
        return probs[:, classes.index(1)] if 1 in classes else np.zeros(len(X))
    return model.predict(X).astype(float)

with st.sidebar:
    st.header("Configuration")
    upload = st.file_uploader("Upload CSV", type=["csv"])
    if upload:
        try:
            df = pd.read_csv(upload)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            st.stop()
        source = upload.name
    else:
        df, source = demo_data(), "Synthetic demo data"
    if df.empty:
        st.error("Dataset is empty.")
        st.stop()
    common = [c for c in ["Machine failure", "Failure", "failure", "target", "Target"] if c in df.columns]
    default_target = common[0] if common else df.columns[-1]
    target = st.selectbox("Binary target column (0/1)", df.columns.tolist(),
        index=df.columns.get_loc(default_target))
    model_kind = st.selectbox("Model", ["Random Forest", "Histogram Gradient Boosting"])
    test_size = st.slider("Test set fraction", .15, .40, .20, .05)
    train = st.button("Train and evaluate", type="primary", use_container_width=True)
    st.caption("Without a CSV, synthetic data is used to demonstrate the workflow.")

st.subheader("Dataset overview")
a,b,c,d = st.columns(4)
a.metric("Rows", f"{len(df):,}")
b.metric("Columns", len(df.columns))
c.metric("Source", "Demo" if source == "Synthetic demo data" else "Uploaded CSV")
d.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")
with st.expander("Preview data"):
    st.dataframe(df.head(20), use_container_width=True)
with st.expander("Data quality report"):
    st.dataframe(pd.DataFrame({"dtype":df.dtypes.astype(str),
        "missing":df.isna().sum(), "missing_%":(df.isna().mean()*100).round(2),
        "unique":df.nunique()}), use_container_width=True)

if "result_bundle" not in st.session_state:
    st.session_state.result_bundle = None

if train:
    y = target_to_binary(df[target])
    if y is None:
        st.error("Target must be binary 0/1 (0 = no failure, 1 = failure).")
        st.stop()
    X = df.drop(columns=[target]).copy()
    X = X.loc[:, X.notna().any()]
    X = X.loc[:, X.nunique(dropna=True) > 1]
    valid = y.notna() & X.notna().any(axis=1)
    X, y = X.loc[valid].copy(), y.loc[valid].astype(int)
    if len(X) < 50 or X.shape[1] == 0 or y.nunique() != 2 or y.value_counts().min() < 2:
        st.error("Need at least 50 usable rows, at least one feature, and examples of both classes.")
        st.stop()
    try:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size,
            random_state=42, stratify=y)
        model = build_model(Xtr, model_kind)
        with st.spinner("Training model..."):
            model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        scores = positive_scores(model, Xte)
        metrics = {
            "Accuracy": accuracy_score(yte, pred),
            "Precision": precision_score(yte, pred, zero_division=0),
            "Recall": recall_score(yte, pred, zero_division=0),
            "F1-score": f1_score(yte, pred, zero_division=0),
            "ROC-AUC": roc_auc_score(yte, scores),
            "Average precision": average_precision_score(yte, scores)
        }
        st.session_state.result_bundle = {
            "model":model, "X":X, "X_test":Xte, "y_test":yte, "pred":pred,
            "metrics":metrics, "report":classification_report(yte,pred,zero_division=0),
            "target":target, "model_kind":model_kind, "source":source,
            "timestamp":datetime.now(timezone.utc).isoformat(), "features":X.columns.tolist()
        }
    except Exception as e:
        st.error(f"Training failed: {e}")
        st.stop()

bundle = st.session_state.result_bundle
if bundle:
    if bundle["source"] == "Synthetic demo data":
        st.warning("These results are from synthetic data only and do not demonstrate real-world performance.")
    st.subheader("Held-out test metrics")
    metric_cols = st.columns(6)
    for col, (name, value) in zip(metric_cols, bundle["metrics"].items()):
        col.metric(name, f"{value:.3f}")
    left, right = st.columns(2)
    with left:
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay.from_predictions(bundle["y_test"], bundle["pred"], ax=ax,
            colorbar=False, values_format="d")
        ax.set_title("Confusion matrix")
        st.pyplot(fig)
        plt.close(fig)
    with right:
        st.markdown("**Classification report**")
        st.code(bundle["report"])

    st.subheader("Explainability: permutation importance")
    if st.button("Calculate feature importance"):
        with st.spinner("Calculating importance..."):
            try:
                imp = permutation_importance(bundle["model"], bundle["X_test"], bundle["y_test"],
                    scoring="f1", n_repeats=5, random_state=42, n_jobs=-1)
                importance = pd.Series(imp.importances_mean, index=bundle["X_test"].columns).sort_values()
                fig, ax = plt.subplots()
                importance.plot(kind="barh", ax=ax)
                ax.set_title("Permutation importance (F1)")
                st.pyplot(fig)
                plt.close(fig)
                st.dataframe(importance.sort_values(ascending=False).rename("Mean importance").to_frame())
            except Exception as e:
                st.error(f"Importance calculation failed: {e}")

    st.subheader("What-if prediction")
    st.write("Change feature values and inspect the model output. This is not a safety verdict.")
    vals = {}
    left, right = st.columns(2)
    X = bundle["X"]
    for i, col in enumerate(bundle["features"]):
        with (left if i % 2 == 0 else right):
            s = X[col]
            if pd.api.types.is_numeric_dtype(s):
                med = float(pd.to_numeric(s, errors="coerce").median())
                lo, hi = float(s.min()), float(s.max())
                if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
                    lo, hi = med-1, med+1
                vals[col] = st.number_input(col, value=med,
                    min_value=float(lo-abs(lo)*.1-1), max_value=float(hi+abs(hi)*.1+1),
                    format="%.3f", key="input_"+col)
            else:
                choices = sorted(s.dropna().astype(str).unique().tolist()) or ["unknown"]
                mode = str(s.mode(dropna=True).iloc[0]) if not s.mode(dropna=True).empty else choices[0]
                vals[col] = st.selectbox(col, choices,
                    index=choices.index(mode) if mode in choices else 0, key="input_"+col)
    if st.button("Predict failure risk", type="primary"):
        row = pd.DataFrame([vals], columns=bundle["features"])
        for col in row.columns:
            if not pd.api.types.is_numeric_dtype(X[col]):
                row[col] = row[col].astype(str)
        score = float(positive_scores(bundle["model"], row)[0])
        cls = int(bundle["model"].predict(row)[0])
        p1,p2 = st.columns(2)
        p1.metric("Model-estimated failure score", f"{score:.1%}")
        p2.metric("Predicted class", "Failure" if cls == 1 else "No failure")
        st.warning("Verify model output using qualified inspection and established procedures.")

    st.subheader("Export trained model and run manifest")
    buffer = io.BytesIO()
    joblib.dump(bundle["model"], buffer)
    st.download_button("Download fitted model (.joblib)", buffer.getvalue(),
        "industrial_ai_model.joblib", "application/octet-stream")
    manifest = {"app":"Industrial AI Copilot", "model":bundle["model_kind"],
        "target":bundle["target"], "data_source":bundle["source"],
        "trained_at_utc":bundle["timestamp"], "features":bundle["features"],
        "metrics":{k:float(v) for k,v in bundle["metrics"].items()},
        "note":"Synthetic-data metrics are illustrative only."}
    st.download_button("Download run manifest (.json)", json.dumps(manifest,indent=2),
        "run_manifest.json", "application/json")

st.divider()
st.caption("Built with Python, Streamlit, pandas, scikit-learn and Matplotlib.")
