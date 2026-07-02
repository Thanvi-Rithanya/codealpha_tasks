"""
app.py — Disease Prediction from Medical Data
Interactive Streamlit frontend. Run with:  streamlit run app.py
Loads pre-trained models from models/ (run train_models.py first).
"""
import json
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
import streamlit as st

# ----------------------------------------------------------------------
# Page configuration
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Disease Prediction from Medical Data",
    page_icon="\U0001FA7A",  # stethoscope
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------
# Custom theme / CSS
# ----------------------------------------------------------------------
CSS = """
<style>
:root {
  --ink:#0F2A3F; --teal:#0E7C7B; --teal-dark:#0A5C5B;
  --mist:#EAF3F4; --amber:#E8A23D; --rose:#D7494E; --line:#D6E2E4;
}
html, body, [class*="css"] { font-family: 'Inter','Segoe UI',sans-serif; }
.block-container { padding-top: 1.6rem; max-width: 1180px; }

/* Hero */
.hero {
  background: linear-gradient(135deg, var(--ink) 0%, var(--teal-dark) 100%);
  border-radius: 18px; padding: 30px 34px; color: #fff; margin-bottom: 14px;
  box-shadow: 0 10px 30px rgba(14,42,63,.18);
}
.hero h1 { color:#fff; font-size: 2.0rem; font-weight: 800; margin:0 0 6px 0; letter-spacing:-.5px;}
.hero p  { color:#CFE5E6; font-size: 1.02rem; margin:0; max-width: 760px;}
.hero .pill { display:inline-block; background: rgba(255,255,255,.14); color:#EAF3F4;
  padding: 3px 12px; border-radius: 999px; font-size:.74rem; font-weight:600;
  letter-spacing:.6px; text-transform:uppercase; margin-bottom: 12px;}

/* Metric cards */
.card { background:#fff; border:1px solid var(--line); border-radius:14px;
  padding:16px 18px; box-shadow:0 2px 8px rgba(15,42,63,.05); height:100%;}
.card .k { font-size:.72rem; text-transform:uppercase; letter-spacing:.7px; color:#5C7480; font-weight:700;}
.card .v { font-size:1.7rem; font-weight:800; color:var(--ink); line-height:1.1; margin-top:3px;}
.card .s { font-size:.78rem; color:#7A909A; margin-top:2px;}

/* Result banners */
.result { border-radius:14px; padding:20px 24px; margin-top:6px; border-left:6px solid;}
.result.pos { background:#FCEDED; border-color:var(--rose);}
.result.neg { background:#E9F6F0; border-color:#1F9D6B;}
.result h2 { margin:0; font-size:1.35rem; font-weight:800;}
.result.pos h2 { color:var(--rose);} .result.neg h2 { color:#1F9D6B;}
.result p { margin:6px 0 0 0; color:#43575F; font-size:.92rem;}

.section-label { font-size:.78rem; text-transform:uppercase; letter-spacing:1px;
  color:var(--teal-dark); font-weight:800; margin: 4px 0 2px 0;}
.note { background:var(--mist); border-radius:10px; padding:12px 16px; font-size:.84rem;
  color:#3D545C; border:1px solid var(--line);}
.stButton>button { background:var(--teal); color:#fff; font-weight:700; border:0;
  border-radius:10px; padding:.55rem 1.3rem; transition:.15s;}
.stButton>button:hover { background:var(--teal-dark); transform: translateY(-1px);}
div[data-testid="stSidebarNav"] { display:none; }
.disclaimer { font-size:.74rem; color:#8A9AA1; border-top:1px solid var(--line);
  margin-top:26px; padding-top:12px;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ----------------------------------------------------------------------
# Load model index + artifacts (cached)
# ----------------------------------------------------------------------
@st.cache_resource
def load_index():
    with open("models/index.json") as f:
        return json.load(f)


@st.cache_resource
def load_artifact(key):
    return joblib.load(f"models/{key}.joblib")


try:
    INDEX = load_index()
except FileNotFoundError:
    st.error("Models not found. Please run `python train_models.py` first to "
             "train and save the models, then reload this page.")
    st.stop()

DATASET_KEYS = {v["name"]: k for k, v in INDEX.items()}


# ----------------------------------------------------------------------
# Gauge chart
# ----------------------------------------------------------------------
def risk_gauge(prob):
    pct = prob * 100
    color = "#1F9D6B" if pct < 33 else ("#E8A23D" if pct < 66 else "#D7494E")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        number={"suffix": "%", "font": {"size": 40, "color": "#0F2A3F"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#9FB4BB"},
            "bar": {"color": color, "thickness": 0.28},
            "bgcolor": "white",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 33], "color": "#E9F6F0"},
                {"range": [33, 66], "color": "#FBF1DF"},
                {"range": [66, 100], "color": "#FCEDED"},
            ],
            "threshold": {"line": {"color": color, "width": 4},
                          "thickness": 0.75, "value": pct},
        },
    ))
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=20, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", font={"family": "Inter"})
    return fig


def importance_chart(imp):
    items = sorted(imp.items(), key=lambda x: x[1])[-8:]
    labels = [k for k, _ in items]
    vals = [v for _, v in items]
    fig = go.Figure(go.Bar(x=vals, y=labels, orientation="h",
                           marker_color="#0E7C7B"))
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10),
                      paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)",
                      xaxis_title="Importance", font={"family": "Inter"})
    return fig


# ----------------------------------------------------------------------
# Sidebar — dataset / disease selector
# ----------------------------------------------------------------------
with st.sidebar:
    st.markdown("### \U0001FA7A  Disease Predictor")
    st.caption("A machine-learning decision-support demo")
    disease = st.radio("Choose a condition to assess",
                       list(DATASET_KEYS.keys()), index=0)
    key = DATASET_KEYS[disease]
    info = INDEX[key]
    st.divider()
    st.markdown("#### Model in use")
    st.markdown(f"**{info['best_model']}**")
    st.caption(f"Selected automatically as the best performer on the "
               f"{info['name']} dataset (highest ROC-AUC).")
    st.divider()
    st.caption("Educational project. Not for clinical use.")


# ----------------------------------------------------------------------
# Hero
# ----------------------------------------------------------------------
st.markdown(f"""
<div class="hero">
  <span class="pill">Machine Learning · Classification</span>
  <h1>Disease Prediction from Medical Data</h1>
  <p>Enter patient measurements to estimate the likelihood of
  <b>{info['name'].lower()}</b>. The model was trained on the UCI benchmark
  dataset and selected for the strongest validated performance.</p>
</div>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------
# Top metric cards
# ----------------------------------------------------------------------
bm = info["best_metrics"]
c1, c2, c3, c4 = st.columns(4)
for col, k, v, s in [
    (c1, "Accuracy", f"{bm['accuracy']*100:.1f}%", "held-out test set"),
    (c2, "ROC-AUC", f"{bm['roc_auc']:.3f}", "ranking quality"),
    (c3, "Recall", f"{bm['recall']*100:.1f}%", "true cases caught"),
    (c4, "Dataset", f"{info['n_samples']}", f"{info['n_positive']} positive"),
]:
    col.markdown(f'<div class="card"><div class="k">{k}</div>'
                 f'<div class="v">{v}</div><div class="s">{s}</div></div>',
                 unsafe_allow_html=True)

st.write("")

tab_predict, tab_model, tab_about = st.tabs(
    ["  Predict  ", "  Model insights  ", "  About  "])

# ----------------------------------------------------------------------
# TAB 1 — Prediction form
# ----------------------------------------------------------------------
with tab_predict:
    left, right = st.columns([1.15, 1], gap="large")

    with left:
        st.markdown('<div class="section-label">Patient inputs</div>',
                    unsafe_allow_html=True)
        st.caption("Adjust the values, then run the prediction. "
                   "Defaults are set to the dataset average.")

        feats = info["features"]
        meta = info["feature_meta"]
        values = {}
        # two-column input grid
        fcols = st.columns(2)
        for i, f in enumerate(feats):
            m = meta[f]
            target = fcols[i % 2]
            lo, hi, mean = m["min"], m["max"], m["mean"]
            span = hi - lo
            # integer-like fields get integer steps
            is_int = abs(span) > 5 and float(span).is_integer() and \
                float(lo).is_integer() and float(hi).is_integer()
            if is_int:
                values[f] = target.slider(
                    m["label"], int(lo), int(hi), int(round(mean)))
            else:
                step = max(round(span / 100, 3), 0.001)
                values[f] = target.slider(
                    m["label"], float(lo), float(hi), float(round(mean, 2)),
                    step=float(step))

        run = st.button("Run prediction", use_container_width=True)

    with right:
        st.markdown('<div class="section-label">Result</div>',
                    unsafe_allow_html=True)
        if run:
            art = load_artifact(key)
            X = pd.DataFrame([[values[f] for f in feats]], columns=feats)
            Xs = art["scaler"].transform(X)
            prob = float(art["model"].predict_proba(Xs)[0, 1])
            pred = int(prob >= 0.5)
            st.plotly_chart(risk_gauge(prob), use_container_width=True,
                            config={"displayModeBar": False})
            cls = info["classes"]
            if pred:
                st.markdown(
                    f'<div class="result pos"><h2>Elevated risk · '
                    f'{cls["pos"]}</h2><p>The model estimates a '
                    f'{prob*100:.1f}% probability consistent with '
                    f'{cls["pos"].lower()}. Recommend clinical follow-up. '
                    f'This is a screening aid, not a diagnosis.</p></div>',
                    unsafe_allow_html=True)
            else:
                st.markdown(
                    f'<div class="result neg"><h2>Lower risk · '
                    f'{cls["neg"]}</h2><p>The model estimates a '
                    f'{prob*100:.1f}% probability of '
                    f'{cls["pos"].lower()}. Routine monitoring is typically '
                    f'sufficient, but clinical judgment always applies.</p></div>',
                    unsafe_allow_html=True)
        else:
            st.markdown('<div class="note">Set the patient values on the left '
                        'and press <b>Run prediction</b>. The gauge will show '
                        'the estimated probability of disease, color-coded from '
                        'green (low) through amber to red (high).</div>',
                        unsafe_allow_html=True)

# ----------------------------------------------------------------------
# TAB 2 — Model insights
# ----------------------------------------------------------------------
with tab_model:
    st.markdown('<div class="section-label">Model comparison on this dataset</div>',
                unsafe_allow_html=True)
    st.caption("All four classifiers were trained and evaluated identically. "
               "The highlighted row is the model used for predictions.")
    rows = []
    for mname, m in info["metrics"].items():
        rows.append({
            "Model": mname, "Accuracy": m["accuracy"], "Precision": m["precision"],
            "Recall": m["recall"], "F1": m["f1"], "ROC-AUC": m["roc_auc"],
            "CV Acc.": m["cv_acc"],
        })
    df = pd.DataFrame(rows)

    def hl(r):
        return ["background-color:#E9F6F0; font-weight:700"
                if r["Model"] == info["best_model"] else "" for _ in r]
    st.dataframe(df.style.apply(hl, axis=1).format(
        {c: "{:.3f}" for c in ["Accuracy", "Precision", "Recall",
                               "F1", "ROC-AUC", "CV Acc."]}),
        use_container_width=True, hide_index=True)

    if info["feature_importance"]:
        st.markdown('<div class="section-label">What drives the prediction</div>',
                    unsafe_allow_html=True)
        st.caption("Top features by importance for the selected model.")
        st.plotly_chart(importance_chart(info["feature_importance"]),
                        use_container_width=True,
                        config={"displayModeBar": False})
    else:
        st.info("Feature importances are not available for this model type "
                "(e.g. SVM/Logistic Regression). Switch datasets to see them "
                "for tree-based models.")

# ----------------------------------------------------------------------
# TAB 3 — About
# ----------------------------------------------------------------------
with tab_about:
    st.markdown("""
This application is the interactive frontend for the **Disease Prediction from
Medical Data** machine-learning project. It loads classifiers trained on three
public UCI benchmark datasets and lets you explore predictions interactively.

**How it works**

1. `train_models.py` trains Logistic Regression, SVM, Random Forest, and
   XGBoost on each dataset, then saves the best model (by ROC-AUC) together
   with its scaler and metrics.
2. This app loads those saved models and applies the same preprocessing
   (median imputation + standardization) to your inputs before predicting.
3. The probability is shown on the gauge, with model comparison and feature
   importance available under **Model insights**.

**Datasets:** Breast Cancer Wisconsin, Pima Indians Diabetes, Cleveland Heart
Disease — all from the UCI Machine Learning Repository.
    """)
    st.markdown('<div class="disclaimer">Disclaimer: This tool is an academic '
                'demonstration of machine-learning methodology. It is not a '
                'medical device and must not be used for real clinical '
                'diagnosis or treatment decisions.</div>',
                unsafe_allow_html=True)
