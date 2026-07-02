# Disease Prediction from Medical Data

A machine learning project that predicts disease risk (Heart Disease, Diabetes,
Breast Cancer) from structured medical data using Logistic Regression, SVM,
Random Forest, and XGBoost — with an interactive **Streamlit** web frontend.

## Features

- Four classifiers compared on three UCI benchmark datasets.
- Best model per disease selected automatically by ROC-AUC.
- Polished, interactive Streamlit UI: adjustable patient inputs, an animated
  risk gauge, model-comparison table, and feature-importance charts.

## Project structure

```
disease-prediction/
├── disease_prediction.py   # full experiment pipeline (all models, metrics, figures)
├── train_models.py         # trains & saves the best model per dataset to models/
├── app.py                  # Streamlit frontend
├── models/                 # generated: saved models + index.json (after training)
├── figs/                   # generated charts
├── requirements.txt
└── README.md
```

## Setup

```bash
# 1. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

## Usage

### Step 1 — Train and save the models (run once)

```bash
python train_models.py
```

This trains all four classifiers on each dataset, picks the best by ROC-AUC,
and writes models/<dataset>.joblib plus models/index.json (metrics + feature
metadata). Internet access is required on first run to download the diabetes
and heart datasets.

### Step 2 — Launch the web app

```bash
streamlit run app.py
```

Your browser opens at http://localhost:8501. Pick a condition in the sidebar,
adjust the patient inputs, and press **Run prediction**.

### Optional — Reproduce the full experiment report

```bash
python disease_prediction.py
```

Prints all metrics, writes results_summary.csv, and saves comparison charts,
ROC curves, confusion matrix, and feature-importance plots to figs/.

## Datasets (auto-downloaded)

- Breast Cancer Wisconsin (via scikit-learn)
- Pima Indians Diabetes (public mirror)
- Cleveland Heart Disease (public mirror)

## Disclaimer

This project is an academic demonstration of machine-learning methodology. It
is **not** a medical device and must not be used for real clinical diagnosis.
