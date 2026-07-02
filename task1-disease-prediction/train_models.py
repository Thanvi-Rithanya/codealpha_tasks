"""
train_models.py
Trains the best classifier per dataset and saves the model, scaler, feature
metadata, and evaluation metrics to the models/ directory so the Streamlit
app can load them instantly for prediction.
"""
import warnings, json, os
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, joblib
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)
from xgboost import XGBClassifier
from sklearn.datasets import load_breast_cancer

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
os.makedirs("models", exist_ok=True)


# ---------------- Dataset loaders ----------------
def load_breast_cancer_data():
    d = load_breast_cancer(as_frame=True)
    X, y = d.data, (1 - d.target)            # 1 = malignant
    meta = {f: {"label": f, "min": float(X[f].min()), "max": float(X[f].max()),
                "mean": float(X[f].mean())} for f in X.columns}
    return X, y, meta, {"pos": "Malignant", "neg": "Benign"}


def load_diabetes_data():
    cols = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]
    url = ("https://raw.githubusercontent.com/jbrownlee/Datasets/"
           "master/pima-indians-diabetes.data.csv")
    df = pd.read_csv(url, header=None, names=cols)
    for c in ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]:
        df[c] = df[c].replace(0, np.nan)
        df[c] = df[c].fillna(df[c].median())
    X, y = df.drop(columns=["Outcome"]), df["Outcome"]
    labels = {
        "Pregnancies": "Number of Pregnancies",
        "Glucose": "Glucose (mg/dL)",
        "BloodPressure": "Blood Pressure (mm Hg)",
        "SkinThickness": "Skin Thickness (mm)",
        "Insulin": "Insulin (mu U/ml)",
        "BMI": "Body Mass Index (BMI)",
        "DiabetesPedigreeFunction": "Diabetes Pedigree Function",
        "Age": "Age (years)",
    }
    meta = {f: {"label": labels[f], "min": float(X[f].min()),
                "max": float(X[f].max()), "mean": float(X[f].mean())}
            for f in X.columns}
    return X, y, meta, {"pos": "Diabetic", "neg": "Not Diabetic"}


def load_heart_data():
    url = ("https://raw.githubusercontent.com/sharmaroshan/"
           "Heart-UCI-Dataset/master/heart.csv")
    df = pd.read_csv(url)
    X, y = df.drop(columns=["target"]), df["target"]
    labels = {
        "age": "Age (years)", "sex": "Sex (1=male, 0=female)",
        "cp": "Chest Pain Type (0-3)", "trestbps": "Resting BP (mm Hg)",
        "chol": "Cholesterol (mg/dL)", "fbs": "Fasting Blood Sugar >120 (1/0)",
        "restecg": "Resting ECG (0-2)", "thalach": "Max Heart Rate",
        "exang": "Exercise Angina (1/0)", "oldpeak": "ST Depression (oldpeak)",
        "slope": "ST Slope (0-2)", "ca": "Major Vessels (0-4)",
        "thal": "Thalassemia (0-3)",
    }
    meta = {f: {"label": labels[f], "min": float(X[f].min()),
                "max": float(X[f].max()), "mean": float(X[f].mean())}
            for f in X.columns}
    return X, y, meta, {"pos": "Heart Disease", "neg": "No Heart Disease"}


DATASETS = {
    "breast_cancer": ("Breast Cancer", load_breast_cancer_data),
    "diabetes":      ("Diabetes",      load_diabetes_data),
    "heart":         ("Heart Disease", load_heart_data),
}


def candidate_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=5000, C=1.0),
        "SVM (RBF)": SVC(kernel="rbf", C=1.0, gamma="scale",
                         probability=True, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=300,
                         random_state=RANDOM_STATE, n_jobs=-1),
        "XGBoost": XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.1,
                    subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
                    random_state=RANDOM_STATE, verbosity=0),
    }


def main():
    index = {}
    for key, (name, loader) in DATASETS.items():
        X, y, meta, classes = loader()
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
        scaler = StandardScaler().fit(Xtr)
        Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

        best, best_name, best_auc, all_metrics = None, None, -1, {}
        for mname, model in candidate_models().items():
            model.fit(Xtr_s, ytr)
            pred = model.predict(Xte_s)
            proba = model.predict_proba(Xte_s)[:, 1]
            m = {
                "accuracy": round(accuracy_score(yte, pred), 4),
                "precision": round(precision_score(yte, pred), 4),
                "recall": round(recall_score(yte, pred), 4),
                "f1": round(f1_score(yte, pred), 4),
                "roc_auc": round(roc_auc_score(yte, proba), 4),
                "cv_acc": round(cross_val_score(model, scaler.transform(X), y,
                                cv=cv, scoring="accuracy").mean(), 4),
            }
            all_metrics[mname] = m
            if m["roc_auc"] > best_auc:
                best_auc, best, best_name = m["roc_auc"], model, mname

        # Feature importance (if available) for display
        feat_imp = {}
        if hasattr(best, "feature_importances_"):
            feat_imp = {f: round(float(v), 4)
                        for f, v in zip(X.columns, best.feature_importances_)}

        joblib.dump({"model": best, "scaler": scaler}, f"models/{key}.joblib")
        index[key] = {
            "name": name, "best_model": best_name,
            "classes": classes, "features": list(X.columns),
            "feature_meta": meta, "metrics": all_metrics,
            "best_metrics": all_metrics[best_name],
            "feature_importance": feat_imp,
            "n_samples": int(len(y)), "n_positive": int(y.sum()),
        }
        print(f"{name:15s} best={best_name:20s} auc={best_auc:.3f}")

    json.dump(index, open("models/index.json", "w"), indent=2)
    print("\nSaved models/ and models/index.json")


if __name__ == "__main__":
    main()
