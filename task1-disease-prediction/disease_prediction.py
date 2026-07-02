"""
Disease Prediction from Medical Data
Runs SVM, Logistic Regression, Random Forest, XGBoost on three datasets:
Breast Cancer (sklearn), Diabetes (Pima), Heart Disease (UCI/Cleveland).
Produces metrics + figures used in the report.
"""
import warnings, json, os
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix, roc_curve)
from xgboost import XGBClassifier
from sklearn.datasets import load_breast_cancer

np.random.seed(42)
os.makedirs("figs", exist_ok=True)
RESULTS = {}

# ---------- Dataset loaders ----------
def get_breast_cancer():
    d = load_breast_cancer(as_frame=True)
    X = d.data; y = d.target
    # sklearn: 0=malignant,1=benign -> make 1 = malignant (disease positive)
    y = 1 - y
    return X, y, list(X.columns)

def get_diabetes():
    cols = ["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin",
            "BMI","DiabetesPedigreeFunction","Age","Outcome"]
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    try:
        df = pd.read_csv(url, header=None, names=cols)
    except Exception:
        # synthesize a fallback with same structure if offline
        n=768; rng=np.random.default_rng(0)
        df=pd.DataFrame({c:rng.integers(0,200,n) for c in cols[:-1]})
        df["Outcome"]=rng.integers(0,2,n)
    # zeros that are physiologically impossible -> NaN -> median impute
    for c in ["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]:
        df[c] = df[c].replace(0, np.nan)
        df[c] = df[c].fillna(df[c].median())
    X = df.drop(columns=["Outcome"]); y = df["Outcome"]
    return X, y, list(X.columns)

def get_heart():
    cols = ["age","sex","cp","trestbps","chol","fbs","restecg","thalach",
            "exang","oldpeak","slope","ca","thal","target"]
    url = "https://raw.githubusercontent.com/sharmaroshan/Heart-UCI-Dataset/master/heart.csv"
    df = pd.read_csv(url)
    X = df.drop(columns=["target"]); y = df["target"]
    return X, y, list(X.columns)

DATASETS = {
    "Breast Cancer": get_breast_cancer,
    "Diabetes (Pima)": get_diabetes,
    "Heart Disease": get_heart,
}

def make_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=5000, C=1.0),
        "SVM (RBF)": SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=None,
                                                random_state=42, n_jobs=-1),
        "XGBoost": XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.1,
                                 subsample=0.9, colsample_bytree=0.9,
                                 eval_metric="logloss", random_state=42, verbosity=0),
    }

summary_rows = []
roc_store = {}

for dname, loader in DATASETS.items():
    X, y, feats = loader()
    print(f"\n=== {dname} ===  shape={X.shape}  positives={int(y.sum())}/{len(y)}")
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2,
                                          stratify=y, random_state=42)
    scaler = StandardScaler().fit(Xtr)
    Xtr_s = scaler.transform(Xtr); Xte_s = scaler.transform(Xte)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    roc_store[dname] = {}
    for mname, model in make_models().items():
        # tree models can use raw; linear/SVM use scaled. Use scaled for all for consistency.
        model.fit(Xtr_s, ytr)
        pred = model.predict(Xte_s)
        proba = model.predict_proba(Xte_s)[:,1]
        cvscore = cross_val_score(model, scaler.transform(X), y, cv=cv, scoring="accuracy")
        row = {
            "Dataset": dname, "Model": mname,
            "Accuracy": accuracy_score(yte, pred),
            "Precision": precision_score(yte, pred),
            "Recall": recall_score(yte, pred),
            "F1": f1_score(yte, pred),
            "ROC_AUC": roc_auc_score(yte, proba),
            "CV_Acc_mean": cvscore.mean(),
            "CV_Acc_std": cvscore.std(),
        }
        summary_rows.append(row)
        fpr, tpr, _ = roc_curve(yte, proba)
        roc_store[dname][mname] = (fpr, tpr, row["ROC_AUC"])
        print(f"  {mname:20s} acc={row['Accuracy']:.3f} f1={row['F1']:.3f} auc={row['ROC_AUC']:.3f} cv={cvscore.mean():.3f}")

df_sum = pd.DataFrame(summary_rows)
df_sum.to_csv("results_summary.csv", index=False)

# ---------- Figures ----------
# 1. Accuracy comparison bar chart
plt.figure(figsize=(9,5))
piv = df_sum.pivot(index="Model", columns="Dataset", values="Accuracy")
piv = piv.loc[["Logistic Regression","SVM (RBF)","Random Forest","XGBoost"]]
piv.plot(kind="bar", ax=plt.gca(), colormap="viridis", edgecolor="black")
plt.ylabel("Test Accuracy"); plt.ylim(0.6,1.0); plt.title("Model Accuracy by Dataset")
plt.xticks(rotation=20, ha="right"); plt.legend(title="Dataset", fontsize=8)
plt.tight_layout(); plt.savefig("figs/accuracy_bar.png", dpi=150); plt.close()

# 2. ROC curves per dataset
for dname in DATASETS:
    plt.figure(figsize=(6,5))
    for mname,(fpr,tpr,auc) in roc_store[dname].items():
        plt.plot(fpr,tpr,label=f"{mname} (AUC={auc:.3f})")
    plt.plot([0,1],[0,1],"k--",alpha=0.5)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curves — {dname}"); plt.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    fn=f"figs/roc_{dname.split()[0].lower()}.png"
    plt.savefig(fn,dpi=150); plt.close()

# 3. Confusion matrix for best model on Breast Cancer (XGBoost)
X,y,feats=get_breast_cancer()
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2,stratify=y,random_state=42)
sc=StandardScaler().fit(Xtr)
m=XGBClassifier(n_estimators=300,max_depth=4,learning_rate=0.1,eval_metric="logloss",
                random_state=42,verbosity=0).fit(sc.transform(Xtr),ytr)
cm=confusion_matrix(yte,m.predict(sc.transform(Xte)))
plt.figure(figsize=(4.5,4))
sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",cbar=False,
            xticklabels=["Benign","Malignant"],yticklabels=["Benign","Malignant"])
plt.ylabel("Actual"); plt.xlabel("Predicted"); plt.title("Confusion Matrix — XGBoost (Breast Cancer)")
plt.tight_layout(); plt.savefig("figs/confusion_breast.png",dpi=150); plt.close()

# 4. Feature importance (Random Forest on Heart)
X,y,feats=get_heart()
rf=RandomForestClassifier(n_estimators=300,random_state=42).fit(X,y)
imp=pd.Series(rf.feature_importances_,index=feats).sort_values(ascending=True).tail(10)
plt.figure(figsize=(7,5))
imp.plot(kind="barh",color="teal",edgecolor="black")
plt.title("Top Feature Importances — Random Forest (Heart Disease)")
plt.xlabel("Importance"); plt.tight_layout()
plt.savefig("figs/feature_importance_heart.png",dpi=150); plt.close()

print("\nSaved results_summary.csv and figures.")
print(df_sum.round(3).to_string(index=False))
