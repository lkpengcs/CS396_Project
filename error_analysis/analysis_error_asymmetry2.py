# analysis_error_asymmetry.py

import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import confusion_matrix
from sklearn.inspection import permutation_importance
import scipy.sparse
import sys
# Get parent directory path
parent_dir = Path(__file__).parent.parent
# Add parent directory to path to import from advanced module
sys.path.insert(0, str(parent_dir))
from advanced.advanced_fairness_logreg import FairnessRegularizedLogReg

def error_breakdown(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan
    fnr = fn / (fn + tp) if (fn + tp) > 0 else np.nan
    tpr = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    tnr = tn / (tn + fp) if (tn + fp) > 0 else np.nan

    return fpr, fnr, tpr, tnr


def group_error_asymmetry(y_test, y_pred, X_test_raw, protected_attr):
    groups = X_test_raw[protected_attr]

    rows = []
    if protected_attr == "age":
        groups = pd.cut(groups, bins=[0, 30, 50, 100], labels=["young", "middle", "old"])
    else:
        groups = groups.astype(str)

    for g in groups.unique():
        idx = (groups == g)
        fpr, fnr, tpr, tnr = error_breakdown(y_test[idx], y_pred[idx])
        rows.append({
            "group": g,
            "FPR": fpr,
            "FNR": fnr,
            "TPR": tpr,
            "TNR": tnr
        })

    return pd.DataFrame(rows)


group_map = {
    "age": ["age"],
    "workclass": ["workclass"],
    "fnlwgt": ["fnlwgt"],
    "education": ["education"],
    "education.num": ["education.num"],
    "marital.status": ["marital.status"],
    "occupation": ["occupation"],
    "relationship": ["relationship"],
    "race": ["race"],
    "sex": ["sex"],
    "capital.gain": ["capital.gain"],
    "capital.loss": ["capital.loss"],
    "hours.per.week": ["hours.per.week"],
    "native.country": ["native.country"],
}

def group_importance(df):
    """
    df: feature_importance_xxx.csv (two columns: feature, importance)
    return: dataframe with grouped importance
    """
    grouped = {}
    for group, prefixes in group_map.items():
        total = 0
        for p in prefixes:
            mask = df["feature"].str.startswith(p)
            total += df.loc[mask, "importance"].sum()
        grouped[group] = total
    return pd.DataFrame(grouped.items(), columns=["feature_group", "importance"])

if __name__ == "__main__":
    print("▶ Loading cached data...")

    X_test_raw = pd.read_csv("X_test_raw.csv")
    y_test = pd.read_csv("y_test.csv").squeeze()

    protected_attrs = ["sex", "race", "age", "marital.status", "education", "relationship"]

    models = ["fair_reg_logreg"]

    for name in models:
        print(f"\n=== Evaluating {name} ===")

        # Load baseline-predicted results
        model = joblib.load(parent_dir / "advanced" / "models" / f"{name}.pkl")

        predictions = pd.read_csv(parent_dir / "advanced" / "models" / f"{name}_predictions.csv")
        y_pred = predictions["y_pred"]
        y_prob = predictions["y_prob"]
        X_test_enc = scipy.sparse.load_npz(parent_dir / "advanced" / "transformed_X_test.npz")
        X_test_enc = X_test_enc.toarray()

        # Permutation feature importance
        feature_names = json.load(open(parent_dir / "baseline" / "feature_names.json"))
        if name == "fair_reg_logreg":
            # Use coef_
            importances = np.abs(model.coef_[0])
            fi_df = pd.DataFrame({
                "feature": feature_names,
                "importance": importances
            })
        else:
            pfi = permutation_importance(model, X_test_enc, y_test, n_repeats=10, random_state=42)
            fi_df = pd.DataFrame({
                "feature": feature_names,
                "importance": pfi.importances_mean
            })
        fi_df = group_importance(fi_df).sort_values("importance", ascending=False)
        # save under each model directory
        # create directory if it doesn't exist
        (parent_dir / "error_analysis" / name).mkdir(parents=True, exist_ok=True)
        fi_df.to_csv(parent_dir / "error_analysis" / name / f"feature_importance2.csv", index=False)

        # Per protected attribute error asymmetry
        for attr in protected_attrs:
            df = group_error_asymmetry(y_test, y_pred, X_test_raw, attr)
            df.to_csv(parent_dir / "error_analysis" / name / f"error_asymmetry2_{attr}.csv", index=False)

    print("\n✔ DONE (no retraining needed!)")
