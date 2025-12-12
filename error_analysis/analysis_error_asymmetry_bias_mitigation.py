# error_analysis/analysis_error_asymmetry_bias_mitigation.py
#
# Error-asymmetry + grouped feature-importance for BIAS-MITIGATED models.
#
# Models (trained in fairness_mitigation.py and evaluated in mitigated/mitigated.py):
#   - reweigh       → saved_models/reweighed_logreg.pkl
#   - expgrad       → saved_models/expgrad_equalized_odds.pkl
#   - cal_eq_odds   → saved_models/calibrated_equalized_odds.pkl
#
# Uses:
#   - mitigated/models/<display_name>_predictions.csv for y_pred, y_prob
#   - baseline/transformed_X_test.npz for encoded features
#   - baseline/feature_names.json for feature names
#
# Writes (under error_analysis/):
#   - error_analysis/reweigh/feature_importance.csv
#   - error_analysis/reweigh/error_asymmetry_<attr>.csv
#   - error_analysis/expgrad/feature_importance.csv
#   - error_analysis/expgrad/error_asymmetry_<attr>.csv
#   - error_analysis/cal_eq_odds/error_asymmetry_<attr>.csv 

import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import confusion_matrix
from sklearn.inspection import permutation_importance
import scipy.sparse as sp
import sys

# ---------------------------------------------------------------------
# Paths & setup
# ---------------------------------------------------------------------
parent_dir = Path(__file__).parent.parent

sys.path.insert(0, str(parent_dir))
try:
    from fairness_mitigation import CalibratedEqualizedOdds  
    globals()["CalibratedEqualizedOdds"] = CalibratedEqualizedOdds
except ImportError:
    pass


def ensure_dense(X):
    return X.toarray() if sp.issparse(X) else X

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

    if protected_attr == "age":
        groups = pd.cut(
            groups,
            bins=[0, 30, 50, 100],
            labels=["young", "middle", "old"],
        )
    else:
        groups = groups.astype(str)

    rows = []
    for g in groups.unique():
        idx = (groups == g)
        fpr, fnr, tpr, tnr = error_breakdown(y_test[idx], y_pred[idx])
        rows.append(
            {
                "group": g,
                "FPR": fpr,
                "FNR": fnr,
                "TPR": tpr,
                "TNR": tnr,
            }
        )

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
        total = 0.0
        for p in prefixes:
            mask = df["feature"].str.startswith(p)
            total += df.loc[mask, "importance"].sum()
        grouped[group] = total
    return pd.DataFrame(grouped.items(), columns=["feature_group", "importance"])


if __name__ == "__main__":
    print("▶ Loading cached data for mitigated error analysis...")

    # -----------------------------------------------------------------
    # Load test data & encoded features
    # -----------------------------------------------------------------
    X_test_raw = pd.read_csv(parent_dir / "X_test_raw.csv")
    y_test = pd.read_csv(parent_dir / "y_test.csv").squeeze()

    protected_attrs = ["sex", "race", "age", "marital.status", "education", "relationship"]

    # Encoded test features (same as baseline)
    X_test_enc = sp.load_npz(parent_dir / "baseline" / "transformed_X_test.npz")
    X_test_enc = ensure_dense(X_test_enc)

    # Feature names
    feature_names = json.load(open(parent_dir / "baseline" / "feature_names.json"))

    models = [
        ("reweigh", "reweighed_logreg", "logreg"),
        ("expgrad", "expgrad_equalized_odds", "perm"),
        ("cal_eq_odds", "calibrated_equalized_odds", "skip"),
    ]

    for display_name, file_stem, fi_mode in models:
        print(f"\n=== Evaluating mitigated model: {display_name} ===")

        # --------------------------------------------------------------
        # Load model
        # --------------------------------------------------------------
        model_path = parent_dir / "saved_models" / f"{file_stem}.pkl"
        if not model_path.exists():
            print(f"  !! Skipping {display_name}: {model_path} not found")
            continue

        try:
            model = joblib.load(model_path)
        except Exception as e:
            print(f"  !! Could not load {model_path} ({type(e).__name__}: {e})")
            print("     Skipping this model.")
            continue

        # --------------------------------------------------------------
        # Load mitigated predictions
        # --------------------------------------------------------------
        preds_path = parent_dir / "mitigated" / "models" / f"{display_name}_predictions.csv"
        if not preds_path.exists():
            print(f"  !! Skipping {display_name}: {preds_path} not found")
            continue

        preds_df = pd.read_csv(preds_path)
        y_pred = preds_df["y_pred"].to_numpy().astype(int)

        # --------------------------------------------------------------
        # Feature importance
        # --------------------------------------------------------------
        out_dir = parent_dir / "error_analysis" / display_name
        out_dir.mkdir(parents=True, exist_ok=True)

        if fi_mode == "logreg":
            # Logistic regression-like: use absolute coefficients
            importances = np.abs(model.coef_[0])
            fi_df_raw = pd.DataFrame(
                {
                    "feature": feature_names,
                    "importance": importances,
                }
            )
            fi_grouped = group_importance(fi_df_raw).sort_values(
                "importance", ascending=False
            )
            fi_grouped.to_csv(out_dir / "feature_importance.csv", index=False)
            print("  Saved grouped feature importance (logreg).")

        elif fi_mode == "perm":
            # Generic model: permutation importance
            try:
                pfi = permutation_importance(
                    model,
                    X_test_enc,
                    y_test,
                    n_repeats=10,
                    random_state=42,
                    scoring="accuracy", 
                )
                fi_df_raw = pd.DataFrame(
                    {
                        "feature": feature_names,
                        "importance": pfi.importances_mean,
                    }
                )
                fi_grouped = group_importance(fi_df_raw).sort_values(
                    "importance", ascending=False
                )
                fi_grouped.to_csv(out_dir / "feature_importance.csv", index=False)
                print("  Saved grouped feature importance (permutation).")
            except Exception as e:
                print(f"  !! Skipping feature importance for {display_name}: {type(e).__name__}: {e}")

        else:

            print("  (Skipping feature importance for cal_eq_odds; requires sensitive_features.)")

        # --------------------------------------------------------------
        # Error asymmetry per protected attribute
        # --------------------------------------------------------------
        for attr in protected_attrs:
            df_err = group_error_asymmetry(y_test, y_pred, X_test_raw, attr)
            df_err.to_csv(out_dir / f"error_asymmetry_{attr}.csv", index=False)
            print(f"  Saved error asymmetry for {attr} to {out_dir / f'error_asymmetry_{attr}.csv'}")

    print("\n✔ DONE (mitigated error analysis complete, no retraining needed!)")