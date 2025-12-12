# counterfactual_analysis/analysis_counterfactual_bias_mitigation.py
#
# Counterfactual simulation for bias-mitigated models.
# Mirrors analysis_counterfactual.py but uses:
#   - saved_models/*.pkl
#   - mitigated/models/<model>_predictions.csv
#
# Outputs in this folder:
#   cf_group_mitigated_reweigh_sex.csv
#   cf_group_mitigated_expgrad_sex.csv
#   cf_group_mitigated_cal_eq_odds_sex.csv
#   cf_summary_mitigated_reweigh.csv
#   cf_summary_mitigated_expgrad.csv
#   cf_summary_mitigated_cal_eq_odds.csv

import numpy as np
import pandas as pd
import joblib
import scipy.sparse as sp
from pathlib import Path
import sys

# ---------------------------------------------------------------------
# Setup & imports
# ---------------------------------------------------------------------
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

try:
    from fairness_mitigation import CalibratedEqualizedOdds  

    globals()["CalibratedEqualizedOdds"] = CalibratedEqualizedOdds
except ImportError:
    # If import fails, we'll still be able to load other models
    pass

# ---- Load data ----
X_test_raw = pd.read_csv(parent_dir / "X_test_raw.csv")
y_test = pd.read_csv(parent_dir / "y_test.csv").squeeze()

# preprocessor 
preprocessor = joblib.load(parent_dir / "preprocessor.pkl")

models = [
    ("reweigh", "reweighed_logreg", False),
    ("expgrad", "expgrad_equalized_odds", False),
    ("cal_eq_odds", "calibrated_equalized_odds", True),
]

protected_attrs = ["sex"]  


def ensure_dense(X):
    """Convert sparse matrix to dense array if needed."""
    return X.toarray() if sp.issparse(X) else X


def generate_counterfactuals(X_raw: pd.DataFrame, attr: str) -> pd.DataFrame:
    """
    Return a new DataFrame where the protected attribute is swapped
    between its two groups (binary attribute only).
    """
    cf_df = X_raw.copy()
    unique_vals = X_raw[attr].unique()

    if len(unique_vals) != 2:
        raise ValueError(
            f"Counterfactual only supported for binary attrs. "
            f"{attr} has {len(unique_vals)} groups."
        )

    a, b = unique_vals
    cf_df[attr] = cf_df[attr].replace({a: b, b: a})
    return cf_df


# ---- MAIN LOOP ----
for display_name, file_stem, uses_sensitive in models:
    print(f"\n=== Counterfactual Fairness for mitigated model: {display_name} ===")

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    model_path = parent_dir / "saved_models" / f"{file_stem}.pkl"
    if not model_path.exists():
        print(f"  !! Skipping {display_name}: {model_path} not found")
        continue

    try:
        model = joblib.load(model_path)
    except Exception as e:
        print(f"  !! Could not load {model_path} ({type(e).__name__}: {e})")
        print("     Skipping this model.\n")
        continue

    # ------------------------------------------------------------------
    # Load original predictions from mitigated/models
    # ------------------------------------------------------------------
    preds_path = parent_dir / "mitigated" / "models" / f"{display_name}_predictions.csv"
    if not preds_path.exists():
        print(f"  !! Skipping {display_name}: {preds_path} not found")
        continue

    preds_df = pd.read_csv(preds_path)
    y_prob_orig = preds_df["y_prob"].to_numpy()
    y_pred_orig = preds_df["y_pred"].to_numpy().astype(int)

    # Storage for summary rows
    cf_summary = []

    for attr in protected_attrs:
        print(f"  → Testing counterfactual for {attr}")

        # --------------------------------------------------------------
        # Step 1: generate CF version of X_raw
        # --------------------------------------------------------------
        X_cf_raw = generate_counterfactuals(X_test_raw, attr)

        # --------------------------------------------------------------
        # Step 2: encode CF features using the SAME preprocessor
        # --------------------------------------------------------------
        X_cf_enc = preprocessor.transform(X_cf_raw)
        X_cf_enc = ensure_dense(X_cf_enc)

        # --------------------------------------------------------------
        # Step 3: model predictions on CF data
        # probabilities if available
        y_cf_prob = None
        if hasattr(model, "predict_proba"):
            try:
                y_cf_prob = model.predict_proba(X_cf_enc)[:, 1]
            except Exception:
                y_cf_prob = None

        # hard labels
        if uses_sensitive:
            A_cf = X_cf_raw[attr].values
            y_cf_pred = model.predict(X_cf_enc, sensitive_features=A_cf)
            y_cf_pred = np.asarray(y_cf_pred).astype(int)
            if y_cf_prob is None:
                y_cf_prob = y_cf_pred.astype(float)
        else:
            if y_cf_prob is not None:
                y_cf_pred = (y_cf_prob >= 0.5).astype(int)
            else:
                y_cf_pred = model.predict(X_cf_enc)
                y_cf_pred = np.asarray(y_cf_pred).astype(int)
                y_cf_prob = y_cf_pred.astype(float)

        # --------------------------------------------------------------
        # Step 4: metrics
        # --------------------------------------------------------------
        flip_rate = np.mean(y_cf_pred != y_pred_orig)
        score_shift = np.mean(y_cf_prob - y_prob_orig)
        rows = []
        for g in X_test_raw[attr].unique():
            idx = (X_test_raw[attr] == g)
            g_flip = np.mean(y_cf_pred[idx] != y_pred_orig[idx])
            rows.append({"group": g, "flip_rate": g_flip})

        df_group = pd.DataFrame(rows)
        group_out = (
            parent_dir
            / "counterfactual_analysis"
            / f"cf_group_mitigated_{display_name}_{attr}.csv"
        )
        df_group.to_csv(group_out, index=False)
        print(f"    Saved group-level flips to {group_out}")

        # Fairness gap (binary attr assumed)
        if len(df_group) == 2:
            gap = abs(df_group.iloc[0]["flip_rate"] - df_group.iloc[1]["flip_rate"])
        else:
            gap = np.nan

        cf_summary.append(
            {
                "protected_attr": attr,
                "flip_rate": flip_rate,
                "score_shift": score_shift,
                "cf_gap": gap,
            }
        )

    # save summary for this mitigated model
    summary_out = (
        parent_dir
        / "counterfactual_analysis"
        / f"cf_summary_mitigated_{display_name}.csv"
    )
    pd.DataFrame(cf_summary).to_csv(summary_out, index=False)
    print(f"  Saved summary to {summary_out}")

print("\n✔ Counterfactual summaries for mitigated models saved.")
