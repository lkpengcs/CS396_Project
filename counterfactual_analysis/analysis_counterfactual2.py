import json
import numpy as np
import pandas as pd
import joblib
import scipy.sparse

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from advanced.advanced_fairness_logreg import FairnessRegularizedLogReg

# ---- Load data ----
parent_dir = Path(__file__).parent.parent
X_test_raw = pd.read_csv(parent_dir / "X_test_raw.csv")
y_test = pd.read_csv(parent_dir / "y_test.csv").squeeze()

# preprocessor (needed to encode CF version)
preprocessor = joblib.load(parent_dir / "preprocessor.pkl")

models = ["fair_reg_logreg"]
protected_attrs = ["sex"]

# ---- COUNTERFACTUAL GENERATOR ----
def generate_counterfactuals(X_raw, attr):
    """Return a new DataFrame where protected attr is swapped to the opposite group."""
    cf_df = X_raw.copy()
    unique_vals = X_raw[attr].unique()

    if len(unique_vals) != 2:
        # Only binary for simplicity (sex, race simplified, age_group)
        raise ValueError(f"Counterfactual only supported for binary attrs. {attr} has {len(unique_vals)} groups.")

    a, b = unique_vals
    # swap a <-> b
    cf_df[attr] = cf_df[attr].replace({a: b, b: a})
    return cf_df

# ---- MAIN LOOP ----
for name in models:
    print(f"\n=== Counterfactual Fairness for {name} ===")

    # Load model + baseline results
    model = joblib.load(parent_dir / "advanced" / "models" / f"{name}.pkl")
    predictions = pd.read_csv(parent_dir / "advanced" / "models" / f"{name}_predictions.csv")
    y_prob = predictions["y_prob"]
    y_pred = predictions["y_pred"]

    # Storage
    cf_summary = []

    for attr in protected_attrs:
        print(f"  → Testing counterfactual for {attr}")

        # Step 1: generate CF version of X_raw
        X_cf_raw = generate_counterfactuals(X_test_raw, attr)

        # Step 2: encode CF features using the SAME preprocessor
        X_cf_enc = preprocessor.transform(X_cf_raw)
        X_cf_enc = X_cf_enc.toarray()

        # Step 3: model predictions
        y_cf_prob = model.predict_proba(X_cf_enc)[:, 1]
        y_cf_pred = (y_cf_prob >= 0.5).astype(int)

        # Step 4: metrics
        flip_rate = np.mean(y_cf_pred != y_pred)
        score_shift = np.mean(y_cf_prob - y_prob)

        # group-level flip
        rows = []
        for g in X_test_raw[attr].unique():
            idx = (X_test_raw[attr] == g)
            g_flip = np.mean(y_cf_pred[idx] != y_pred[idx])
            rows.append({"group": g, "flip_rate": g_flip})

        df_group = pd.DataFrame(rows)
        df_group.to_csv(parent_dir / "counterfactual_analysis" / f"cf_group_{name}_{attr}.csv", index=False)

        # Fairness gap
        if len(df_group) == 2:
            gap = abs(df_group.iloc[0]["flip_rate"] - df_group.iloc[1]["flip_rate"])
        else:
            gap = np.nan

        cf_summary.append({
            "protected_attr": attr,
            "flip_rate": flip_rate,
            "score_shift": score_shift,
            "cf_gap": gap
        })

    pd.DataFrame(cf_summary).to_csv(parent_dir / "counterfactual_analysis" / f"cf_summary_{name}.csv", index=False)

print("\n✔ Step D complete: Counterfactual summary + group-level results saved.")
