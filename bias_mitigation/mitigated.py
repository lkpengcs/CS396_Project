# mitigated/mitigated.py
#
# Evaluate 3 bias-mitigated models trained in fairness_mitigation.py:
#   1) reweigh     → saved_models/reweighed_logreg.pkl
#   2) expgrad     → saved_models/expgrad_equalized_odds.pkl
#   3) cal_eq_odds → saved_models/calibrated_equalized_odds.pkl
#
# Writes:
#   - mitigated/mitigated_results.json
#   - mitigated/models/<model_name>_predictions.csv

import json
import joblib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

# ---------------------------------------------------------------------
# Make sure we can import metrics and fairness_mitigation
# ---------------------------------------------------------------------
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))  # so we can import fairness_mitigation + metrics

from metrics import evaluate_metrics, calculate_performance_metrics, prepare_groups  # noqa: E402

# Import the CalibratedEqualizedOdds class used when training
try:
    from fairness_mitigation import CalibratedEqualizedOdds  # noqa: F401

    # Make pickle happy when it looks for __main__.CalibratedEqualizedOdds
    globals()["CalibratedEqualizedOdds"] = CalibratedEqualizedOdds
except ImportError:
    # If this fails, we'll just skip cal_eq_odds model later
    pass


def ensure_dense(X):
    """Convert sparse matrix to dense array if needed."""
    return X.toarray() if sp.issparse(X) else X


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # 1. Load raw splits
    # ------------------------------------------------------------------
    X_train_raw = pd.read_csv(parent_dir / "X_train_raw.csv")
    y_train = pd.read_csv(parent_dir / "y_train.csv").squeeze()

    X_test_raw = pd.read_csv(parent_dir / "X_test_raw.csv")
    y_test = pd.read_csv(parent_dir / "y_test.csv").squeeze()

    # ------------------------------------------------------------------
    # 2. Load preprocessor + transform test set
    # ------------------------------------------------------------------
    preprocessor = joblib.load(parent_dir / "preprocessor.pkl")

    X_test = preprocessor.transform(X_test_raw)
    X_test_dense = ensure_dense(X_test)

    # Protected attributes for fairness evaluation
    protected_attrs = ["sex", "race", "age", "marital.status", "education", "relationship"]

    # ------------------------------------------------------------------
    # 3. Define mitigated models to evaluate
    #    (display_name, file_stem, use_dense, use_sensitive_for_predict)
    # ------------------------------------------------------------------
    models = [
        ("reweigh",     "reweighed_logreg",          False, False),
        ("expgrad",     "expgrad_equalized_odds",    True,  False),
        ("cal_eq_odds", "calibrated_equalized_odds", True,  True),
    ]

    all_results = []

    # directory to save predictions (parallel to baseline/baseline_models.py)
    model_dir = parent_dir / "mitigated" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    for display_name, file_stem, use_dense, use_sensitive in models:
        print(f"\n{'='*60}")
        print(f"Evaluating mitigated model: {display_name} ({file_stem}.pkl)")
        print(f"{'='*60}")

        model_path = parent_dir / "saved_models" / f"{file_stem}.pkl"
        if not model_path.exists():
            print(f"  !! Skipping {display_name}: {model_path} not found")
            continue

        # --------------------------------------------------------------
        # 3a. Load model (robustly: skip if CalibratedEqualizedOdds pickle fails)
        # --------------------------------------------------------------
        try:
            model = joblib.load(model_path)
        except Exception as e:
            print(f"  !! Could not load {model_path} ({type(e).__name__}: {e})")
            print("     Skipping this model.\n")
            continue

        # Choose representation of X for this model
        X_for_model = X_test_dense if use_dense else X_test

        # --------------------------------------------------------------
        # 3b. Get predictions & (if available) probabilities
        # --------------------------------------------------------------
        if use_sensitive:
            # Needed for CalibratedEqualizedOdds ThresholdOptimizer
            sens = X_test_raw["sex"].values
            y_pred = model.predict(X_for_model, sensitive_features=sens)
        else:
            y_pred = model.predict(X_for_model)

        y_pred = np.asarray(y_pred).astype(int)

        y_prob = None
        if hasattr(model, "predict_proba"):
            try:
                y_prob = model.predict_proba(X_for_model)[:, 1]
            except Exception:
                y_prob = None

        # For performance metrics, fall back to hard preds if no probs
        if y_prob is None:
            y_prob_for_metrics = y_pred.astype(float)
        else:
            y_prob_for_metrics = y_prob

        # --------------------------------------------------------------
        # 3b.5 Save predictions (just like baseline_models.py)
        # --------------------------------------------------------------
        predictions_path = model_dir / f"{display_name}_predictions.csv"
        predictions_df = pd.DataFrame(
            {
                "y_pred": y_pred,
                "y_prob": y_prob_for_metrics,  # use true probs if available, else 0/1
            }
        )
        predictions_df.to_csv(predictions_path, index=False)
        print(f"  Saved predictions to {predictions_path}")

        # Optionally also save a copy of the model here (not strictly needed)
        joblib.dump(model, model_dir / f"{display_name}.pkl")

        # --------------------------------------------------------------
        # 3c. Performance metrics
        # --------------------------------------------------------------
        performance = calculate_performance_metrics(
            y_true=y_test.values,
            y_pred=y_pred,
            y_prob=y_prob_for_metrics,
        )

        # --------------------------------------------------------------
        # 3d. Fairness metrics for each protected attribute
        # --------------------------------------------------------------
        fairness = {}
        for protected_attr in protected_attrs:
            print(f"  Evaluating with protected attribute: {protected_attr}")
            groups = prepare_groups(X_test_raw, protected_attr)
            fairness_results = evaluate_metrics(
                y_true=y_test.values,
                y_pred=y_pred,
                y_prob=y_prob_for_metrics,
                groups=groups,
            )

            fairness[protected_attr] = {
                "dpd":         fairness_results["dpd"],
                "eod":         fairness_results["eod"],
                "fprd":        fairness_results["fprd"],
                "calibration": fairness_results["calibration"],
            }

        # --------------------------------------------------------------
        # 3e. Collect result entry
        # --------------------------------------------------------------
        model_result = {
            "model_name": display_name,
            "model_file": file_stem,
            "performance": performance,
            "fairness": fairness,
        }
        all_results.append(model_result)

    # ------------------------------------------------------------------
    # 4. Save results
    # ------------------------------------------------------------------
    output_path = Path(__file__).parent / "mitigated_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=4)

    print(f"\n✔ All mitigated metrics saved to {output_path}")
    print(f"✔ Predictions saved under {model_dir}")
