# evaluate_bias_models.py

# Loads the three trained bias-mitigated models:
#   - reweighed_logreg.pkl
#   - expgrad_equalized_odds.pkl
#   - calibrated_equalized_odds.pkl
#
# Evaluates them using metrics.py (same as baseline) and writes:
#   fairness_analysis/bias_mitigation_eval_metrics.csv

from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from fairlearn.postprocessing import ThresholdOptimizer

# import your baseline metrics helpers
from metrics import (
    prepare_groups,
    calculate_performance_metrics,
    evaluate_metrics,
)

# ---------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
SAVE_DIR = ROOT / "saved_models"
OUT_DIR = ROOT / "fairness_analysis"
OUT_DIR.mkdir(exist_ok=True)


def ensure_dense(X):
    """Convert sparse matrices to dense numpy arrays if needed."""
    return X.toarray() if hasattr(X, "toarray") else X


# ---------------------------------------------------------------------
# the custom class needed to unpickle calibrated_equalized_odds.pkl
# ---------------------------------------------------------------------

class CalibratedEqualizedOdds:
    """
    'Calibrated Equalized Odds' using:
      - CalibratedClassifierCV for probability calibration
      - ThresholdOptimizer(constraints="equalized_odds") for post-processing

    NOTE: For evaluation, we only need predict() and predict_proba().
          __init__/fit exist to satisfy unpickling, but won't be called.
    """

    def __init__(self, base_estimator=None,
                 method: str = "isotonic",
                 cv: str | int = "prefit"):
        # These defaults are irrelevant for loading, but we keep them for safety.
        LOGREG_KWARGS = dict(
            max_iter=3000,
            n_jobs=-1,
            solver="saga",
            random_state=0,
            C=1.0,
            penalty="l2",
        )
        if base_estimator is None:
            base_estimator = LogisticRegression(**LOGREG_KWARGS)
        self.base_estimator = base_estimator
        self.method = method
        self.cv = cv

        self._calibrated = None
        self._postproc = None

    def fit(self, X, y, sensitive_features):
        X_dense = ensure_dense(X)

        calib = CalibratedClassifierCV(
            self.base_estimator,
            method=self.method,
            cv=self.cv,
        )
        calib.fit(X_dense, y)
        self._calibrated = calib

        self._postproc = ThresholdOptimizer(
            estimator=calib,
            constraints="equalized_odds",
            prefit=True,
        )
        self._postproc.fit(X_dense, y, sensitive_features=sensitive_features)
        return self

    def predict(self, X, sensitive_features):
        X_dense = ensure_dense(X)
        return self._postproc.predict(X_dense, sensitive_features=sensitive_features)

    def predict_proba(self, X):
        X_dense = ensure_dense(X)
        return self._calibrated.predict_proba(X_dense)


# ---------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------

def main(label_col="income", protected_attr="sex"):
    # -------------------------------------------------
    # 1. Load raw test data + labels
    # -------------------------------------------------
    X_test_raw = pd.read_csv(ROOT / "X_test_raw.csv")
    y_test = pd.read_csv(ROOT / "y_test.csv")[label_col].values

    # protected groups (e.g. "sex", "race", or binned "age")
    groups = prepare_groups(X_test_raw, protected_attr)

    # -------------------------------------------------
    # 2. Load preprocessor and transform X_test
    # -------------------------------------------------
    preprocessor = joblib.load(ROOT / "preprocessor.pkl")
    X_test_sparse = preprocessor.transform(X_test_raw)
    # Make test data dense to avoid fairlearn/sparse length issues
    X_test = ensure_dense(X_test_sparse)

    # -------------------------------------------------
    # 3. Load trained bias-mitigated models
    # -------------------------------------------------
    reweigh_model = joblib.load(SAVE_DIR / "reweighed_logreg.pkl")
    eg_model      = joblib.load(SAVE_DIR / "expgrad_equalized_odds.pkl")
    ceo_model     = joblib.load(SAVE_DIR / "calibrated_equalized_odds.pkl")

    # -------------------------------------------------
    # 4. Helper to evaluate one model with metrics.py
    # -------------------------------------------------
    def eval_model(model, name, uses_sensitive=False):
        # Predictions
        if uses_sensitive:
            y_pred = model.predict(X_test, sensitive_features=groups)
        else:
            y_pred = model.predict(X_test)

        # Probabilities for AUROC + calibration
        if hasattr(model, "predict_proba"):
            try:
                y_prob = model.predict_proba(X_test)[:, 1]
            except Exception:
                y_prob = y_pred.astype(float)
        else:
            y_prob = y_pred.astype(float)

        # Performance (accuracy, precision, recall, f1, auroc)
        perf = calculate_performance_metrics(y_test, y_pred, y_prob)

        # Fairness & calibration
        fair = evaluate_metrics(y_test, y_pred, y_prob, groups)

        row = {
            "model": name,
            "accuracy":    perf["accuracy"],
            "precision":   perf["precision"],
            "recall":      perf["recall"],
            "f1":          perf["f1"],
            "auroc":       perf["auroc"],
            "dpd":         fair["dpd"],
            "eod":         fair["eod"],
            "fprd":        fair["fprd"],
            "overall_ECE": fair["calibration"]["overall_ECE"],
        }
        return row

    # -------------------------------------------------
    # 5. Evaluate all three bias-mitigated models
    # -------------------------------------------------
    rows = []
    rows.append(eval_model(reweigh_model, "reweigh"))
    rows.append(eval_model(eg_model,      "expgrad"))
    rows.append(eval_model(ceo_model,     "cal_eq_odds", uses_sensitive=True))

    df = pd.DataFrame(rows)

    out_path = OUT_DIR / "bias_mitigation_eval_metrics.csv"
    df.to_csv(out_path, index=False)

    print("=== Bias-mitigated models evaluated with metrics.py ===")
    print(df.to_string(index=False))
    print(f"\n[+] Saved evaluation metrics → {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main(label_col="income", protected_attr="sex")
