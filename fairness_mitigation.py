# fairness_mitigation.py
#
# Trains 3 bias-mitigated models:
#   1) Reweighed Logistic Regression        [pre-processing]
#   2) ExponentiatedGradient (Eq. Odds)     [in-processing]
#   3) CalibratedEqualizedOdds              [post-processing, custom class]
#
# Saves:
#   - Models → saved_models/*.pkl

from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.calibration import CalibratedClassifierCV

from fairlearn.reductions import ExponentiatedGradient, EqualizedOdds
from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.metrics import (
    MetricFrame,
    selection_rate,
    true_positive_rate,
    false_positive_rate,
)

# ---------------------------------------------------------------------
# Global config for Logistic Regression


LOGREG_KWARGS = dict(
    max_iter=3000,     
    n_jobs=-1,
    solver="saga",    
    random_state=0,
    C=1.0,
    penalty="l2",
)

EG_EPS = 0.03        
EG_MAX_ITER = 50     

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
SAVE_DIR = ROOT / "saved_models"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

FAIR_DIR = ROOT / "fairness_analysis"
FAIR_DIR.mkdir(parents=True, exist_ok=True)


def save_model(model, name: str):
    path = SAVE_DIR / f"{name}.pkl"
    joblib.dump(model, path)
    print(f"[+] Saved model → {path.relative_to(ROOT)}")


# ---------------------------------------------------------------------
# Data loading & preprocessing
# ---------------------------------------------------------------------

def load_splits(label_col: str = "income"):
    X_train_raw = pd.read_csv(ROOT / "X_train_raw.csv")
    X_val_raw   = pd.read_csv(ROOT / "X_val_raw.csv")
    X_test_raw  = pd.read_csv(ROOT / "X_test_raw.csv")

    y_train = pd.read_csv(ROOT / "y_train.csv")[label_col].values
    y_val   = pd.read_csv(ROOT / "y_val.csv")[label_col].values
    y_test  = pd.read_csv(ROOT / "y_test.csv")[label_col].values

    return X_train_raw, X_val_raw, X_test_raw, y_train, y_val, y_test


def load_preprocessor():
    # fitted ColumnTransformer/Pipeline from baseline code
    return joblib.load(ROOT / "preprocessor.pkl")


def transform_features(preprocessor, X_train_raw, X_val_raw, X_test_raw):
    X_train = preprocessor.transform(X_train_raw)
    X_val   = preprocessor.transform(X_val_raw)
    X_test  = preprocessor.transform(X_test_raw)
    return X_train, X_val, X_test


def ensure_dense(X):
    return X.toarray() if hasattr(X, "toarray") else X


# ---------------------------------------------------------------------
# Pre-processing: Kamiran & Calders reweighing
# ---------------------------------------------------------------------

def compute_reweighing_weights(A, y):
    df = pd.DataFrame({"A": A, "y": y})
    n = len(df)

    p_A = df["A"].value_counts(normalize=True)
    p_y = df["y"].value_counts(normalize=True)
    p_Ay = df.groupby(["A", "y"]).size() / n

    weights = np.zeros(n, dtype=float)
    for i, (a_i, y_i) in enumerate(zip(df["A"], df["y"])):
        num = p_A[a_i] * p_y[y_i]
        den = p_Ay.loc[(a_i, y_i)]
        weights[i] = num / den if den > 0 else 0.0
    return weights


def train_reweighed_logreg(X_train, y_train, A_train):
    sample_weight = compute_reweighing_weights(A_train, y_train)
    clf = LogisticRegression(**LOGREG_KWARGS)
    clf.fit(X_train, y_train, sample_weight=sample_weight)
    return clf


# ---------------------------------------------------------------------
# In-processing: Exponentiated Gradient (Equalized Odds, full data)
# ---------------------------------------------------------------------

def train_exponentiated_gradient(X_train, y_train, A_train):
    X_train_dense = ensure_dense(X_train)

    base_estimator = LogisticRegression(**LOGREG_KWARGS)
    constraint = EqualizedOdds()
    eg = ExponentiatedGradient(
        estimator=base_estimator,
        constraints=constraint,
        eps=0.03,     
        max_iter=50, 
    )
    eg.fit(X_train_dense, y_train, sensitive_features=A_train)
    return eg



# ---------------------------------------------------------------------
# Post-processing: CalibratedEqualizedOdds
#   - Step 1: calibrate probabilities (isotonic) on dense validation data
#   - Step 2: ThresholdOptimizer with equalized odds on calibrated scores
# ---------------------------------------------------------------------

class CalibratedEqualizedOdds:
    """
    'Calibrated Equalized Odds' using:
      - CalibratedClassifierCV for probability calibration
      - ThresholdOptimizer(constraints="equalized_odds") for post-processing
    """
    def __init__(self, base_estimator=None,
                 method: str = "isotonic",
                 cv: str | int = "prefit"):
        if base_estimator is None:
            base_estimator = LogisticRegression(**LOGREG_KWARGS)
        self.base_estimator = base_estimator
        self.method = method
        self.cv = cv

        self._calibrated = None
        self._postproc = None

    def fit(self, X, y, sensitive_features):
        X_dense = ensure_dense(X)

        if self.cv != "prefit":
            self.base_estimator.fit(X_dense, y)

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


def train_calibrated_equalized_odds(X_train, y_train, X_val, y_val, A_val):
    X_train_dense = ensure_dense(X_train)
    X_val_dense   = ensure_dense(X_val)

    base = LogisticRegression(**LOGREG_KWARGS)
    base.fit(X_train_dense, y_train)

    ceo = CalibratedEqualizedOdds(
        base_estimator=base,
        method="isotonic",
        cv="prefit",
    )
    ceo.fit(X_val_dense, y_val, sensitive_features=A_val)
    return ceo


# ---------------------------------------------------------------------
# Metrics (performance + fairness)
# ---------------------------------------------------------------------

def performance_metrics(y_true, y_pred, y_score=None):
    m = {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
    }
    if y_score is not None:
        try:
            m["roc_auc"] = roc_auc_score(y_true, y_score)
        except Exception:
            m["roc_auc"] = np.nan
    else:
        m["roc_auc"] = np.nan
    return m


def fairness_metrics(y_true, y_pred, A):
    mf = MetricFrame(
        metrics={
            "selection_rate": selection_rate,
            "tpr": true_positive_rate,
            "fpr": false_positive_rate,
        },
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=A,
    )

    disparities = {
        "selection_rate_gap": mf["selection_rate"].max() - mf["selection_rate"].min(),
        "tpr_gap":            mf["tpr"].max()             - mf["tpr"].min(),
        "fpr_gap":            mf["fpr"].max()             - mf["fpr"].min(),
    }
    return mf.by_group, disparities


def evaluate_model(name, model, X_test, y_test, A_test,
                   uses_sensitive_in_predict=False):
    if uses_sensitive_in_predict:
        y_pred = model.predict(X_test, sensitive_features=A_test)
    else:
        y_pred = model.predict(X_test)

    y_score = None
    if hasattr(model, "predict_proba"):
        try:
            y_score = model.predict_proba(X_test)[:, 1]
        except Exception:
            y_score = None

    perf = performance_metrics(y_test, y_pred, y_score)
    group_df, gaps = fairness_metrics(y_test, y_pred, A_test)
    group_df = group_df.reset_index().rename(columns={"index": "group"})
    group_df.insert(0, "model", name)

    return perf, gaps, group_df


# ---------------------------------------------------------------------
# Main: train + save
# ---------------------------------------------------------------------

def main(label_col="income", protected_attribute="sex"):
    X_train_raw, X_val_raw, X_test_raw, y_train, y_val, y_test = load_splits(label_col)
    preprocessor = load_preprocessor()
    X_train, X_val, X_test = transform_features(
        preprocessor, X_train_raw, X_val_raw, X_test_raw
    )

    # NOTE: X_* stay sparse here for reweigh + EG
    A_train = X_train_raw[protected_attribute].values
    A_val   = X_val_raw[protected_attribute].values
    A_test  = X_test_raw[protected_attribute].values

    # --- train 3 bias-mitigated models on full data ---

    print("Training reweighed logistic regression (full data)...")
    reweigh_model = train_reweighed_logreg(X_train, y_train, A_train)
    save_model(reweigh_model, "reweighed_logreg")

    print("Training ExponentiatedGradient (Equalized Odds, full data)...")
    eg_model = train_exponentiated_gradient(X_train, y_train, A_train)
    save_model(eg_model, "expgrad_equalized_odds")

    print("Training Calibrated Equalized Odds post-processor...")
    ceo_model = train_calibrated_equalized_odds(
        X_train, y_train, X_val, y_val, A_val
    )
    save_model(ceo_model, "calibrated_equalized_odds") ##


if __name__ == "__main__":
    main(label_col="income", protected_attribute="sex")
