# bias_mitigation_models.py
import json
import joblib
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import scipy.sparse

try:
    from fairlearn.postprocessing import CalibratedEqualizedOdds
    from fairlearn.reductions import ExponentiatedGradient

    globals()["CalibratedEqualizedOdds"] = CalibratedEqualizedOdds
    globals()["ExponentiatedGradient"] = ExponentiatedGradient
except ImportError:
    pass

# Add parent directory to path to import metrics
sys.path.insert(0, str(Path(__file__).parent.parent))
from metrics import evaluate_metrics, calculate_performance_metrics, prepare_groups


def get_labels(model, X_enc, sens=None):
    """
    Unified helper for hard predictions:
    - tries predict(X, sensitive_features=sens) (fairlearn postprocessing)
    - falls back to ExponentiatedGradient's _pmf_predict
    - then falls back to plain sklearn predict(X)
    """
    # fairlearn-style predict with sensitive_features
    try:
        y = model.predict(X_enc, sensitive_features=sens)
        return np.asarray(y).astype(int)
    except Exception:
        pass

    # ExponentiatedGradient internal pmf
    try:
        pmf = model._pmf_predict(X_enc, sensitive_features=sens)
        return np.argmax(pmf, axis=1).astype(int)
    except Exception:
        pass

    # plain sklearn
    y = model.predict(X_enc)
    return np.asarray(y).astype(int)


def get_probs(model, X_enc, sens=None):
    """
    Try to get P(y=1) from the model:
    - predict_proba(X, sensitive_features=sens)
    - predict_proba(X)
    - ExponentiatedGradient._pmf_predict
    - CalibratedEqualizedOdds.probability
    Returns np.ndarray or None if not available.
    """
    # predict_proba with sensitive_features
    try:
        proba = model.predict_proba(X_enc, sensitive_features=sens)[:, 1]
        return np.asarray(proba)
    except Exception:
        pass

    # plain predict_proba
    try:
        proba = model.predict_proba(X_enc)[:, 1]
        return np.asarray(proba)
    except Exception:
        pass

    # ExponentiatedGradient internal pmf
    try:
        pmf = model._pmf_predict(X_enc, sensitive_features=sens)
        return np.asarray(pmf[:, 1])
    except Exception:
        pass

    # CalibratedEqualizedOdds probability()
    try:
        proba = model.probability(X_enc, sensitive_features=sens)
        return np.asarray(proba)
    except Exception:
        pass

    # No probabilities
    return None


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
if __name__ == "__main__":
    parent_dir = Path(__file__).parent.parent

    # Raw splits
    X_train_raw = pd.read_csv(parent_dir / "X_train_raw.csv")
    y_train = pd.read_csv(parent_dir / "y_train.csv").squeeze()

    X_test_raw = pd.read_csv(parent_dir / "X_test_raw.csv")
    y_test = pd.read_csv(parent_dir / "y_test.csv").squeeze()

    # Encoder (same as baseline)
    preprocessor = joblib.load(parent_dir / "preprocessor.pkl")

    # Protected attributes for fairness evaluation (same as baseline)
    protected_attrs = ["sex", "race", "age", "marital.status", "education", "relationship"]

    # Bias-mitigated models saved in saved_models/
    mitigated_models = [
        ("reweigh",     parent_dir / "saved_models" / "reweighed_logreg.pkl"),
        ("expgrad",     parent_dir / "saved_models" / "expgrad_equalized_odds.pkl"),
        ("cal_eq_odds", parent_dir / "saved_models" / "calibrated_equalized_odds.pkl"),
    ]

    # Directory to store predictions / results
    model_dir = parent_dir / "bias_mitigation" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    # ---------------------------------------------------------------
    # Prepare encoded features + feature names (for consistency)
    # ---------------------------------------------------------------
    X_train = preprocessor.transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)

    # Identify numeric and categorical
    num_cols = X_train_raw.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X_train_raw.select_dtypes(include=["object"]).columns.tolist()

    # Get OHE feature names
    ohe = preprocessor.named_transformers_["cat"]
    cat_feature_names = ohe.get_feature_names_out(cat_cols).tolist()

    feature_names = num_cols + cat_feature_names

    # Save feature names once (parallel to baseline/feature_names.json)
    feature_names_path = parent_dir / "bias_mitigation" / "feature_names.json"
    feature_names_path.parent.mkdir(parents=True, exist_ok=True)
    with open(feature_names_path, "w") as f:
        json.dump(feature_names, f)

    # Save transformed X_test (optional, for later analysis)
    transformed_X_test_path = parent_dir / "bias_mitigation" / "transformed_X_test.npz"
    scipy.sparse.save_npz(transformed_X_test_path, X_test)

    # Sensitive features used during mitigation (sex)
    sens_test = X_test_raw["sex"]

    for model_name, pkl_path in mitigated_models:
        print(f"\n{'='*60}")
        print(f"Evaluating bias-mitigated model: {model_name}")
        print(f"{'='*60}")

        if not pkl_path.exists():
            print(f"  !! Skipping {model_name}: {pkl_path} not found")
            continue

        # Load pre-trained mitigated model (no re-training here)
        try:
            model = joblib.load(pkl_path)
        except Exception as e:
            print(f"  !! Could not load {pkl_path} ({type(e).__name__}: {e}). Skipping.")
            continue

        # Convert to dense if sparse
        X_test_enc = X_test
        if scipy.sparse.issparse(X_test_enc):
            X_test_enc = X_test_enc.toarray()

        # Get predictions and probabilities
        y_pred = get_labels(model, X_test_enc, sens=sens_test)
        y_prob = get_probs(model, X_test_enc, sens=sens_test)

        # If we truly can't get probabilities, fall back to 0/1 as "probabilities"
        if y_prob is None:
            y_prob = y_pred.astype(float)

        # Save predictions + probabilities
        predictions_path = model_dir / f"{model_name}_predictions.csv"
        predictions_df = pd.DataFrame({
            "y_pred": y_pred,
            "y_prob": y_prob,
        })
        predictions_df.to_csv(predictions_path, index=False)

        # Performance metrics (same structure as baseline)
        performance = calculate_performance_metrics(
            y_true=y_test.values,
            y_pred=y_pred,
            y_prob=y_prob,
        )

        # Fairness metrics for each protected attribute
        fairness = {}
        for protected_attr in protected_attrs:
            print(f"  Evaluating with protected attribute: {protected_attr}")
            groups = prepare_groups(X_test_raw, protected_attr)
            fairness_results = evaluate_metrics(
                y_true=y_test.values,
                y_pred=y_pred,
                y_prob=y_prob,
                groups=groups,
            )

            fairness[protected_attr] = {
                "dpd": fairness_results["dpd"],
                "eod": fairness_results["eod"],
                "fprd": fairness_results["fprd"],
                "calibration": fairness_results["calibration"],
            }

        model_result = {
            "model_name": model_name,
            "performance": performance,
            "fairness": fairness,
        }
        all_results.append(model_result)

    # Save combined results JSON (parallel to baseline_results.json)
    output_path = Path(__file__).parent / "bias_mitigation_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=4)

    print(f"\n✔ All bias-mitigation metrics saved to {output_path}")
    print(f"✔ Predictions saved under {model_dir}")
