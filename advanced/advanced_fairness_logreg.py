# advanced_fairness_logreg.py
#
# Advanced Fairness–Performance Joint Optimization:
#   Train a fairness-regularized logistic regression:
#     loss = cross_entropy + lambda_fair * (DP_gap)^2
#   where DP_gap = mean(p_hat | sex=Male) - mean(p_hat | sex=Female)
#
# Outputs:
#   - advanced/models/fair_reg_logreg.pkl
#   - advanced/models/fair_reg_logreg_predictions.csv
#   - advanced/advanced_results.json  (performance + fairness metrics)

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp

# Allow importing metrics.py from project root
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))
from metrics import evaluate_metrics, calculate_performance_metrics, prepare_groups  # noqa: E402

# ---------------- Hyperparameters ----------------
LAMBDA_FAIR = 5.0       # strength of fairness penalty (tune this)
LEARNING_RATE = 0.1     # gradient descent step size
N_EPOCHS = 800          # number of training iterations
PRINT_EVERY = 100       # how often to print training progress


# Utility: make dense
# --------------------------------------------------
def ensure_dense(X):
    return X.toarray() if sp.issparse(X) else np.asarray(X)


# --------------------------------------------------
# Fairness-regularized logistic regression training
# --------------------------------------------------
def train_fair_logreg(X, y, A_binary, lambda_fair=LAMBDA_FAIR,
                      lr=LEARNING_RATE, n_epochs=N_EPOCHS):
    """
    X: (n, d) dense numpy array of features
    y: (n,) numpy array of labels in {0,1}
    A_binary: (n,) numpy array of sensitive attr in {0,1} (e.g., sex)
              0 = reference group, 1 = other group
    """
    n, d = X.shape
    # Initialize weights
    w = np.zeros(d, dtype=float)
    b = 0.0

    # Precompute group indices
    idx0 = (A_binary == 0)
    idx1 = (A_binary == 1)

    for epoch in range(1, n_epochs + 1):
        # Forward pass
        z = X @ w + b              
        p = 1.0 / (1.0 + np.exp(-z))

        # ----- Cross-entropy loss -----
        eps = 1e-8
        ce_loss = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))

        # ----- Fairness penalty: demographic parity gap -----
        loss_fair = 0.0
        grad_fair_w = np.zeros_like(w)
        grad_fair_b = 0.0

        n0 = idx0.sum()
        n1 = idx1.sum()

        if n0 > 0 and n1 > 0:
            mu0 = p[idx0].mean()
            mu1 = p[idx1].mean()
            dp_gap = mu0 - mu1   # Demographic parity gap

            # L_fair = (dp_gap)^2
            loss_fair = dp_gap ** 2

            # dL_fair/dp_i
            dL_dp = np.zeros_like(p)
            dL_dp[idx0] = 2 * dp_gap * (1.0 / n0)
            dL_dp[idx1] = 2 * dp_gap * (-1.0 / n1)

            # Chain rule through sigmoid
            grad_common = dL_dp * p * (1.0 - p)  # (n,)
            grad_fair_w = X.T @ grad_common      # (d,)
            grad_fair_b = np.sum(grad_common)

        # ----- Total loss -----
        total_loss = ce_loss + lambda_fair * loss_fair

        # ----- Gradients for CE part -----
        # dL_ce/dz = p - y
        ce_grad_common = (p - y) / n
        grad_ce_w = X.T @ ce_grad_common
        grad_ce_b = np.sum(ce_grad_common)

        # Combine gradients
        grad_w = grad_ce_w + lambda_fair * grad_fair_w
        grad_b = grad_ce_b + lambda_fair * grad_fair_b

        # Gradient descent update
        w -= lr * grad_w
        b -= lr * grad_b

        if epoch % PRINT_EVERY == 0 or epoch == 1 or epoch == n_epochs:
            print(
                f"Epoch {epoch:4d}/{n_epochs} "
                f"CE_loss={ce_loss:.4f} "
                f"Fair_loss={loss_fair:.4f} "
                f"Total={total_loss:.4f}"
            )

    return w, b


# --------------------------------------------------
# Model wrapper class
# --------------------------------------------------
class FairnessRegularizedLogReg:
    """
    Simple wrapper around trained (w, b) to look like an sklearn classifier:
      - .predict_proba(X)
      - .predict(X)
    """

    def __init__(self, w, b):
        self.coef_ = np.asarray(w).reshape(1, -1)
        self.intercept_ = np.array([b])

    def _linear_logits(self, X):
        X_dense = ensure_dense(X)
        return X_dense @ self.coef_.ravel() + self.intercept_[0]

    def predict_proba(self, X):
        z = self._linear_logits(X)
        p1 = 1.0 / (1.0 + np.exp(-z))
        p0 = 1.0 - p1
        return np.vstack([p0, p1]).T

    def predict(self, X):
        proba = self.predict_proba(X)[:, 1]
        return (proba >= 0.5).astype(int)


# --------------------------------------------------
# MAIN
# --------------------------------------------------
if __name__ == "__main__":
    # 1. Load raw splits
    X_train_raw = pd.read_csv(parent_dir / "X_train_raw.csv")
    y_train = pd.read_csv(parent_dir / "y_train.csv").squeeze()

    X_test_raw = pd.read_csv(parent_dir / "X_test_raw.csv")
    y_test = pd.read_csv(parent_dir / "y_test.csv").squeeze()

    # 2. Load preprocessor & transform
    preprocessor = joblib.load(parent_dir / "preprocessor.pkl")
    X_train_enc = preprocessor.transform(X_train_raw)
    X_test_enc = preprocessor.transform(X_test_raw)

    X_train = ensure_dense(X_train_enc)
    X_test = X_test_enc   # keep sparse/dense; wrapper will handle

    # 3. Build binary sensitive attribute for fairness (sex)
    #    We only use 'sex' for training the fairness penalty.
    sex_series = X_train_raw["sex"].astype(str).str.strip()
    # Map to binary: 1 if "Male", 0 otherwise (e.g., "Female")
    A_train = (sex_series == "Male").astype(int).to_numpy()

    # 4. Train fairness-regularized logistic regression
    print("\n=== Training fairness-regularized logistic regression (sex DP penalty) ===")
    w, b = train_fair_logreg(
        X_train,
        y_train.to_numpy(),
        A_train,
        lambda_fair=LAMBDA_FAIR,
        lr=LEARNING_RATE,
        n_epochs=N_EPOCHS,
    )

    fair_model = FairnessRegularizedLogReg(w, b)

    # 5. Evaluate on test set
    print("\n=== Evaluating fairness-regularized model on test set ===")
    y_prob = fair_model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    # 6. Save model + predictions
    model_dir = parent_dir / "advanced" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(fair_model, model_dir / "fair_reg_logreg.pkl")
    print(f"Saved model → {model_dir / 'fair_reg_logreg.pkl'}")

    pred_df = pd.DataFrame({"y_pred": y_pred, "y_prob": y_prob})
    pred_path = model_dir / "fair_reg_logreg_predictions.csv"
    pred_df.to_csv(pred_path, index=False)
    print(f"Saved predictions → {pred_path}")

    # 7. Compute performance + fairness metrics using your existing metrics.py
    protected_attrs = ["sex", "race", "age", "marital.status", "education", "relationship"]

    performance = calculate_performance_metrics(
        y_true=y_test.values,
        y_pred=y_pred,
        y_prob=y_prob,
    )

    fairness = {}
    for protected_attr in protected_attrs:
        print(f"  Evaluating fairness with protected attribute: {protected_attr}")
        groups = prepare_groups(X_test_raw, protected_attr)
        fairness_results = evaluate_metrics(
            y_true=y_test.values,
            y_pred=y_pred,
            y_prob=y_prob,
            groups=groups,
        )

        fairness[protected_attr] = {
            "dpd":         fairness_results["dpd"],
            "eod":         fairness_results["eod"],
            "fprd":        fairness_results["fprd"],
            "calibration": fairness_results["calibration"],
        }

    # 8. Save a compact JSON results file
    advanced_dir = parent_dir / "advanced"
    advanced_dir.mkdir(parents=True, exist_ok=True)

    advanced_results = [
        {
            "model_name": "fair_reg_logreg",
            "performance": performance,
            "fairness": fairness,
        }
    ]

    results_path = advanced_dir / "advanced_results.json"
    with open(results_path, "w") as f:
        json.dump(advanced_results, f, indent=4)

    print(f"\n✔ Advanced fairness-regularized results saved to {results_path}\n")
