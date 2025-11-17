import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import brier_score_loss

import joblib
import scipy.sparse
from pathlib import Path
# Get parent directory path
parent_dir = Path(__file__).parent.parent

# === Utility functions ===

def compute_group_metrics(y_true, y_pred_bin, X_raw, protected_attr):
    groups = X_raw[protected_attr].unique()
    results = []

    for g in groups:
        idx = (X_raw[protected_attr] == g)
        acc = accuracy_score(y_true[idx], y_pred_bin[idx])
        results.append({"group": g, "accuracy": acc})

    return pd.DataFrame(results)


def calibration_error(y_true, y_prob, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins) - 1
    ece = 0.0
    for i in range(n_bins):
        idx = (bin_ids == i)
        if np.sum(idx) > 0:
            ece += np.abs(np.mean(y_prob[idx]) - np.mean(y_true[idx])) * (np.sum(idx) / len(y_prob))
    return ece


# === MAIN ANALYSIS ===
if __name__ == "__main__":
    print("▶ Loading baseline cached data...")

    X_test_raw = pd.read_csv("X_test_raw.csv")
    y_test = pd.read_csv("y_test.csv").squeeze()

    protected_attrs = ["sex", "race", "age", "marital.status", "education", "relationship"]
    models = ["logistic_regression", "random_forest", "gradient_boosting"]

    for name in models:
        print(f"\n=== Threshold + Calibration Analysis for {name} ===")

        # Load stored model outputs
        model = joblib.load(parent_dir / "baseline" / "models" / f"{name}.pkl")
        predictions = pd.read_csv(parent_dir / "baseline" / "models" / f"{name}_predictions.csv")
        y_pred = predictions["y_pred"]
        y_prob = predictions["y_prob"]
        X_test_enc = scipy.sparse.load_npz(parent_dir / "baseline" / "transformed_X_test.npz")
        X_test_enc = X_test_enc.toarray()

        # ============================
        # 1. Threshold sensitivity
        # ============================

        thresholds = np.linspace(0.0, 1.0, 21)
        sensitivity_rows = []

        for th in thresholds:
            y_bin = (y_prob >= th).astype(int)
            acc = accuracy_score(y_test, y_bin)
            auc = roc_auc_score(y_test, y_prob)

            sensitivity_rows.append({"threshold": th, "accuracy": acc})

        df_sens = pd.DataFrame(sensitivity_rows)
        df_sens.to_csv(parent_dir / "threshold_calibration" / f"threshold_sensitivity_{name}.csv", index=False)

        # Plot accuracy vs threshold
        plt.figure()
        plt.plot(df_sens["threshold"], df_sens["accuracy"], marker="o")
        plt.xlabel("Threshold")
        plt.ylabel("Accuracy")
        plt.title(f"Accuracy vs Threshold ({name})")
        plt.grid(True)
        plt.savefig(parent_dir / "threshold_calibration" / f"threshold_accuracy_plot_{name}.png")
        plt.close()

        # ============================
        # 2. Calibration curves
        # ============================

        prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10, strategy="uniform")

        plt.figure()
        plt.plot(prob_pred, prob_true, marker="o")
        plt.plot([0, 1], [0, 1], "k--")  # perfect calibration
        plt.xlabel("Predicted probability")
        plt.ylabel("True frequency")
        plt.title(f"Calibration Curve ({name})")
        plt.grid(True)
        plt.savefig(parent_dir / "threshold_calibration" / f"calibration_curve_{name}.png")
        plt.close()

        # Compute ECE
        ece_val = calibration_error(y_test, y_prob)
        with open(parent_dir / "threshold_calibration" / f"calibration_metrics_{name}.txt", "w") as f:
            f.write(f"ECE: {ece_val:.5f}\n")
            f.write(f"Brier score: {brier_score_loss(y_test, y_prob):.5f}\n")

        # ============================
        # 3. Group-wise calibration
        # ============================

        for attr in protected_attrs:
            groups = X_test_raw[attr]
            if attr == "age":
                groups = pd.cut(groups, bins=[0, 30, 50, 100], labels=["young", "middle", "old"])
            else:
                groups = groups.astype(str)
            groups = groups.unique()
            rows = []
            for g in groups:
                idx = (X_test_raw[attr] == g)
                ece_g = calibration_error(y_test[idx], y_prob[idx])
                rows.append({"group": g, "ECE": ece_g})
            df_gcal = pd.DataFrame(rows)
            df_gcal.to_csv(parent_dir / "threshold_calibration" / f"group_calibration_{name}_{attr}.csv", index=False)

    print("\n✔ Step C complete: saved threshold_sensitivity + calibration metrics + plots.")
