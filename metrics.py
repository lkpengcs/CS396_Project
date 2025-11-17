# metrics.py
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# -------------------------------------
# Utility: Prepare groups for fairness evaluation
# -------------------------------------
def prepare_groups(X_test_raw, protected_attr):
    """
    Prepare groups for fairness evaluation.
    For numeric attributes like age, bin them into categories.
    
    Parameters:
    -----------
    X_test_raw : pandas.DataFrame
        Raw test data with protected attributes
    protected_attr : str
        Name of the protected attribute column
        
    Returns:
    --------
    groups : array-like
        Group labels for each sample
    """
    if protected_attr == "age":
        # Bin age into categories
        age = X_test_raw[protected_attr].values
        groups = pd.cut(age, bins=[0, 30, 50, 100], labels=["young", "middle", "old"]).astype(str)
    else:
        # For categorical attributes, use as-is
        groups = X_test_raw[protected_attr].values
    return groups


# -------------------------------------
# Performance Metrics
# -------------------------------------
def calculate_performance_metrics(y_true, y_pred, y_prob):
    """
    Calculate standard classification performance metrics.
    
    Parameters:
    -----------
    y_true : array-like
        True binary labels
    y_pred : array-like
        Predicted binary labels
    y_prob : array-like
        Predicted probabilities for positive class
        
    Returns:
    --------
    dict : Dictionary containing performance metrics
    """
    performance = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "auroc": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) == 2 else None
    }
    return performance


# -------------------------------------
# Utility: Expected Calibration Error
# -------------------------------------
def expected_calibration_error(y_true, y_prob, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        start, end = bins[i], bins[i+1]
        # Include endpoint for the last bin
        if i == n_bins - 1:
            idx = (y_prob >= start) & (y_prob <= end)
        else:
            idx = (y_prob >= start) & (y_prob < end)
        if idx.sum() > 0:
            bin_acc = y_true[idx].mean()
            bin_conf = y_prob[idx].mean()
            ece += (idx.mean()) * abs(bin_acc - bin_conf)
    return ece


# -------------------------------------
# Group Fairness Metrics
# -------------------------------------
def demographic_parity_difference(y_pred, groups):
    """Difference in positive prediction rates."""
    rates = []
    for g in np.unique(groups):
        idx = (groups == g)
        rates.append(y_pred[idx].mean())
    return np.max(rates) - np.min(rates)


def equal_opportunity_difference(y_true, y_pred, groups):
    """Difference in True Positive Rates (TPR)."""
    tprs = []
    for g in np.unique(groups):
        idx = (groups == g)
        positives = (y_true[idx] == 1)
        if positives.sum() == 0:
            continue
        tpr = (y_pred[idx][positives] == 1).mean()
        tprs.append(tpr)
    if len(tprs) == 0:
        return np.nan  # No group has positive samples
    return np.max(tprs) - np.min(tprs)


def false_positive_rate_difference(y_true, y_pred, groups):
    """Difference in False Positive Rates (FPR)."""
    fprs = []
    for g in np.unique(groups):
        idx = (groups == g)
        negatives = (y_true[idx] == 0)
        if negatives.sum() == 0:
            continue
        fpr = (y_pred[idx][negatives] == 1).mean()
        fprs.append(fpr)
    if len(fprs) == 0:
        return np.nan  # No group has negative samples
    return np.max(fprs) - np.min(fprs)


def calibration_by_group(y_true, y_prob, groups, n_bins=10):
    """Return ECE + per-group calibration scores."""
    results = {}
    overall_ece = expected_calibration_error(y_true, y_prob, n_bins)
    results["overall_ECE"] = overall_ece

    for g in np.unique(groups):
        idx = (groups == g)
        results[g] = expected_calibration_error(
            y_true[idx], y_prob[idx], n_bins
        )
    return results


# -------------------------------------
# Main evaluation function
# -------------------------------------
def evaluate_metrics(y_true, y_pred, y_prob, groups):
    """
    y_true: array of 0/1 labels
    y_pred: predicted labels 0/1
    y_prob: predicted probabilities (for AUROC, calibration)
    groups: protected attribute (e.g., sex or race)
    """

    results = {}

    # Group fairness metrics
    results["dpd"] = demographic_parity_difference(y_pred, groups)
    results["eod"] = equal_opportunity_difference(y_true, y_pred, groups)
    results["fprd"] = false_positive_rate_difference(y_true, y_pred, groups)

    # Calibration by group
    results["calibration"] = calibration_by_group(y_true, y_prob, groups)

    return results
