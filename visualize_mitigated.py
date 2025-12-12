import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# New folder for mitigated figures
FIG_DIR = Path("figures_mitigated")
FIG_DIR.mkdir(exist_ok=True)

# Mitigated models (display_name from mitigated_results.json)
MITIGATED_MODELS = ["reweigh", "expgrad", "cal_eq_odds"]
PROTECTED_ATTRS = ["sex", "race", "age", "marital.status", "education", "relationship"]

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 11


# ============================================================
# 1. MODEL PERFORMANCE (accuracy, F1, AUROC) – MITIGATED
# ============================================================
def plot_model_performance_mitigated():
    """Plot performance metrics comparison across mitigated models."""
    mitigated_path = Path("mitigated/mitigated_results.json")
    if not mitigated_path.exists():
        print(f"Warning: {mitigated_path} not found, skipping mitigated performance plot")
        return

    with open(mitigated_path, "r") as f:
        results = json.load(f)

    perf_rows = []
    for model_result in results:
        perf = model_result["performance"]
        perf_rows.append(
            {
                "model": model_result["model_name"],
                "accuracy": perf["accuracy"],
                "f1": perf["f1"],
                "auroc": perf["auroc"],
            }
        )

    df = pd.DataFrame(perf_rows)
    ax = df.set_index("model")[["accuracy", "f1", "auroc"]].plot(
        kind="bar",
        figsize=(10, 6),
        rot=0,
    )
    plt.title("Mitigated Model Performance Comparison", fontsize=14, fontweight="bold")
    plt.ylabel("Score", fontsize=12)
    plt.xlabel("Mitigated Model", fontsize=12)
    plt.ylim(0, 1.0)
    plt.legend(title="Metric", fontsize=10)
    plt.tight_layout()
    out_path = FIG_DIR / "mitigated_performance_comparison.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: {out_path}")


# ============================================================
# 2. FAIRNESS GAPS (DPD / EOD / FPRD) – MITIGATED
# ============================================================
def plot_fairness_gaps_mitigated():
    """Plot fairness gaps (DPD, EOD, FPRD) for mitigated models."""
    mitigated_path = Path("mitigated/mitigated_results.json")
    if not mitigated_path.exists():
        print(f"Warning: {mitigated_path} not found, skipping mitigated fairness gaps plot")
        return

    with open(mitigated_path, "r") as f:
        results = json.load(f)

    rows = []
    for model_result in results:
        model_name = model_result["model_name"]
        fairness = model_result["fairness"]
        for attr, metrics in fairness.items():
            rows.append(
                {
                    "model": model_name,
                    "protected_attr": attr,
                    "DPD": metrics["dpd"],
                    "EOD": metrics["eod"],
                    "FPRD": metrics["fprd"],
                }
            )

    df = pd.DataFrame(rows)

    # Plot 1: Fairness gaps by protected attribute
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    metrics = ["DPD", "EOD", "FPRD"]

    for idx, metric in enumerate(metrics):
        pivot_df = df.pivot(index="protected_attr", columns="model", values=metric)
        pivot_df.plot(kind="bar", ax=axes[idx], rot=45, legend=True)
        axes[idx].set_title(f"{metric} by Protected Attribute (Mitigated)", fontweight="bold")
        axes[idx].set_ylabel(metric)
        axes[idx].set_xlabel("Protected Attribute")
        axes[idx].legend(title="Mitigated Model", fontsize=9)

    plt.tight_layout()
    out_path1 = FIG_DIR / "mitigated_fairness_gaps_by_attr.png"
    plt.savefig(out_path1, dpi=300, bbox_inches="tight")
    plt.close()

    # Plot 2: Average fairness gaps by mitigated model
    avg_fairness = df.groupby("model")[["DPD", "EOD", "FPRD"]].mean()
    ax = avg_fairness.plot(kind="bar", figsize=(10, 6), rot=0)
    plt.title("Average Fairness Gaps by Mitigated Model", fontsize=14, fontweight="bold")
    plt.ylabel("Average Gap")
    plt.xlabel("Mitigated Model")
    plt.legend(title="Metric", fontsize=10)
    plt.tight_layout()
    out_path2 = FIG_DIR / "mitigated_fairness_gaps_by_model.png"
    plt.savefig(out_path2, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✓ Saved: {out_path1}")
    print(f"✓ Saved: {out_path2}")


# ============================================================
# 3. ERROR ASYMMETRY (FPR / FNR by protected attr) – MITIGATED
# ============================================================
def plot_error_asymmetry_mitigated():
    """Plot error asymmetry (FPR, FNR) for each mitigated model and protected attribute."""
    for model in MITIGATED_MODELS:
        for attr in PROTECTED_ATTRS:
            file_path = Path(f"error_analysis/{model}/error_asymmetry_{attr}.csv")
            if not file_path.exists():
                continue

            df = pd.read_csv(file_path)

            df_plot = df.set_index("group")[["FPR", "FNR"]]
            ax = df_plot.plot(kind="bar", figsize=(10, 6), rot=45)
            plt.title(f"Error Asymmetry (Mitigated: {model}) — {attr}", fontsize=12, fontweight="bold")
            plt.ylabel("Rate")
            plt.xlabel("Group")
            plt.legend(title="Error Type", fontsize=10)
            plt.tight_layout()

            safe_attr = attr.replace(".", "_")
            out_path = FIG_DIR / f"mitigated_error_asymmetry_{model}_{safe_attr}.png"
            plt.savefig(out_path, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"✓ Saved: {out_path}")


# ============================================================
# 4. COUNTERFACTUAL FLIP RATE (CF Gap) – MITIGATED
# ============================================================
def plot_counterfactual_summary_mitigated():
    """
    Plot counterfactual fairness gap (CF Gap) by mitigated model and protected attribute.
    Uses cf_summary_mitigated_<model>.csv written earlier.
    """
    rows = []
    for model in MITIGATED_MODELS:
        file_path = Path(f"counterfactual_analysis/cf_summary_mitigated_{model}.csv")
        if not file_path.exists():
            continue

        df = pd.read_csv(file_path)
        for _, row in df.iterrows():
            rows.append(
                {
                    "model": model,
                    "protected_attr": row["protected_attr"],
                    "cf_gap": row["cf_gap"],
                }
            )

    if not rows:
        print("Warning: No mitigated counterfactual summary files found")
        return

    df = pd.DataFrame(rows)

    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x="protected_attr", y="cf_gap", hue="model", palette="Set2")
    plt.title("Counterfactual Fairness Gap (CF Gap) – Mitigated Models", fontsize=14, fontweight="bold")
    plt.ylabel("Flip Rate Gap", fontsize=11)
    plt.xlabel("Protected Attribute", fontsize=11)
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Mitigated Model", fontsize=10)
    plt.tight_layout()
    out_path = FIG_DIR / "mitigated_counterfactual_gap.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: {out_path}")


# ============================================================
# 5. FEATURE IMPORTANCE (Top 5) – MITIGATED
# ============================================================
def plot_feature_importance_mitigated():
    """Plot top 5 grouped feature importance for each mitigated model (where available)."""
    for model in MITIGATED_MODELS:
        file_path = Path(f"error_analysis/{model}/feature_importance.csv")
        if not file_path.exists():
            continue

        df = pd.read_csv(file_path)
        df = df.sort_values("importance", ascending=False).head(5)

        plt.figure(figsize=(10, 8))
        sns.barplot(data=df, x="importance", y="feature_group", palette="viridis")
        plt.title(f"Top 5 Feature Importance — Mitigated {model}", fontsize=12, fontweight="bold")
        plt.xlabel("Importance", fontsize=11)
        plt.ylabel("Feature Group", fontsize=11)
        plt.tight_layout()
        out_path = FIG_DIR / f"mitigated_feature_importance_{model}.png"
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✓ Saved: {out_path}")


# ============================================================
# 6. CALIBRATION BY GROUP (ECE by protected attribute) – MITIGATED
# ============================================================
def plot_calibration_by_group_mitigated():
    """Plot Expected Calibration Error (ECE) by protected attribute for mitigated models."""
    mitigated_path = Path("mitigated/mitigated_results.json")
    if not mitigated_path.exists():
        print(f"Warning: {mitigated_path} not found, skipping mitigated calibration plot")
        return

    with open(mitigated_path, "r") as f:
        results = json.load(f)

    for model_result in results:
        model_name = model_result["model_name"]
        fairness = model_result["fairness"]

        rows = []
        for attr, metrics in fairness.items():
            calibration = metrics["calibration"]
            overall_ece = calibration.get("overall_ECE", 0)
            rows.append(
                {
                    "protected_attr": attr,
                    "group": "Overall",
                    "ECE": overall_ece,
                }
            )

            for group, ece in calibration.items():
                if group != "overall_ECE":
                    rows.append(
                        {
                            "protected_attr": attr,
                            "group": group,
                            "ECE": ece,
                        }
                    )

        df = pd.DataFrame(rows)

        overall_df = df[df["group"] == "Overall"]
        if not overall_df.empty:
            plt.figure(figsize=(10, 6))
            sns.barplot(data=overall_df, x="protected_attr", y="ECE", palette="coolwarm")
            plt.title(
                f"Overall ECE by Protected Attribute — Mitigated {model_name}",
                fontsize=12,
                fontweight="bold",
            )
            plt.ylabel("Expected Calibration Error", fontsize=11)
            plt.xlabel("Protected Attribute", fontsize=11)
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            safe_model = model_name.replace("_", "_")
            out_path = FIG_DIR / f"mitigated_calibration_overall_{safe_model}.png"
            plt.savefig(out_path, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"✓ Saved: {out_path}")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Generating all MITIGATED visualizations...")
    print("=" * 60)

    plot_model_performance_mitigated()
    plot_fairness_gaps_mitigated()
    plot_error_asymmetry_mitigated()
    plot_counterfactual_summary_mitigated()
    plot_feature_importance_mitigated()
    plot_calibration_by_group_mitigated()

    print("\n" + "=" * 60)
    print("✔ All mitigated figures saved in ./figures_mitigated/")
    print("=" * 60)
