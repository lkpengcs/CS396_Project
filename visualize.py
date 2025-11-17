import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

os.makedirs("figures", exist_ok=True)

MODELS = ["logistic_regression", "random_forest", "gradient_boosting"]
PROTECTED_ATTRS = ["sex", "race", "age", "marital.status", "education", "relationship"]

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 11


# ============================================================
# 1. MODEL PERFORMANCE (accuracy, F1, AUROC)
# ============================================================
def plot_model_performance():
    """Plot performance metrics comparison across models."""
    # Load from baseline_results.json
    baseline_path = Path("baseline/baseline_results.json")
    if not baseline_path.exists():
        print(f"Warning: {baseline_path} not found, skipping performance plot")
        return
    
    with open(baseline_path, "r") as f:
        results = json.load(f)
    
    perf_rows = []
    for model_result in results:
        perf = model_result["performance"]
        perf_rows.append({
            "model": model_result["model_name"],
            "accuracy": perf["accuracy"],
            "f1": perf["f1"],
            "auroc": perf["auroc"]
        })
    
    df = pd.DataFrame(perf_rows)
    df.set_index("model")[["accuracy", "f1", "auroc"]].plot(
        kind="bar", 
        figsize=(10, 6),
        rot=0
    )
    plt.title("Model Performance Comparison", fontsize=14, fontweight="bold")
    plt.ylabel("Score", fontsize=12)
    plt.xlabel("Model", fontsize=12)
    plt.legend(title="Metric", fontsize=10)
    plt.tight_layout()
    plt.savefig("figures/performance_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("✓ Saved: figures/performance_comparison.png")


# ============================================================
# 2. FAIRNESS GAPS (DPD / EOD / FPRD)
# ============================================================
def plot_fairness_gaps():
    """Plot fairness gaps (DPD, EOD, FPRD) across models and protected attributes."""
    baseline_path = Path("baseline/baseline_results.json")
    if not baseline_path.exists():
        print(f"Warning: {baseline_path} not found, skipping fairness gaps plot")
        return
    
    with open(baseline_path, "r") as f:
        results = json.load(f)
    
    rows = []
    for model_result in results:
        model_name = model_result["model_name"]
        fairness = model_result["fairness"]
        for attr, metrics in fairness.items():
            rows.append({
                "model": model_name,
                "protected_attr": attr,
                "DPD": metrics["dpd"],
                "EOD": metrics["eod"],
                "FPRD": metrics["fprd"]
            })
    
    df = pd.DataFrame(rows)
    
    # Plot 1: Fairness gaps by model
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    metrics = ["DPD", "EOD", "FPRD"]
    
    for idx, metric in enumerate(metrics):
        pivot_df = df.pivot(index="protected_attr", columns="model", values=metric)
        pivot_df.plot(kind="bar", ax=axes[idx], rot=45, legend=True)
        axes[idx].set_title(f"{metric} by Protected Attribute", fontweight="bold")
        axes[idx].set_ylabel(metric)
        axes[idx].set_xlabel("Protected Attribute")
        axes[idx].legend(title="Model", fontsize=9)
    
    plt.tight_layout()
    plt.savefig("figures/fairness_gaps_by_attr.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # Plot 2: Average fairness gaps by model
    avg_fairness = df.groupby("model")[["DPD", "EOD", "FPRD"]].mean()
    avg_fairness.plot(kind="bar", figsize=(10, 6), rot=0)
    plt.title("Average Fairness Gaps by Model", fontsize=14, fontweight="bold")
    plt.ylabel("Average Gap")
    plt.xlabel("Model")
    plt.legend(title="Metric", fontsize=10)
    plt.tight_layout()
    plt.savefig("figures/fairness_gaps_by_model.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    print("✓ Saved: figures/fairness_gaps_by_attr.png")
    print("✓ Saved: figures/fairness_gaps_by_model.png")


# ============================================================
# 3. ERROR ASYMMETRY (FPR / FNR by protected attr)
# ============================================================
def plot_error_asymmetry():
    """Plot error asymmetry (FPR, FNR) for each model and protected attribute."""
    for model in MODELS:
        for attr in PROTECTED_ATTRS:
            file_path = Path(f"error_analysis/{model}/error_asymmetry_{attr}.csv")
            if not file_path.exists():
                continue
            
            df = pd.read_csv(file_path)
            
            # Create bar plot
            df_plot = df.set_index("group")[["FPR", "FNR"]]
            df_plot.plot(kind="bar", figsize=(10, 6), rot=45)
            plt.title(f"Error Asymmetry ({model}) — {attr}", fontsize=12, fontweight="bold")
            plt.ylabel("Rate")
            plt.xlabel("Group")
            plt.legend(title="Error Type", fontsize=10)
            plt.tight_layout()
            
            # Clean filename for saving
            safe_attr = attr.replace(".", "_")
            plt.savefig(f"figures/error_asymmetry_{model}_{safe_attr}.png", dpi=300, bbox_inches="tight")
            plt.close()
            print(f"✓ Saved: figures/error_asymmetry_{model}_{safe_attr}.png")


# ============================================================
# 4. THRESHOLD SENSITIVITY (accuracy vs threshold)
# ============================================================
def plot_threshold_sensitivity():
    """Plot accuracy vs threshold for each model."""
    for model in MODELS:
        file_path = Path(f"threshold_calibration/threshold_sensitivity_{model}.csv")
        if not file_path.exists():
            continue
        
        df = pd.read_csv(file_path)
        plt.figure(figsize=(10, 6))
        plt.plot(df["threshold"], df["accuracy"], marker="o", linewidth=2, markersize=6)
        plt.title(f"Accuracy vs Threshold — {model}", fontsize=12, fontweight="bold")
        plt.xlabel("Threshold", fontsize=11)
        plt.ylabel("Accuracy", fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"figures/threshold_sensitivity_{model}.png", dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✓ Saved: figures/threshold_sensitivity_{model}.png")


# ============================================================
# 5. CALIBRATION CURVES (all models combined)
# ============================================================
def plot_calibration_curves():
    """Combine calibration curve images for all models."""
    import matplotlib.image as mpimg
    
    images = []
    model_names = []
    for model in MODELS:
        img_path = Path(f"threshold_calibration/calibration_curve_{model}.png")
        if img_path.exists():
            images.append(mpimg.imread(str(img_path)))
            model_names.append(model)
    
    if not images:
        print("Warning: No calibration curve images found")
        return
    
    # Create a grid layout
    n = len(images)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 6*rows))
    if n == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, img in enumerate(images):
        axes[idx].imshow(img)
        axes[idx].axis("off")
        axes[idx].set_title(model_names[idx].replace("_", " ").title(), fontsize=12, fontweight="bold")
    
    # Hide extra subplots
    for idx in range(len(images), len(axes)):
        axes[idx].axis("off")
    
    plt.tight_layout()
    plt.savefig("figures/calibration_curves_combined.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("✓ Saved: figures/calibration_curves_combined.png")


# ============================================================
# 6. COUNTERFACTUAL FLIP RATE (CF Gap)
# ============================================================
def plot_counterfactual_summary():
    """Plot counterfactual fairness gap (CF Gap) by model and protected attribute."""
    rows = []
    for model in MODELS:
        file_path = Path(f"counterfactual_analysis/cf_summary_{model}.csv")
        if not file_path.exists():
            continue
        
        df = pd.read_csv(file_path)
        for _, row in df.iterrows():
            rows.append({
                "model": model,
                "protected_attr": row["protected_attr"],
                "cf_gap": row["cf_gap"]
            })
    
    if not rows:
        print("Warning: No counterfactual summary files found")
        return
    
    df = pd.DataFrame(rows)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x="protected_attr", y="cf_gap", hue="model", palette="Set2")
    plt.title("Counterfactual Fairness Gap (CF Gap)", fontsize=14, fontweight="bold")
    plt.ylabel("Flip Rate Gap", fontsize=11)
    plt.xlabel("Protected Attribute", fontsize=11)
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Model", fontsize=10)
    plt.tight_layout()
    plt.savefig("figures/counterfactual_gap.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("✓ Saved: figures/counterfactual_gap.png")


# ============================================================
# 7. FEATURE IMPORTANCE (Top 20)
# ============================================================
def plot_feature_importance():
    """Plot top 5 feature importance for each model."""
    for model in MODELS:
        file_path = Path(f"error_analysis/{model}/feature_importance.csv")
        if not file_path.exists():
            continue
        
        df = pd.read_csv(file_path)
        df = df.sort_values("importance", ascending=False).head(5)
        
        plt.figure(figsize=(10, 8))
        sns.barplot(data=df, x="importance", y="feature_group", palette="viridis")
        plt.title(f"Top 5 Feature Importance — {model}", fontsize=12, fontweight="bold")
        plt.xlabel("Importance", fontsize=11)
        plt.ylabel("Feature", fontsize=11)
        plt.tight_layout()
        plt.savefig(f"figures/feature_importance_{model}.png", dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✓ Saved: figures/feature_importance_{model}.png")


# ============================================================
# 8. CALIBRATION BY GROUP (ECE by protected attribute)
# ============================================================
def plot_calibration_by_group():
    """Plot Expected Calibration Error (ECE) by group for each model and protected attribute."""
    baseline_path = Path("baseline/baseline_results.json")
    if not baseline_path.exists():
        print(f"Warning: {baseline_path} not found, skipping calibration plot")
        return
    
    with open(baseline_path, "r") as f:
        results = json.load(f)
    
    for model_result in results:
        model_name = model_result["model_name"]
        fairness = model_result["fairness"]
        
        # Collect ECE data
        rows = []
        for attr, metrics in fairness.items():
            calibration = metrics["calibration"]
            overall_ece = calibration.get("overall_ECE", 0)
            rows.append({
                "protected_attr": attr,
                "group": "Overall",
                "ECE": overall_ece
            })
            
            # Add per-group ECE
            for group, ece in calibration.items():
                if group != "overall_ECE":
                    rows.append({
                        "protected_attr": attr,
                        "group": group,
                        "ECE": ece
                    })
        
        df = pd.DataFrame(rows)
        
        # Plot overall ECE by protected attribute
        overall_df = df[df["group"] == "Overall"]
        if not overall_df.empty:
            plt.figure(figsize=(10, 6))
            sns.barplot(data=overall_df, x="protected_attr", y="ECE", palette="coolwarm")
            plt.title(f"Overall ECE by Protected Attribute — {model_name}", fontsize=12, fontweight="bold")
            plt.ylabel("Expected Calibration Error", fontsize=11)
            plt.xlabel("Protected Attribute", fontsize=11)
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            safe_model = model_name.replace("_", "_")
            plt.savefig(f"figures/calibration_overall_{safe_model}.png", dpi=300, bbox_inches="tight")
            plt.close()
            print(f"✓ Saved: figures/calibration_overall_{safe_model}.png")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Generating all visualizations...")
    print("=" * 60)
    
    plot_model_performance()
    plot_fairness_gaps()
    plot_error_asymmetry()
    plot_threshold_sensitivity()
    plot_calibration_curves()
    plot_counterfactual_summary()
    plot_feature_importance()
    plot_calibration_by_group()
    
    print("\n" + "=" * 60)
    print("✔ All figures saved in ./figures/")
    print("=" * 60)
