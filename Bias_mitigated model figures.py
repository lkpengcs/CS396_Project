# fairness_analysis/plot_bias_mitigation.py

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid", context="paper", font_scale=1.1)
sns.set_palette("Set2")


BASE_DIR = os.path.dirname("fairness_analysis/")        

BASELINE_PERF_CSV = os.path.join(BASE_DIR, "baseline_performance_summary.csv")
BASELINE_FAIR_CSV = os.path.join(BASE_DIR, "baseline_fairness_summary.csv")
BIAS_MIT_CSV      = os.path.join(BASE_DIR, "bias_mitigation_fairness and performance_summary.csv")

FIG_DIR_MIT  = os.path.join(BASE_DIR, "figures_bias_mitigation")
FIG_DIR_COMP = os.path.join(BASE_DIR, "figures_comparison")

os.makedirs(FIG_DIR_MIT, exist_ok=True)
os.makedirs(FIG_DIR_COMP, exist_ok=True)

# -------------------------------------------------------------------
# 1. Load data
# -------------------------------------------------------------------
baseline_perf = pd.read_csv(BASELINE_PERF_CSV)
baseline_fair = pd.read_csv(BASELINE_FAIR_CSV)
bias_mit      = pd.read_csv(BIAS_MIT_CSV)   

# For nicer ordering (optional)
model_order_baseline = list(baseline_perf["model"].unique())
model_order_mit      = list(bias_mit["model"].unique())

baseline_perf["model"] = pd.Categorical(
    baseline_perf["model"],
    categories=model_order_baseline,
    ordered=True,
)
bias_mit["model"] = pd.Categorical(
    bias_mit["model"],
    categories=model_order_mit,
    ordered=True,
)

# Performance comparison (accuracy, f1, auroc) 
perf_metrics = ["accuracy", "f1", "auroc"]

mit_perf_long = bias_mit.melt(
    id_vars="model",
    value_vars=perf_metrics,
    var_name="Metric",
    value_name="Score",
)

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(data=mit_perf_long, x="model", y="Score", hue="Metric", ax=ax)

ax.set_ylim(0, 1.0)
ax.set_xlabel("Model")
ax.set_ylabel("Score")
ax.set_title("Model Performance Comparison (Bias-Mitigation Models)")
plt.setp(ax.get_xticklabels(), rotation=25, ha="right")

handles, labels = ax.get_legend_handles_labels()
ax.legend(
    handles,
    labels,
    title="Metric",
    loc="upper right",
    frameon=False,
)

plt.tight_layout()
plt.savefig(
    os.path.join(FIG_DIR_MIT, "performance_bias_mitigation.png"),
    dpi=300,
    bbox_inches="tight",
)
plt.close(fig)

#  Average fairness gaps by model (DPD, EOD, FPRD) 
fair_metrics = ["dpd", "eod", "fprd"]
metric_pretty = {"dpd": "DPD", "eod": "EOD", "fprd": "FPRD"}

mit_fair_long = bias_mit.melt(
    id_vars="model",
    value_vars=fair_metrics,
    var_name="Metric",
    value_name="Gap",
)
mit_fair_long["Metric"] = mit_fair_long["Metric"].map(metric_pretty)

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(data=mit_fair_long, x="model", y="Gap", hue="Metric", ax=ax)

ax.set_xlabel("Model")
ax.set_ylabel("Average Gap")
ax.set_title("Average Fairness Gaps by Model (Bias-Mitigation Models)")
plt.setp(ax.get_xticklabels(), rotation=25, ha="right")

handles, labels = ax.get_legend_handles_labels()
ax.legend(
    handles,
    labels,
    title="Metric",
    loc="upper right",
    frameon=False,
)

plt.tight_layout()
plt.savefig(
    os.path.join(FIG_DIR_MIT, "fairness_gaps_bias_mitigation.png"),
    dpi=300,
    bbox_inches="tight",
)
plt.close(fig)

# Overall calibration error by model (ECE)
fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(data=bias_mit, x="model", y="overall_ECE", ax=ax)

ax.set_xlabel("Model")
ax.set_ylabel("ECE")
ax.set_title("Overall Calibration Error (Bias-Mitigation Models)")
plt.setp(ax.get_xticklabels(), rotation=25, ha="right")

plt.tight_layout()
plt.savefig(
    os.path.join(FIG_DIR_MIT, "calibration_ECE_bias_mitigation.png"),
    dpi=300,
    bbox_inches="tight",
)
plt.close(fig)

#
# 3. COMPARISON: BASELINE VS BIAS-MITIGATED

#Performance comparison for each metric 
baseline_perf["setting"] = "baseline"
mit_perf = bias_mit[["model", "accuracy", "precision", "recall", "f1", "auroc"]].copy()
mit_perf["setting"] = "bias_mitigation"

combined_perf = pd.concat([baseline_perf, mit_perf], ignore_index=True)

pretty_metric_name = {
    "accuracy": "Accuracy",
    "f1": "F1 score",
    "auroc": "AUROC",
}

for metric in ["accuracy", "f1", "auroc"]:
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.barplot(
        data=combined_perf,
        x="model",
        y=metric,
        hue="setting",
        ax=ax,
    )

    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Model")
    ax.set_ylabel(pretty_metric_name[metric])
    ax.set_title(f"{pretty_metric_name[metric]} – Baseline vs Bias-Mitigation")
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles,
        ["baseline", "bias_mitigation"],
        title="Setting",
        loc="upper right",
        frameon=False,
    )

    plt.tight_layout()
    fname = f"{metric}_baseline_vs_bias_mitigation.png"
    plt.savefig(
        os.path.join(FIG_DIR_COMP, fname),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

# Average fairness gaps by model: baseline vs mitigated
baseline_avg_fair = (
    baseline_fair
    .groupby("model")[["DPD", "EOD", "FPRD"]]
    .mean()
    .reset_index()
)
baseline_avg_fair["setting"] = "baseline"

mit_avg_fair = bias_mit[["model", "dpd", "eod", "fprd"]].copy()
mit_avg_fair.rename(columns={"dpd": "DPD", "eod": "EOD", "fprd": "FPRD"}, inplace=True)
mit_avg_fair["setting"] = "bias_mitigation"

combined_fair = pd.concat([baseline_avg_fair, mit_avg_fair], ignore_index=True)

fair_long = combined_fair.melt(
    id_vars=["model", "setting"],
    value_vars=["DPD", "EOD", "FPRD"],
    var_name="Metric",
    value_name="Gap",
)

for metric in ["DPD", "EOD", "FPRD"]:
    sub = fair_long[fair_long["Metric"] == metric]

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=sub,
        x="model",
        y="Gap",
        hue="setting",
        ax=ax,
    )

    ax.set_xlabel("Model")
    ax.set_ylabel("Average Gap")
    ax.set_title(f"{metric} – Baseline vs Bias-Mitigation")
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles,
        ["baseline", "bias_mitigation"],
        title="Setting",
        loc="upper right",
        frameon=False,
    )

    plt.tight_layout()
    fname = f"{metric}_baseline_vs_bias_mitigation.png"
    plt.savefig(
        os.path.join(FIG_DIR_COMP, fname),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

print("Saved bias-mitigation figures to:", FIG_DIR_MIT)
print("Saved comparison figures to:", FIG_DIR_COMP)
