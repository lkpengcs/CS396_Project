import json
import sys
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

PARENT = Path(__file__).resolve().parent.parent
sys.path.append(str(PARENT))
from metrics import evaluate_metrics

SCRIPT_DIR = Path(__file__).resolve().parent
FIG_DIR = SCRIPT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

RESULT_PATHS = [
    PARENT / "baseline" / "baseline_results.json",
    PARENT / "mitigated" / "mitigated_results.json",
    PARENT / "advanced" / "advanced_results.json",
]

PRED_PATHS = {
    "baseline_logistic_regression": PARENT / "baseline" / "models" / "logistic_regression_predictions.csv",
    "baseline_random_forest": PARENT / "baseline" / "models" / "random_forest_predictions.csv",
    "baseline_gradient_boosting": PARENT / "baseline" / "models" / "gradient_boosting_predictions.csv",
    "advanced_fair_reg_logreg": PARENT / "advanced" / "models" / "fair_reg_logreg_predictions.csv",
    "mitigated_reweigh": PARENT / "mitigated" / "models" / "reweigh_predictions.csv",
    "mitigated_expgrad": PARENT / "mitigated" / "models" / "expgrad_predictions.csv",
    "mitigated_cal_eq_odds": PARENT / "mitigated" / "models" / "cal_eq_odds_predictions.csv",
}

ATTR_COMBOS = [
    ("sex", "race"),
    ("sex", "age"),
    ("race", "education"),
    ("sex", "relationship"),
    ("marital.status", "education"),
]


# -------------------- Fairness trade-offs --------------------
def load_fairness_rows():
    rows = []
    for path in RESULT_PATHS:
        if not path.exists():
            continue
        run_tag = path.parent.name
        with open(path, "r") as f:
            data = json.load(f)
        for entry in data:
            perf = entry["performance"]
            for attr, metrics in entry["fairness"].items():
                rows.append(
                    {
                        "run": run_tag,
                        "model": entry["model_name"],
                        "attr": attr,
                        "DPD": metrics["dpd"],
                        "EOD": metrics["eod"],
                        "FPRD": metrics["fprd"],
                        "ECE": metrics["calibration"]["overall_ECE"],
                        "accuracy": perf["accuracy"],
                        "precision": perf["precision"],
                        "recall": perf["recall"],
                        "f1": perf["f1"],
                        "auroc": perf["auroc"],
                    }
                )
    return pd.DataFrame(rows)


def plot_fairness_pairplot(df: pd.DataFrame) -> None:
    subset = df[["run", "DPD", "EOD", "FPRD", "ECE"]].copy()
    sns.pairplot(subset, vars=["DPD", "EOD", "FPRD", "ECE"], hue="run", corner=True)
    plt.savefig(FIG_DIR / "fairness_pairplot.png", dpi=200, bbox_inches="tight")
    plt.close("all")


def plot_fairness_vs_performance(df: pd.DataFrame) -> None:
    metrics = ["DPD", "EOD", "FPRD", "ECE"]
    for metric in metrics:
        ax = sns.scatterplot(
            data=df,
            x=metric,
            y="accuracy",
            hue="run",
            style="model",
        )
        ax.set_title(f"Accuracy vs {metric}")
        plt.savefig(FIG_DIR / f"accuracy_vs_{metric}.png", dpi=200, bbox_inches="tight")
        plt.close("all")


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    cols = ["DPD", "EOD", "FPRD", "ECE", "accuracy", "f1", "auroc"]
    corr = df[cols].corr(method="spearman")
    sns.heatmap(corr, annot=True, vmin=-1, vmax=1, cmap="coolwarm")
    plt.title("Fairness vs Performance (Spearman)")
    plt.savefig(FIG_DIR / "fairness_perf_correlation.png", dpi=200, bbox_inches="tight")
    plt.close("all")


# -------------------- Intersectional fairness --------------------
def bin_age(series: pd.Series) -> pd.Series:
    return pd.cut(series, bins=[0, 30, 50, 100], labels=["young", "middle", "old"]).astype(str)


def make_groups(df: pd.DataFrame, attrs) -> pd.Series:
    cols = []
    for attr in attrs:
        if attr == "age":
            cols.append(bin_age(df[attr]))
        else:
            cols.append(df[attr].astype(str))
    combo = pd.concat(cols, axis=1)
    return combo.apply(lambda row: " | ".join([f"{a}={v}" for a, v in zip(attrs, row)]), axis=1)


def load_predictions(model_key: str, path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing predictions for {model_key}: {path}")
    df = pd.read_csv(path)
    if not {"y_pred", "y_prob"}.issubset(df.columns):
        raise ValueError(f"Prediction file for {model_key} must have y_pred and y_prob columns")
    return df


def evaluate_intersectional_fairness(x_raw: pd.DataFrame, y_true: pd.Series) -> pd.DataFrame:
    rows = []
    for model_key, path in PRED_PATHS.items():
        if not path.exists():
            continue
        preds = load_predictions(model_key, path)
        y_pred = preds["y_pred"].values
        y_prob = preds["y_prob"].values
        for attrs in ATTR_COMBOS:
            groups = make_groups(x_raw, attrs).values
            metrics = evaluate_metrics(y_true.values, y_pred, y_prob, groups)
            rows.append(
                {
                    "model": model_key,
                    "attrs": "+".join(attrs),
                    "DPD": metrics["dpd"],
                    "EOD": metrics["eod"],
                    "FPRD": metrics["fprd"],
                    "ECE": metrics["calibration"]["overall_ECE"],
                }
            )
    return pd.DataFrame(rows)


def plot_intersectional(df: pd.DataFrame) -> None:
    if df.empty:
        return
    ax = sns.scatterplot(data=df, x="FPRD", y="EOD", hue="attrs", style="model")
    ax.set_title("Intersectional: FPRD vs EOD")
    plt.savefig(FIG_DIR / "intersectional_fprd_eod.png", dpi=200, bbox_inches="tight")
    plt.close("all")

    ax = sns.scatterplot(data=df, x="DPD", y="ECE", hue="attrs", style="model")
    ax.set_title("Intersectional: DPD vs Calibration")
    plt.savefig(FIG_DIR / "intersectional_dpd_ece.png", dpi=200, bbox_inches="tight")
    plt.close("all")


def main() -> None:
    sns.set(style="whitegrid")

    # Fairness trade-offs across runs
    fairness_df = load_fairness_rows()
    fairness_df.to_csv(SCRIPT_DIR / "fairness_tradeoffs_long.csv", index=False)
    if not fairness_df.empty:
        plot_fairness_pairplot(fairness_df)
        plot_fairness_vs_performance(fairness_df)
        plot_correlation_heatmap(fairness_df)

    # Intersectional fairness
    x_raw = pd.read_csv(PARENT / "X_test_raw.csv")
    y_true = pd.read_csv(PARENT / "y_test.csv")["income"]
    inter_df = evaluate_intersectional_fairness(x_raw, y_true)
    inter_df.to_csv(SCRIPT_DIR / "intersectional_fairness.csv", index=False)
    plot_intersectional(inter_df)


if __name__ == "__main__":
    main()