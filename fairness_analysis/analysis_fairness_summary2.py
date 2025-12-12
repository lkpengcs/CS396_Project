# analysis_fairness_summary.py
import json
import pandas as pd
import sys
from pathlib import Path
# Get parent directory path
parent_dir = Path(__file__).parent.parent

def load_baseline_results(json_path="baseline_results_compact.json"):
    """Load compact baseline JSON."""
    with open(json_path, "r") as f:
        data = json.load(f)
    return data


def build_fairness_summary(data):
    """
    Build a DataFrame:
    Model | Protected Attribute | DPD | EOD | FPRD | ECE
    """
    rows = []

    for entry in data:
        model_name = entry["model_name"]
        fairness_dict = entry["fairness"]

        for protected_attr, metrics in fairness_dict.items():
            rows.append({
                "model": model_name,
                "protected_attr": protected_attr,
                "DPD": metrics["dpd"],
                "EOD": metrics["eod"],
                "FPRD": metrics["fprd"],
                "ECE": metrics["calibration"]["overall_ECE"]
            })

    df = pd.DataFrame(rows)
    return df


def build_performance_summary(data):
    """
    Build a DataFrame:
    Model | accuracy | precision | recall | f1 | auroc
    """
    rows = []

    for entry in data:
        p = entry["performance"]
        rows.append({
            "model": entry["model_name"],
            "accuracy": p["accuracy"],
            "precision": p["precision"],
            "recall": p["recall"],
            "f1": p["f1"],
            "auroc": p["auroc"]
        })

    df = pd.DataFrame(rows)
    return df


if __name__ == "__main__":

    print(f"▶ Loading compact baseline results ...")
    data = load_baseline_results(parent_dir / "advanced" / "advanced_results.json")

    print(f"▶ Building fairness summary table ...")
    fairness_df = build_fairness_summary(data)
    fairness_df.to_csv(parent_dir / "fairness_analysis" / "fairness_summary2.csv", index=False)

    # Also save to markdown (good for reports)
    fairness_df.to_markdown(parent_dir / "fairness_analysis" / "fairness_summary2.md", index=False)

    print(f"Saved fairness_summary2.csv and fairness_summary2.md to {parent_dir / 'fairness_analysis'}")

    print(f"▶ Building performance summary ...")
    perf_df = build_performance_summary(data)
    perf_df.to_csv(parent_dir / "fairness_analysis" / "performance_summary2.csv", index=False)
    perf_df.to_markdown(parent_dir / "fairness_analysis" / "performance_summary2.md", index=False)

    print(f"Saved performance_summary2.csv and performance_summary2.md to {parent_dir} / fairness_analysis /")

    print("\n✔ DONE. You can now inspect the summary tables!")
