# analysis_fairness_summary_mitigated.py

import json
import pandas as pd
from pathlib import Path

# Get parent directory path
parent_dir = Path(__file__).parent.parent


def load_mitigated_results(json_path):
    """Load mitigated results JSON."""
    with open(json_path, "r") as f:
        data = json.load(f)
    return data


def build_fairness_summary(data):
    """
    Build a DataFrame:
    model | protected_attr | DPD | EOD | FPRD | ECE
    (same structure as baseline version)
    """
    rows = []

    for entry in data:
        model_name = entry["model_name"]
        fairness_dict = entry["fairness"]

        for protected_attr, metrics in fairness_dict.items():
            rows.append(
                {
                    "model": model_name,
                    "protected_attr": protected_attr,
                    "DPD": metrics["dpd"],
                    "EOD": metrics["eod"],
                    "FPRD": metrics["fprd"],
                    "ECE": metrics["calibration"]["overall_ECE"],
                }
            )

    df = pd.DataFrame(rows)
    return df


def build_performance_summary(data):
    """
    Build a DataFrame:
    model | accuracy | precision | recall | f1 | auroc
    """
    rows = []

    for entry in data:
        p = entry["performance"]
        rows.append(
            {
                "model": entry["model_name"],
                "accuracy": p["accuracy"],
                "precision": p["precision"],
                "recall": p["recall"],
                "f1": p["f1"],
                "auroc": p["auroc"],
            }
        )

    df = pd.DataFrame(rows)
    return df


if __name__ == "__main__":
    fa_dir = parent_dir / "fairness_analysis"
    fa_dir.mkdir(parents=True, exist_ok=True)

    json_path = parent_dir / "mitigated" / "mitigated_results.json"
    print(f"▶ Loading mitigated results from {json_path} ...")
    data = load_mitigated_results(json_path)

    # ---------------- Fairness summary ----------------
    print("▶ Building mitigated fairness summary table ...")
    fairness_df = build_fairness_summary(data)

    fairness_csv = fa_dir / "bias_mitigation_fairness_summary.csv"
    fairness_md = fa_dir / "bias_mitigation_fairness_summary.md"
    fairness_df.to_csv(fairness_csv, index=False)
    fairness_df.to_markdown(fairness_md, index=False)
    print(f"  Saved {fairness_csv.name} and {fairness_md.name} in {fa_dir}")

    # ---------------- Performance summary ----------------
    print("▶ Building mitigated performance summary ...")
    perf_df = build_performance_summary(data)

    perf_csv = fa_dir / "bias_mitigation_performance_summary.csv"
    perf_md = fa_dir / "bias_mitigation_performance_summary.md"
    perf_df.to_csv(perf_csv, index=False)
    perf_df.to_markdown(perf_md, index=False)
    print(f"  Saved {perf_csv.name} and {perf_md.name} in {fa_dir}")
    
    # Average fairness metrics across protected attributes per model
    avg_fair = (
        fairness_df.groupby("model")[["DPD", "EOD", "FPRD", "ECE"]]
        .mean()
        .reset_index()
    )

    combined = perf_df.merge(avg_fair, on="model")

    combined_csv = fa_dir / "bias_mitigation_fairness and performance_summary.csv"
    combined.to_csv(combined_csv, index=False)
    print(f"  Saved combined summary to {combined_csv}")

    print("\n✔ DONE. Mitigated fairness & performance summaries are ready.")