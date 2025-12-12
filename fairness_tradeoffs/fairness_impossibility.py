"""
Analyze fairness trade-offs rooted in the “impossibility” results:
- When Equalized Odds (low EOD) is enforced, does calibration degrade?
- Quantify calibration drift (range/std across groups) as EOD is reduced.
Outputs: CSV + markdown summary + printed stats.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path

PARENT = Path(__file__).resolve().parent.parent

RESULT_PATHS = [
    PARENT / "baseline" / "baseline_results.json",
    PARENT / "mitigated" / "mitigated_results.json",
    PARENT / "advanced" / "advanced_results.json",
]


def load_rows() -> pd.DataFrame:
    rows = []
    for path in RESULT_PATHS:
        if not path.exists():
            continue
        run_tag = path.parent.name
        with open(path, "r") as f:
            data = json.load(f)
        for entry in data:
            perf = entry["performance"]
            model_name = entry["model_name"]
            eo_enforced = "cal_eq_odds" in model_name  # proxy for explicit EO mitigation
            for attr, metrics in entry["fairness"].items():
                calib = metrics["calibration"]
                group_ece = [v for k, v in calib.items() if k != "overall_ECE"]
                calib_range = np.ptp(group_ece) if group_ece else np.nan
                calib_std = np.std(group_ece) if group_ece else np.nan
                rows.append(
                    {
                        "run": run_tag,
                        "model": model_name,
                        "attr": attr,
                        "EO_enforced": eo_enforced,
                        "DPD": metrics["dpd"],
                        "EOD": metrics["eod"],
                        "FPRD": metrics["fprd"],
                        "ECE_overall": calib["overall_ECE"],
                        "ECE_range": calib_range,
                        "ECE_std": calib_std,
                        "accuracy": perf["accuracy"],
                        "f1": perf["f1"],
                        "auroc": perf["auroc"],
                    }
                )
    return pd.DataFrame(rows)


def compute_impossibility_gap(df: pd.DataFrame) -> pd.DataFrame:
    """
    Define an "impossibility gap" as the extra calibration dispersion
    (range across groups) incurred when EOD is pushed to its lower quartile.
    """
    q_low = df["EOD"].quantile(0.25)
    q_high = df["EOD"].quantile(0.75)
    low_eod = df[df["EOD"] <= q_low]
    high_eod = df[df["EOD"] >= q_high]
    gap = low_eod["ECE_range"].mean() - high_eod["ECE_range"].mean()
    return pd.DataFrame(
        {
            "metric": ["impossibility_gap_ECE_range"],
            "description": [
                "Mean calibration range (low EOD quartile) minus mean range (high EOD quartile)"
            ],
            "value": [gap],
            "low_eod_threshold": [q_low],
            "high_eod_threshold": [q_high],
        }
    )


def summarize(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Correlations: low EOD vs calibration dispersion
    corr = df[["EOD", "ECE_overall", "ECE_range", "ECE_std"]].corr(method="spearman")

    # Linear fit to show directionality
    slope_range, intercept_range = np.polyfit(df["EOD"], df["ECE_range"], 1)
    slope_overall, intercept_overall = np.polyfit(df["EOD"], df["ECE_overall"], 1)

    # EO-enforced vs others
    eo_summary = (
        df.groupby("EO_enforced")[["EOD", "ECE_overall", "ECE_range", "ECE_std"]]
        .mean()
        .reset_index()
    )

    # Impossibility gap quantification
    gap_df = compute_impossibility_gap(df)

    # Persist tables
    df.to_csv(out_dir / "impossibility_long.csv", index=False)
    corr.to_csv(out_dir / "impossibility_corr.csv")
    eo_summary.to_csv(out_dir / "impossibility_eo_vs_other.csv", index=False)
    gap_df.to_csv(out_dir / "impossibility_gap.csv", index=False)

    # Markdown narrative
    lines = []
    lines.append("# Fairness Impossibility Analysis\n")
    lines.append("## Key correlations (Spearman)\n")
    lines.append(corr.to_markdown())
    lines.append("\n## EO-enforced vs others (means)\n")
    lines.append(eo_summary.to_markdown(index=False))
    lines.append("\n## Impossibility gap (calibration dispersion when pushing EOD low)\n")
    lines.append(gap_df.to_markdown(index=False))
    lines.append(
        "\nNotes:\n"
        f"- Slope(EOD -> ECE_range): {slope_range:.4f} (negative means tighter EO increases calibration spread if positive).\n"
        f"- Slope(EOD -> ECE_overall): {slope_overall:.4f}.\n"
        "- Use the gap to flag whether driving EOD to lower quartile systematically raises calibration dispersion.\n"
    )
    (out_dir / "impossibility_report.md").write_text("\n".join(lines))


def main() -> None:
    out_dir = PARENT / "fairness_tradeoffs" / "impossibility"
    df = load_rows()
    if df.empty:
        raise SystemExit("No data available for impossibility analysis.")
    summarize(df, out_dir)
    print(f"Wrote impossibility analysis to {out_dir}")


if __name__ == "__main__":
    main()
