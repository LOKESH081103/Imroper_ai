"""
dashboard_report.py - build a single, self-contained, presentation-ready HTML
dashboard (proper/improper breakdown, issue categories, model comparison)
from an address file, using the SAME rule logic as app.py (rule_engine.py -
no duplicated logic, no drift between the two).

Usage
-----
    # Rule-based findings only (no ground-truth labels available):
    python dashboard_report.py agreements.xlsx --address-col Address

    # With ground truth, for real accuracy/model-comparison numbers:
    python dashboard_report.py agreements.xlsx --address-col Address \
        --label-col type --improper-values improper,issue

Output: dashboard.html (or --out path) next to this script. Open it in any
browser - it's fully self-contained except for the Chart.js CDN script tag,
so it needs internet only to load that one library, nothing else.

Notes
-----
- Layer 2 (pincode master-data check) needs a live network call per unique
  pincode and is intentionally NOT run here - use the Streamlit app
  (`streamlit run app.py`) for the full 4-layer numbers if you need Layer 2
  included in the dashboard. This script covers Layers 1, 3, and (if you
  supply ground truth) 4.
- If you pass --label-col, the model-comparison section trains and
  cross-validates for real on your data (same code path as the app's Layer
  4) - these are not simulated numbers.
"""

import argparse
import sys
from collections import Counter
from datetime import datetime
from typing import List, Optional

import pandas as pd

from rule_engine import analyze_address_local, severity_for
import ml_classifier as mc
from dashboard_builder import render_html


def read_any(path: str) -> pd.DataFrame:
    if path.lower().endswith(".csv"):
        return pd.read_csv(path, keep_default_na=False, na_values=[])
    return pd.read_excel(path, keep_default_na=False, na_values=[])


def run_rules(df: pd.DataFrame, address_col: str, min_words: int = 5, merge_len_threshold: int = 15):
    severities, categories_per_row = [], []
    for addr in df[address_col]:
        issues, _, _ = analyze_address_local(addr, min_words, merge_len_threshold)
        base_issues = [i.split("(")[0] for i in issues]
        severities.append(severity_for(issues))
        categories_per_row.append(base_issues)
    return severities, categories_per_row


def build_report_data(df: pd.DataFrame, address_col: str, label_col: Optional[str],
                       improper_values: Optional[List[str]], min_words: int, merge_len_threshold: int) -> dict:
    total = len(df)
    severities, categories_per_row = run_rules(df, address_col, min_words, merge_len_threshold)
    rule_pred = ["improper" if s in ("Critical", "Warning") else "proper" for s in severities]

    sev_counts = Counter(severities)
    issue_counts = Counter(c for row in categories_per_row for c in row)
    top_issues = issue_counts.most_common(10)

    data = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "total": total,
        "layer2_included": False,
        "has_ground_truth": False,
        "severity": {
            "critical": sev_counts.get("Critical", 0),
            "warning": sev_counts.get("Warning", 0),
            "clean": sev_counts.get("Clean", 0),
        },
        "rule_flagged": sum(1 for p in rule_pred if p == "improper"),
        "top_issues": top_issues,
        "model_comparison": None,
        "confusion": None,
        "ground_truth": None,
    }

    if label_col:
        improper_set = {str(v).strip() for v in (improper_values or [])}
        truth = df[label_col].astype(str).str.strip().map(
            lambda v: "improper" if v in improper_set else "proper"
        )
        data["has_ground_truth"] = True
        data["ground_truth"] = {
            "improper": int((truth == "improper").sum()),
            "proper": int((truth == "proper").sum()),
        }
        tp = int(((truth == "improper") & (pd.Series(rule_pred) == "improper")).sum())
        fn = int(((truth == "improper") & (pd.Series(rule_pred) == "proper")).sum())
        fp = int(((truth == "proper") & (pd.Series(rule_pred) == "improper")).sum())
        tn = int(((truth == "proper") & (pd.Series(rule_pred) == "proper")).sum())
        data["confusion"] = {"tp": tp, "fn": fn, "fp": fp, "tn": tn,
                              "accuracy": round(100 * (tp + tn) / max(total, 1), 1)}

        training_df = pd.DataFrame({
            "Address": df[address_col],
            "Label": (truth == "improper").astype(int),
        })
        training_df = training_df[training_df["Address"].astype(str).str.strip() != ""]
        ok, reason = mc.can_train(training_df)
        if ok:
            comparison = mc.compare_algorithms(training_df)
            data["model_comparison"] = {
                alg: {k: round(v * 100, 2) for k, v in r.items() if k in ("accuracy", "precision", "recall", "f1")}
                for alg, r in comparison.items()
            }
        else:
            data["model_comparison"] = {"skipped_reason": reason}

    return data


def main():
    parser = argparse.ArgumentParser(description="Build a presentation-ready address-quality dashboard.")
    parser.add_argument("file", help="Path to your .xlsx, .xls, or .csv address file")
    parser.add_argument("--address-col", required=True, help="Column name holding the address text")
    parser.add_argument("--label-col", default=None, help="Optional: column holding ground-truth labels")
    parser.add_argument("--improper-values", default=None,
                         help="Comma-separated values in --label-col that mean 'improper' (e.g. improper,issue)")
    parser.add_argument("--min-words", type=int, default=5)
    parser.add_argument("--merge-len-threshold", type=int, default=15)
    parser.add_argument("--title", default="Address quality report")
    parser.add_argument("--out", default="dashboard.html")
    args = parser.parse_args()

    df = read_any(args.file)
    if args.address_col not in df.columns:
        sys.exit(f"Column '{args.address_col}' not found. Available columns: {list(df.columns)}")

    improper_values = None
    if args.label_col:
        if args.label_col not in df.columns:
            sys.exit(f"Column '{args.label_col}' not found. Available columns: {list(df.columns)}")
        improper_values = [v.strip() for v in (args.improper_values or "improper").split(",")]

    data = build_report_data(df, args.address_col, args.label_col, improper_values,
                              args.min_words, args.merge_len_threshold)
    html = render_html(data, source_name=args.file, title=args.title)

    with open(args.out, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Wrote {args.out} ({data['total']} rows analyzed)")


if __name__ == "__main__":
    main()
