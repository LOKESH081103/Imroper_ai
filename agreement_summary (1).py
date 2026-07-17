"""
agreement_summary.py - roll address-level results up to the agreement level.

Business rule (from the manager, this is the whole point of this module):
one Agreement No can appear multiple times with a DIFFERENT address on each
row (e.g. AG001 x5, five different addresses on file for it). The agreement
is only as good as its worst address - if even ONE of an agreement's
addresses is Critical, the entire agreement gets flagged Critical for
follow-up, even if the other 4 are perfectly clean. Same idea one level
down for Warning.

Severity rollup precedence (worst wins): Critical > Warning > Clean.
"Clean (reviewer-cleared)" counts as Clean here - a human already reviewed
and cleared that specific address.
"""

import pandas as pd

_RANK = {"Critical": 2, "Warning": 1, "Clean": 0, "Clean (reviewer-cleared)": 0}
_RANK_TO_SEVERITY = {2: "Critical", 1: "Warning", 0: "Clean"}

ROLLUP_COLUMNS = [
    "Agreement No", "Agreement Severity", "Address Count",
    "Critical Addresses", "Warning Addresses", "Clean Addresses",
]


def _norm(sev: str) -> str:
    return "Clean" if sev == "Clean (reviewer-cleared)" else sev


def agreement_rollup(result_df: pd.DataFrame, agreement_col: str = "Agreement No") -> pd.DataFrame:
    """
    One row per agreement: its worst-address severity, how many addresses
    it has on file, and the Critical/Warning/Clean split among them.
    """
    if result_df is None or result_df.empty or agreement_col not in result_df.columns:
        return pd.DataFrame(columns=ROLLUP_COLUMNS)

    df = result_df[[agreement_col, "Severity"]].copy()
    df["_norm_sev"] = df["Severity"].map(_norm)
    df["_rank"] = df["_norm_sev"].map(_RANK).fillna(0).astype(int)

    grouped = df.groupby(agreement_col, sort=False, dropna=False)
    out = grouped.agg(
        **{
            "Address Count": ("_norm_sev", "size"),
            "_worst_rank": ("_rank", "max"),
            "Critical Addresses": ("_norm_sev", lambda s: int((s == "Critical").sum())),
            "Warning Addresses": ("_norm_sev", lambda s: int((s == "Warning").sum())),
            "Clean Addresses": ("_norm_sev", lambda s: int((s == "Clean").sum())),
        }
    ).reset_index()

    out["Agreement Severity"] = out["_worst_rank"].map(_RANK_TO_SEVERITY)
    out = out.drop(columns=["_worst_rank"])
    out = out.rename(columns={agreement_col: "Agreement No"})
    return out[ROLLUP_COLUMNS]


def summarize_agreements(result_df: pd.DataFrame, agreement_col: str = "Agreement No") -> dict:
    """
    The headline counts for the summary cards:
      - total agreements / total addresses
      - clean/warning/critical agreement counts, and how many addresses
        each of those buckets contains in total.
    Also returns the full per-agreement rollup table under "rollup" so the
    UI can show/download the detail.
    """
    rollup = agreement_rollup(result_df, agreement_col)
    total_agreements = len(rollup)
    total_addresses = int(rollup["Address Count"].sum()) if total_agreements else 0

    summary = {
        "total_agreements": total_agreements,
        "total_addresses": total_addresses,
        "rollup": rollup,
    }
    for sev in ("Clean", "Warning", "Critical"):
        subset = rollup[rollup["Agreement Severity"] == sev]
        summary[f"{sev.lower()}_agreements"] = len(subset)
        summary[f"{sev.lower()}_addresses"] = int(subset["Address Count"].sum()) if len(subset) else 0
    return summary
