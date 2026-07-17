"""
Persists reviewer decisions (Confirmed Issue / False Positive) to a local
CSV so the app "remembers" corrections across sessions and can auto-clear
previously-approved addresses on future runs.
"""

import os

import pandas as pd

FEEDBACK_FILE = "reviewer_feedback.csv"
COLUMNS = ["Agreement No", "Address", "Decision", "Notes"]


def load_feedback(path: str = FEEDBACK_FILE) -> pd.DataFrame:
    if os.path.exists(path):
        try:
            # keep_default_na=False: an Address of literally "NA" is a common
            # real value in this dataset (it's one of the placeholder junk
            # values Layer 3 detects) and must not be read back as a null.
            df = pd.read_csv(path, keep_default_na=False, na_values=[])
            for c in COLUMNS:
                if c not in df.columns:
                    df[c] = ""
            return df[COLUMNS]
        except Exception:
            return pd.DataFrame(columns=COLUMNS)
    return pd.DataFrame(columns=COLUMNS)


def save_feedback(df: pd.DataFrame, path: str = FEEDBACK_FILE) -> None:
    df.to_csv(path, index=False)


def normalize(addr: str) -> str:
    return " ".join(str(addr).strip().upper().split())


def previously_cleared_addresses(feedback_df: pd.DataFrame) -> set:
    """Addresses a human has already marked as false positives."""
    if feedback_df.empty:
        return set()
    cleared = feedback_df[feedback_df["Decision"] == "False Positive"]
    return {normalize(a) for a in cleared["Address"].tolist()}