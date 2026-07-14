"""
Layer 4 - ML pattern classifier (TF-IDF + engineered features -> Logistic
Regression / Naive Bayes), self-trained from your own pipeline's history.

Why this exists on top of Layers 1-3
--------------------------------------
The rules only catch what someone already thought to write a regex or
dictionary entry for. This layer instead LEARNS what "clean" vs "problem"
looks like from data your own runs have produced, so it generalizes to junk
patterns nobody encoded a rule for yet - and it tells you, honestly, how
well it's actually doing, instead of just handing back a probability you
have to trust blindly.

Where training labels come from (self-training, two tiers)
-------------------------------------------------------------
1. Bootstrap labels: every run, each address's Layers 1-3 Severity
   (Clean -> 0, Critical/Warning -> 1) is stored as a weak label.
2. Human labels: a reviewer marking "Confirmed Issue" / "False Positive" in
   the review queue REPLACES the bootstrap label for that address. Human
   labels are never overwritten by a later bootstrap pass.

Features (why it's more than a bag-of-words toy)
-----------------------------------------------------
Three feature groups are fused together:
  - Word-level TF-IDF (uni+bigrams)   - placeholder words, foreign city names
  - Char-level TF-IDF (3-5 grams)     - glued words, typos, gibberish
  - Hand-engineered numeric stats     - length, digit ratio, repeated-char
                                         runs, glued digit/letter boundaries,
                                         vowel ratio, etc. These mirror the
                                         signals Layers 1 and 3 already look
                                         for, given to the model as explicit
                                         numeric features rather than making
                                         it re-discover them purely from text.

Evaluation (why the numbers you see are honest)
---------------------------------------------------
With small, growing datasets, a single train/test split is noisy - whichever
few rows land in the test fold can swing "accuracy" wildly. Instead this
module uses stratified K-fold cross-validation and evaluates every row on a
fold where it was held out during training (cross_val_predict), then reports
accuracy / precision / recall / F1 and a confusion matrix from those
out-of-fold predictions. The deployed model is then refit on 100% of the
data for actual inference - evaluation and deployment never share a
contaminated fit.

100% offline: no network calls, no API key, no per-row cost. Requires
scikit-learn and numpy (see requirements.txt).
"""

import os
import re
import pickle

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

TRAINING_DATA_FILE = "ml_training_data.csv"
MODEL_FILE = "ml_address_model.pkl"

MIN_SAMPLES = 20
MIN_PER_CLASS = 5
MAX_CV_SPLITS = 5

# Large files (1 lakh+ rows) would otherwise make ml_training_data.csv grow
# without bound - every run adds a bootstrap row per address - which makes
# retraining progressively slower over time. Cap it: always keep every
# human-reviewed row (limited and valuable), then fill the remaining budget
# with the most recent rule-labeled rows, balanced across classes. This
# keeps training/CV at a consistent, fast wall-clock cost regardless of how
# many times huge files get processed.
MAX_TRAINING_ROWS = 20000

TRAINING_COLUMNS = ["Address", "Label", "Source"]  # Label: 1=issue, 0=clean. Source: "rule" or "human"


def _normalize(addr) -> str:
    return " ".join(str(addr).strip().upper().split())


# ----------------------------------------------------------------------
# Persistent training set
# ----------------------------------------------------------------------
def load_training_data(path: str = TRAINING_DATA_FILE) -> pd.DataFrame:
    if os.path.exists(path):
        try:
            # keep_default_na=False is essential: pandas' default NA sentinel
            # list includes the literal string "NA", which is one of the most
            # common junk addresses this tool exists to catch. Without this,
            # every "NA" address silently becomes a real NaN on reload.
            df = pd.read_csv(path, keep_default_na=False, na_values=[])
            for c in TRAINING_COLUMNS:
                if c not in df.columns:
                    df[c] = ""
            return df[TRAINING_COLUMNS]
        except Exception:
            return pd.DataFrame(columns=TRAINING_COLUMNS)
    return pd.DataFrame(columns=TRAINING_COLUMNS)


def save_training_data(df: pd.DataFrame, path: str = TRAINING_DATA_FILE) -> None:
    df.to_csv(path, index=False)


def _cap_training_data(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) <= MAX_TRAINING_ROWS:
        return df
    human = df[df["Source"] == "human"]
    rule = df[df["Source"] == "rule"]

    if len(human) >= MAX_TRAINING_ROWS:
        # Even human-only rows exceed the cap (very unlikely) - keep the
        # most recent ones rather than dropping human feedback entirely.
        return human.tail(MAX_TRAINING_ROWS).reset_index(drop=True)

    budget = MAX_TRAINING_ROWS - len(human)
    rule_0 = rule[rule["Label"] == 0].tail(budget // 2 + budget % 2)
    rule_1 = rule[rule["Label"] == 1].tail(budget // 2)
    capped = pd.concat([human, rule_0, rule_1], ignore_index=True)
    return capped.reset_index(drop=True)


def update_training_data(result_df: pd.DataFrame, feedback_df: pd.DataFrame,
                          path: str = TRAINING_DATA_FILE) -> pd.DataFrame:
    """
    Fold this run's rule-based severities into the cumulative training set as
    bootstrap labels, then re-apply human reviewer decisions on top so they
    always win. Returns the updated cumulative training DataFrame.
    """
    existing = load_training_data(path)

    bootstrap_rows = []
    for _, row in result_df.iterrows():
        addr = row.get("Address")
        if not isinstance(addr, str) or not addr.strip():
            continue
        severity = row.get("Severity", "")
        label = 0 if severity in ("Clean", "Clean (reviewer-cleared)") else 1
        bootstrap_rows.append({"Address": addr, "Label": label, "Source": "rule"})

    combined = pd.concat(
        [existing, pd.DataFrame(bootstrap_rows, columns=TRAINING_COLUMNS)],
        ignore_index=True,
    )

    if feedback_df is not None and not feedback_df.empty:
        human_rows = []
        judged = set()
        for _, row in feedback_df.iterrows():
            addr = row.get("Address")
            decision = row.get("Decision")
            if not isinstance(addr, str) or not addr.strip():
                continue
            if decision not in ("Confirmed Issue", "False Positive"):
                continue
            label = 1 if decision == "Confirmed Issue" else 0
            human_rows.append({"Address": addr, "Label": label, "Source": "human"})
            judged.add(_normalize(addr))

        if judged:
            combined = combined[~combined["Address"].apply(lambda a: _normalize(a) in judged)]
        combined = pd.concat(
            [combined, pd.DataFrame(human_rows, columns=TRAINING_COLUMNS)],
            ignore_index=True,
        )

    # One label per unique address text; human rows are appended last, so
    # they naturally survive the "keep last" de-dup over older rule rows.
    combined = combined.drop_duplicates(subset=["Address"], keep="last").reset_index(drop=True)
    combined = _cap_training_data(combined)
    save_training_data(combined, path)
    return combined


def can_train(training_df: pd.DataFrame):
    """Returns (ok: bool, reason: str)."""
    if len(training_df) < MIN_SAMPLES:
        return False, f"Need at least {MIN_SAMPLES} labeled addresses to train (have {len(training_df)} so far)."
    counts = training_df["Label"].astype(int).value_counts()
    n_clean, n_issue = counts.get(0, 0), counts.get(1, 0)
    if n_clean < MIN_PER_CLASS or n_issue < MIN_PER_CLASS:
        return False, (f"Need at least {MIN_PER_CLASS} examples of both clean and flagged addresses "
                        f"(have {n_clean} clean, {n_issue} flagged).")
    return True, ""


# ----------------------------------------------------------------------
# Hand-engineered numeric features
# ----------------------------------------------------------------------
_VOWELS = set("AEIOU")
_PINCODE_RUN_RE = re.compile(r"\d{6}")
_GLUED_RE = re.compile(r"[A-Za-z]\d|\d[A-Za-z]")
_REPEAT_RUN_RE = re.compile(r"(.)\1{3,}")


def _address_stats(addr: str) -> dict:
    a = "" if not isinstance(addr, str) else addr.strip()
    au = a.upper()
    length = len(a)
    words = a.split()
    word_count = len(words)
    digits = sum(ch.isdigit() for ch in a)
    alpha = [ch for ch in au if ch.isalpha()]
    vowels = sum(ch in _VOWELS for ch in alpha)

    longest_repeat = 0
    m = _REPEAT_RUN_RE.search(au)
    if m:
        longest_repeat = len(m.group(0))

    return {
        "length": length,
        "word_count": word_count,
        "avg_word_len": (length / word_count) if word_count else 0.0,
        "digit_ratio": (digits / length) if length else 0.0,
        "special_char_ratio": (sum(not ch.isalnum() and not ch.isspace() for ch in a) / length) if length else 0.0,
        "vowel_ratio": (vowels / len(alpha)) if alpha else 0.0,
        "has_pincode_run": 1.0 if _PINCODE_RUN_RE.search(a) else 0.0,
        "has_glued_alnum": 1.0 if _GLUED_RE.search(a) else 0.0,
        "longest_repeat_run": float(longest_repeat),
        "comma_count": float(a.count(",")),
    }


_STAT_NAMES = list(_address_stats("").keys())


class AddressStatsExtractor(BaseEstimator, TransformerMixin):
    """
    Turns raw address strings into the numeric feature table above, then
    min-max scales each column to [0, 1] using ranges learned at fit time
    (and clips out-of-range values at inference time). Scaling is done
    manually here, rather than via a separate sklearn scaler, so this stays
    a single drop-in transformer inside the FeatureUnion. All outputs are
    non-negative by construction, which keeps MultinomialNB usable.
    """

    def fit(self, X, y=None):
        raw = np.array([[v for v in _address_stats(a).values()] for a in X], dtype=float)
        self._min = raw.min(axis=0)
        self._max = raw.max(axis=0)
        span = self._max - self._min
        span[span == 0] = 1.0  # avoid divide-by-zero for constant columns
        self._span = span
        return self

    def transform(self, X):
        raw = np.array([[v for v in _address_stats(a).values()] for a in X], dtype=float)
        scaled = (raw - self._min) / self._span
        return np.clip(scaled, 0.0, 1.0)

    def get_feature_names_out(self, input_features=None):
        return np.array([f"stat__{n}" for n in _STAT_NAMES])


# ----------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------
def _build_pipeline(algorithm: str) -> Pipeline:
    word_vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1, max_features=3000, lowercase=True)
    char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, max_features=2000, lowercase=True)
    stats = AddressStatsExtractor()
    features = FeatureUnion([("word", word_vec), ("char", char_vec), ("stats", stats)])

    if algorithm == "nb":
        clf = MultinomialNB()
    else:
        algorithm = "logreg"
        clf = LogisticRegression(max_iter=1000, class_weight="balanced")

    return Pipeline([("features", features), ("clf", clf)])


def _cv_splits(labels) -> int:
    counts = pd.Series(labels).value_counts()
    return max(2, min(MAX_CV_SPLITS, int(counts.min())))


def evaluate_model(training_df: pd.DataFrame, algorithm: str) -> dict:
    """
    Honest performance estimate via stratified K-fold cross-validation:
    every row is scored only by a fold that did NOT see it during training,
    so these numbers aren't inflated by the model grading its own homework.
    """
    texts = training_df["Address"].fillna("").astype(str).tolist()
    labels = training_df["Label"].astype(int).tolist()

    n_splits = _cv_splits(labels)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    pipe = _build_pipeline(algorithm)

    oof_preds = cross_val_predict(pipe, texts, labels, cv=skf, method="predict")
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, oof_preds, average="binary", pos_label=1, zero_division=0
    )
    return {
        "algorithm": algorithm,
        "n_splits": n_splits,
        "n_samples": len(labels),
        "accuracy": accuracy_score(labels, oof_preds),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": confusion_matrix(labels, oof_preds, labels=[0, 1]),
    }


def compare_algorithms(training_df: pd.DataFrame) -> dict:
    """Cross-validate both algorithms and return {'logreg': {...}, 'nb': {...}}."""
    return {alg: evaluate_model(training_df, alg) for alg in ("logreg", "nb")}


def train_model(training_df: pd.DataFrame, algorithm: str = "logreg"):
    """Fits the deployed model on ALL available training data (post-evaluation)."""
    texts = training_df["Address"].fillna("").astype(str).tolist()
    labels = training_df["Label"].astype(int).tolist()

    pipe = _build_pipeline(algorithm)
    pipe.fit(texts, labels)

    with open(MODEL_FILE, "wb") as f:
        pickle.dump({"pipeline": pipe, "algorithm": algorithm, "n_samples": len(texts)}, f)

    return pipe


def load_model():
    if os.path.exists(MODEL_FILE):
        try:
            with open(MODEL_FILE, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None
    return None


def predict(addresses, pipeline):
    """Returns a list of (predicted_label, probability_of_issue) tuples."""
    texts = [a if isinstance(a, str) and a.strip() else "" for a in addresses]
    probs = pipeline.predict_proba(texts)
    classes = list(pipeline.classes_)
    issue_idx = classes.index(1) if 1 in classes else None

    results = []
    for row in probs:
        p_issue = float(row[issue_idx]) if issue_idx is not None else 0.0
        results.append((1 if p_issue >= 0.5 else 0, p_issue))
    return results


# ----------------------------------------------------------------------
# Interpretability
# ----------------------------------------------------------------------
def top_features(pipeline, n: int = 15):
    """
    Returns (top_issue, top_clean): each a list of (feature_name, weight)
    tuples, ranked by how strongly that feature pushes the prediction
    toward "issue" or toward "clean". Works for both Logistic Regression
    (raw coefficients) and Naive Bayes (log-probability ratio between
    classes), so the model's decisions aren't a pure black box.
    """
    clf = pipeline.named_steps["clf"]
    feature_union = pipeline.named_steps["features"]

    names = []
    for _, transformer in feature_union.transformer_list:
        names.extend(transformer.get_feature_names_out())
    names = np.array(names)

    if isinstance(clf, LogisticRegression):
        weights = clf.coef_[0]
    else:
        # classes_ is sorted ascending -> index 0 = clean, index 1 = issue
        weights = clf.feature_log_prob_[1] - clf.feature_log_prob_[0]

    order = np.argsort(weights)
    top_issue = [(names[i], float(weights[i])) for i in order[::-1][:n]]
    top_clean = [(names[i], float(weights[i])) for i in order[:n]]
    return top_issue, top_clean
