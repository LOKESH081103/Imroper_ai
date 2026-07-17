"""
Layer 4 - ML pattern classifier (TF-IDF + engineered features -> XGBoost /
Logistic Regression / Naive Bayes), self-trained from your own pipeline's
history. This is now the ONLY intelligence layer in the pipeline - there is
no external AI/LLM step. Everything here is offline, free, and scales to
hundreds of thousands of rows in seconds.

Why this exists on top of Layers 1-3
--------------------------------------
The rules only catch what someone already thought to write a regex or
dictionary entry for. This layer instead LEARNS what "clean" vs "problem"
looks like from data your own runs have produced, so it generalizes to junk
patterns nobody encoded a rule for yet - and it tells you, honestly, how
well it's actually doing, instead of just handing back a probability you
have to trust blindly.

Where training labels come from (three tiers, by priority)
-------------------------------------------------------------
1. Bootstrap labels (lowest priority, "rule"): every run, each address's
   Layers 1-3 Severity (Clean -> 0, Critical/Warning -> 1) is stored as a
   weak label.
2. Imported ground truth ("imported"): a bulk file you already have with a
   known-correct label per address (e.g. a "type" column of
   proper/improper) via `import_labeled_dataset()`. Treated as authoritative
   - never overwritten by a later bootstrap pass, and never trimmed by the
   training-set size cap the way bootstrap rows are.
3. Human labels (highest priority, "human"): a reviewer marking "Confirmed
   Issue" / "False Positive" in the review queue for one specific address.
   Wins over both of the above for that exact address, since a direct human
   judgment on this specific instance outranks a bulk import or a heuristic.

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

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except ImportError:  # pragma: no cover - degrade gracefully if xgboost isn't installed yet
    _HAS_XGB = False

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

TRAINING_COLUMNS = ["Address", "Label", "Source"]  # Label: 1=issue, 0=clean. Source: "rule" < "imported" < "human"
SOURCE_PRIORITY = {"rule": 0, "imported": 1, "human": 2}
PROTECTED_SOURCES = ("imported", "human")  # never trimmed by the row cap, unlike "rule"


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


def _dedupe_by_priority(df: pd.DataFrame) -> pd.DataFrame:
    """
    One label per unique (normalized) address. When the same address shows
    up with multiple Source tiers (e.g. an old bootstrap guess and a newly
    imported ground-truth label), the higher-priority tier wins - "human" >
    "imported" > "rule" - regardless of which one happened to be appended
    more recently. Within the same tier, the most recently appended row wins
    (e.g. re-importing an updated file overrides your previous import for
    any address it also covers).
    """
    if df.empty:
        return df
    work = df.copy()
    work["_norm"] = work["Address"].map(_normalize)
    work["_prio"] = work["Source"].map(SOURCE_PRIORITY).fillna(0).astype(int)
    # Stable sort by priority (ascending) preserves original relative order
    # within each priority tier, so `keep="last"` within a tie still keeps
    # the most-recently-appended row for that tier.
    work = work.sort_values("_prio", kind="stable")
    work = work.drop_duplicates(subset="_norm", keep="last")
    return work.drop(columns=["_norm", "_prio"]).reset_index(drop=True)


def _cap_training_data(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) <= MAX_TRAINING_ROWS:
        return df
    protected = df[df["Source"].isin(PROTECTED_SOURCES)]
    rule = df[df["Source"] == "rule"]

    if len(protected) >= MAX_TRAINING_ROWS:
        # Human decisions and imported ground truth are never guessed - keep
        # every single one even past the cap. The cap exists to stop
        # unbounded, low-value bootstrap growth, not to throw away real
        # labels. (Training/CV will simply take a bit longer than usual.)
        return protected.reset_index(drop=True)

    budget = MAX_TRAINING_ROWS - len(protected)
    rule_0 = rule[rule["Label"] == 0].tail(budget // 2 + budget % 2)
    rule_1 = rule[rule["Label"] == 1].tail(budget // 2)
    capped = pd.concat([protected, rule_0, rule_1], ignore_index=True)
    return capped.reset_index(drop=True)


def update_training_data(result_df: pd.DataFrame, feedback_df: pd.DataFrame,
                          path: str = TRAINING_DATA_FILE) -> pd.DataFrame:
    """
    Fold this run's rule-based severities into the cumulative training set as
    bootstrap labels, then re-apply human reviewer decisions on top so they
    always win (and always outrank any imported ground truth too - see
    _dedupe_by_priority). Returns the updated cumulative training DataFrame.
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

    human_rows = []
    if feedback_df is not None and not feedback_df.empty:
        for _, row in feedback_df.iterrows():
            addr = row.get("Address")
            decision = row.get("Decision")
            if not isinstance(addr, str) or not addr.strip():
                continue
            if decision not in ("Confirmed Issue", "False Positive"):
                continue
            label = 1 if decision == "Confirmed Issue" else 0
            human_rows.append({"Address": addr, "Label": label, "Source": "human"})

    combined = pd.concat(
        [existing, pd.DataFrame(bootstrap_rows, columns=TRAINING_COLUMNS),
         pd.DataFrame(human_rows, columns=TRAINING_COLUMNS)],
        ignore_index=True,
    )
    combined = _dedupe_by_priority(combined)
    combined = _cap_training_data(combined)
    save_training_data(combined, path)
    return combined


def import_labeled_dataset(label_df: pd.DataFrame, address_col: str, label_col: str,
                            positive_values, path: str = TRAINING_DATA_FILE) -> pd.DataFrame:
    """
    Bulk-import a dataset you already have ground-truth labels for - e.g. a
    'type' column with values like 'proper'/'improper' - straight into
    Layer 4's training data.

    address_col / label_col: column names in `label_df` holding the address
    text and the label, respectively.
    positive_values: the set of raw values in `label_col` that mean
    "improper / issue" (e.g. {"improper"} or {"issue", "invalid"}).
    Matching is exact-string after stripping whitespace (case-sensitive by
    design, since the caller picks these values directly from the UI's
    detected distinct values - no silent case-folding surprises).
    Everything else in that column is treated as "proper / clean".

    These rows get Source="imported": authoritative like a human review
    decision (never overwritten by a later bootstrap pass, never trimmed by
    the training-set size cap), but a specific reviewer decision on an
    individual address still outranks it if the two ever disagree - see
    _dedupe_by_priority.
    """
    positive_values = {str(v).strip() for v in positive_values}

    rows = []
    for _, row in label_df.iterrows():
        addr = row.get(address_col)
        if not isinstance(addr, str) or not addr.strip():
            continue
        raw_label = str(row.get(label_col, "")).strip()
        label = 1 if raw_label in positive_values else 0
        rows.append({"Address": addr, "Label": label, "Source": "imported"})

    existing = load_training_data(path)
    imported_df = pd.DataFrame(rows, columns=TRAINING_COLUMNS)
    combined = pd.concat([existing, imported_df], ignore_index=True)
    combined = _dedupe_by_priority(combined)
    combined = _cap_training_data(combined)
    save_training_data(combined, path)
    return combined


def training_data_summary(training_df: pd.DataFrame) -> dict:
    """Small breakdown for the UI: counts by source tier and by label."""
    if training_df.empty:
        return {"total": 0, "rule": 0, "imported": 0, "human": 0, "clean": 0, "issue": 0}
    src_counts = training_df["Source"].value_counts()
    label_counts = training_df["Label"].astype(int).value_counts()
    return {
        "total": len(training_df),
        "rule": int(src_counts.get("rule", 0)),
        "imported": int(src_counts.get("imported", 0)),
        "human": int(src_counts.get("human", 0)),
        "clean": int(label_counts.get(0, 0)),
        "issue": int(label_counts.get(1, 0)),
    }


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
ALGORITHMS = ["xgb", "logreg", "nb"] if _HAS_XGB else ["logreg", "nb"]
DEFAULT_ALGORITHM = "xgb" if _HAS_XGB else "logreg"


def _build_pipeline(algorithm: str) -> Pipeline:
    word_vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1, max_features=3000, lowercase=True)
    char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, max_features=2000, lowercase=True)
    stats = AddressStatsExtractor()
    features = FeatureUnion([("word", word_vec), ("char", char_vec), ("stats", stats)])

    if algorithm == "nb":
        clf = MultinomialNB()
    elif algorithm == "xgb" and _HAS_XGB:
        # Gradient-boosted trees on the same TF-IDF + hand-engineered feature
        # set. Handles nonlinear feature interactions (e.g. "short address
        # AND foreign word AND low digit ratio") that a linear model can't
        # express directly, usually giving better F1 than Logistic
        # Regression / Naive Bayes on this kind of data - while still
        # predicting hundreds of thousands of rows in a couple of seconds.
        clf = XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.8,
            eval_metric="logloss",
            tree_method="hist",   # fast histogram-based training, scales well
            n_jobs=-1,
            random_state=42,
        )
    else:
        algorithm = "logreg"
        clf = LogisticRegression(max_iter=1000, class_weight="balanced")

    return Pipeline([("features", features), ("clf", clf)])


def _apply_class_balance(pipe: Pipeline, labels) -> None:
    """
    XGBoost has no built-in class_weight='balanced' like LogisticRegression -
    it uses scale_pos_weight instead (ratio of negative to positive class
    count). Set it here from the actual label distribution so a dataset with
    far more "clean" than "issue" rows (or vice versa) doesn't bias the
    model toward the majority class.
    """
    clf = pipe.named_steps["clf"]
    if _HAS_XGB and isinstance(clf, XGBClassifier):
        counts = pd.Series(labels).value_counts()
        n_neg, n_pos = counts.get(0, 1), counts.get(1, 1)
        clf.set_params(scale_pos_weight=n_neg / max(n_pos, 1))


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
    _apply_class_balance(pipe, labels)

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
    """Cross-validate every available algorithm and return {'xgb': {...}, 'logreg': {...}, 'nb': {...}}."""
    return {alg: evaluate_model(training_df, alg) for alg in ALGORITHMS}


def train_model(training_df: pd.DataFrame, algorithm: str = DEFAULT_ALGORITHM):
    """Fits the deployed model on ALL available training data (post-evaluation)."""
    texts = training_df["Address"].fillna("").astype(str).tolist()
    labels = training_df["Label"].astype(int).tolist()

    pipe = _build_pipeline(algorithm)
    _apply_class_balance(pipe, labels)
    pipe.fit(texts, labels)

    if _HAS_XGB and isinstance(pipe.named_steps["clf"], XGBClassifier):
        _attach_direction(pipe, texts, labels)

    with open(MODEL_FILE, "wb") as f:
        pickle.dump({"pipeline": pipe, "algorithm": algorithm, "n_samples": len(texts)}, f)

    return pipe


def _attach_direction(pipe: Pipeline, texts, labels) -> None:
    """
    XGBoost's feature_importances_ is an unsigned magnitude (how much a
    feature reduced loss across all trees), unlike a Logistic Regression
    coefficient, which is signed (pushes toward "issue" or "clean"). To
    still show a meaningful "pushes toward Issue / pushes toward Clean"
    breakdown in the UI, approximate direction by comparing each feature's
    average value across the two classes in the actual training data, then
    combine that sign with the tree-based importance magnitude. This is an
    approximation (trees can pick up non-monotonic effects a single mean
    difference won't fully capture) but is accurate enough for a sanity-check
    feature panel.
    """
    labels_arr = np.array(labels)
    feat_matrix = pipe.named_steps["features"].transform(texts)
    if hasattr(feat_matrix, "toarray"):
        mean_pos = np.asarray(feat_matrix[labels_arr == 1].mean(axis=0)).ravel()
        mean_neg = np.asarray(feat_matrix[labels_arr == 0].mean(axis=0)).ravel()
    else:
        mean_pos = feat_matrix[labels_arr == 1].mean(axis=0)
        mean_neg = feat_matrix[labels_arr == 0].mean(axis=0)
    direction = np.sign(mean_pos - mean_neg)
    direction[direction == 0] = 1.0
    pipe.named_steps["clf"]._direction = direction


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
    elif _HAS_XGB and isinstance(clf, XGBClassifier):
        importances = clf.feature_importances_
        direction = getattr(clf, "_direction", np.ones_like(importances))
        weights = importances * direction
    else:
        # classes_ is sorted ascending -> index 0 = clean, index 1 = issue
        weights = clf.feature_log_prob_[1] - clf.feature_log_prob_[0]

    order = np.argsort(weights)
    top_issue = [(names[i], float(weights[i])) for i in order[::-1][:n]]
    top_clean = [(names[i], float(weights[i])) for i in order[:n]]
    return top_issue, top_clean