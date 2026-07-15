# Agreement Address Quality Checker (4-layer pipeline)

Flags improper addresses in an Excel sheet of customer agreements using four
layers, from cheapest/fastest to smartest — **100% offline, no external AI/LLM
calls, no API keys, no rate limits, no per-row cost, anywhere in this pipeline**:

1. **Structural rules** — mechanical corruption (free, offline, instant)
2. **Pincode master-data validation** — geographic plausibility (free public API)
3. **Placeholder/gibberish dictionary** — junk/fake entries (free, offline)
4. **ML pattern classifier** — TF-IDF + XGBoost (default) / Logistic Regression / Naive Bayes, self-trained from your own data (free, offline, seconds even on huge files)

## 1. Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL Streamlit prints. Click **"Try demo data"** to see all
four layers work on real examples (including "Main Road 1, Dubai" and "NA"),
or upload your `.xlsx` and pick your Agreement No / Address columns.

## 2. What each layer catches

### Layer 1 — Structural rules
| Flag | Meaning |
|---|---|
| `EMPTY_ADDRESS` | Address field is blank |
| `DOUBLE_COMMA_EMPTY_FIELD` | Contains `,,` — empty field between commas |
| `PINCODE_DUPLICATED` | Pincode repeated back-to-back, e.g. `400054400054` |
| `PINCODE_GLUED_TO_TEXT` | Pincode stuck to a word, e.g. `MUMBAI400206` |
| `STATE_NOT_FOUND` | No recognizable Indian state name present |
| `REPEATED_PHRASE(...)` | A phrase repeats, e.g. "MAIN ROAD ... MAIN ROAD" |
| `ADDRESS_TOO_SHORT` | Fewer words than the minimum (adjustable, default 5) |
| `POSSIBLE_MERGED_WORDS(...)` | A long word may be two+ words glued together |
| `HOUSE_NO_ZERO_OR_PLACEHOLDER` | House number looks like a placeholder, e.g. `# 0` |

### Layer 2 — Pincode master-data validation (needs internet)
Every pincode is checked against the free, keyless **All India Pincode API**
(static JSON on GitHub Pages, sourced from the official Dept. of Posts /
data.gov.in dataset — no server, no rate limit, no signup):
`https://aniket-thapa.github.io/india-pincode-api`

| Flag | Meaning |
|---|---|
| `PINCODE_NOT_FOUND_IN_INDIA` | Pincode doesn't exist in the official directory — catches fake numbers and non-Indian addresses like "Main Road 1, Dubai" |
| `PINCODE_STATE_MISMATCH(...)` | The pincode belongs to a different state than what's written in the address |

**License note:** this API's data is CC BY-NC 4.0 (non-commercial use with
attribution). If this tool will be used for a commercial product rather than
internal data-quality review, either get written permission from the API
author or swap in the official data.gov.in "All India Pincode Directory"
download (link in the API repo) as a local file instead — the app's design
makes that a small change confined to `pincode_lookup.py`.

If there's no internet or the API is briefly down, this layer is skipped
gracefully — you'll see a warning banner, and the row still gets Layers 1
and 3.

### Layer 3 — Placeholder / gibberish dictionary (always on, free, offline)
| Flag | Meaning |
|---|---|
| `PLACEHOLDER_ADDRESS` | Entire field is a placeholder value: NA, TEST, XXX, TBD, "same as above", etc. |
| `PLACEHOLDER_WORD(...)` | Contains a junk word like TEST, DUMMY, ASDF |
| `FOREIGN_LOCATION_MENTIONED(...)` | Mentions a non-Indian city/country (Dubai, Singapore, London, etc.) |
| `REPEATED_CHARACTER_RUN` | Same character repeated 4+ times, e.g. `aaaa`, `9999` |
| `POSSIBLE_GIBBERISH_TEXT` | Long consonant run suggests random typing |

Extend `PLACEHOLDER_PHRASES`, `PLACEHOLDER_WORDS`, and
`FOREIGN_LOCATION_HINTS` at the top of `app.py` as you discover new junk
patterns in your real data — this list is meant to grow over time.

### Layer 4 — ML pattern classifier (free, offline, self-trained)
A **TF-IDF + XGBoost** (default) — or Logistic Regression / Naive Bayes —
model that learns what "clean" vs "problem" addresses look like from your
own data, instead of relying only on hand-written rules. Trains in seconds,
predicts a full file (even 8+ lakh rows) in a couple of seconds, no internet
and no API key required, ever.

| Flag | Meaning |
|---|---|
| `ML_FLAGGED_PATTERN(p=...)` | The classifier judged this address's patterns as issue-like, with `p` its estimated probability |

**Features it actually looks at** (not just a bag-of-words guess):
- Word-level TF-IDF (unigrams + bigrams) — placeholder words, foreign city names
- Character-level TF-IDF (3–5 grams) — glued-together words, typos, gibberish
- Hand-engineered numeric stats — length, word count, digit ratio, vowel
  ratio, longest repeated-character run, glued digit/letter boundaries,
  presence of a 6-digit pincode run, comma count. These give the model
  explicit access to the same kind of signal Layers 1 and 3 look for,
  instead of making it re-discover them purely from raw text.

**Why XGBoost by default:** gradient-boosted trees pick up nonlinear
*combinations* of the signals above (e.g. "short address AND foreign word
AND low digit ratio, together") that a linear model (Logistic Regression)
can't represent directly. On the kind of messy, rule-adjacent text this tool
sees, that usually means a meaningfully higher F1 — while still predicting
hundreds of thousands of rows in a couple of seconds using scikit-learn's
histogram-based training (`tree_method="hist"`). Class imbalance (e.g. far
more clean than flagged addresses) is corrected automatically via
`scale_pos_weight`, the XGBoost equivalent of `class_weight="balanced"`.

**How it trains itself — no manual labeling needed to get started:**
- Every run, each address's Layer 1–3 **Severity** (Clean → "clean" label,
  Critical/Warning → "issue" label) is stored as a *bootstrap* label in
  `ml_training_data.csv` (created automatically, lives next to the app).
- Any decision you save in the **review queue** ("Confirmed Issue" /
  "False Positive") **overrides** the bootstrap label for that exact
  address, since a human call is worth more than a heuristic. Human labels
  are never overwritten by a later bootstrap pass.
- The model retrains from this accumulated file on every run — the more
  you use the tool (and the more reviewer decisions you save), the sharper
  it gets.
- It needs at least 20 labeled addresses (5+ of each class) before it will
  train at all; below that it shows an info banner explaining what's
  missing instead of guessing.

**Honest accuracy, not a black box:** every time it trains, the app also
runs **stratified K-fold cross-validation** and reports Accuracy,
Precision, Recall, F1, and a confusion matrix computed only from
predictions made on data each fold never saw during its own training —
so the numbers you see aren't the model grading its own homework. With
"Auto-select best model" enabled (default on), it cross-validates *all
three* algorithms (XGBoost, Logistic Regression, Naive Bayes) and keeps
whichever scores higher F1, showing you a side-by-side comparison. An
expander after each run also shows the top word/character/stat features
pushing predictions toward "Issue" vs "Clean" (for XGBoost this direction
is an approximation — see the code comment in `ml_classifier.py` — since
tree ensembles don't have a single signed coefficient the way a linear
model does), so you can sanity-check what it actually learned instead of
trusting it blindly.

With small datasets these metrics are noisy — the app flags this
explicitly below 50 labeled samples — and they'll stabilize as more real
data (and reviewer decisions) accumulate.

Back up `ml_training_data.csv` along with `reviewer_feedback.csv` if you
redeploy the app somewhere new — losing it just means the classifier
starts cold again, not that anything breaks.

## 3. Severity levels
- **Critical** — undeliverable: empty, no pincode, pincode doesn't exist,
  placeholder value, or far too short
- **Warning** — needs a human look: structural glitches, mismatches,
  possible merged words, gibberish signals
- **Clean** — passed all active layers

## 4. Import a labeled dataset for Layer 4 (optional)
Already have a file where each address is marked, e.g., proper/improper (or
clean/issue, valid/invalid — any wording works)? You don't have to wait for
the bootstrap process to relearn that from scratch. Under **"📥 Import a
labeled dataset to train Layer 4"** (appears once you've loaded a file):

1. Upload that file (`.xlsx`, `.xls`, or `.csv`).
2. Pick which column holds the address text, and which column holds the
   label (e.g. your `type` column).
3. Select which value(s) in that column mean **improper/issue** — common
   ones like `improper`, `issue`, `invalid` are pre-selected automatically;
   everything else in that column is treated as **proper/clean**.
4. Click **"Import into Layer 4 training data"**.

These rows are written into `ml_training_data.csv` as ground truth:
- **Never overwritten by a later bootstrap pass** — Layers 1-3's automatic
  guesses can't quietly replace a label you already know is correct.
- **Never trimmed by the training-set size cap** — even if you import
  50,000+ pre-labeled addresses, every one of them is kept; only the
  automatic bootstrap rows get trimmed to make room.
- Still **outranked by the review queue** (Section 5) for any individual
  address — if you explicitly mark one specific address "Confirmed Issue"
  or "False Positive" later, that decision wins over the bulk import for
  that address, since it's the most direct signal you can give.

You can import multiple files over time (e.g. add a second labeled batch
later) — re-importing an address that's already there just updates its
label with whatever the newest import says.

## 5. Review queue (human-in-the-loop)
After running a check, flagged rows appear in an editable table. Mark each
as **Confirmed Issue** or **False Positive** and click **Save reviewer
decisions**. This is written to `reviewer_feedback.csv` next to the app and
is automatically re-applied on every future run — an address marked "False
Positive" once won't be flagged again, so the tool gets quieter over time as
you use it.

Back up `reviewer_feedback.csv` if you redeploy the app somewhere new.

## 6. Exporting results
Two download buttons after each run:
- **Flagged only** — for sending back to whoever owns the data
- **Full results** — every row with status, for audit/record-keeping

## 7. Large files (up to 8+ lakh rows)
Built to handle big batches without hanging:
- **Layer 2 is deduplicated and parallelized.** India only has ~19,000 real
  pincodes total, so however many rows you upload, that's the real ceiling
  on how many pincodes actually need a network call — everything else is a
  cache hit. Lookups run concurrently (default 40 at once, adjustable in
  the sidebar), with a live progress bar so you can see it's working. This
  does **not** get slower as row count grows past 100k — the ceiling is
  fixed at ~19,000 lookups either way.
- **Layers 1 & 3 are pure regex/string logic**, no network — roughly a
  second for 100,000 rows, still well under a minute at 8 lakh.
- **Layer 4 (ML/XGBoost) caps its training set** at 20,000 labeled addresses
  so retraining stays fast (a few seconds) no matter how many times huge
  files get reprocessed. All human reviewer feedback is always kept; only
  the oldest bootstrap labels get trimmed once the cap is hit. *Prediction*
  on the full uploaded file (all 8 lakh rows) is a single vectorized
  scikit-learn pass — a few seconds, not minutes, regardless of file size,
  since it's not calling any external API per row.
- **Uploads are read with a faster Excel engine** (`python-calamine`) when
  available, falling back automatically to `openpyxl` if not.

Rough expectation: an 8-lakh-row file with realistic (heavily-reused)
Indian pincodes should finish Layer 2 in a few minutes on a normal
connection, with Layers 1, 3, and 4 combined adding well under a minute on
top. If your file has an unusually high share of unique/fake pincodes,
Layer 2 will take longer — the progress bar shows real numbers either way,
and the "Reset & retry Layer 2" flow (see the network warning banner)
handles any rows that fail mid-batch.

**At this scale, also watch:**
- `st.dataframe` rendering all 8 lakh rows in the browser can feel sluggish
  — the two download buttons (flagged-only / full results) are the
  intended way to actually consume output at this size, not scrolling the
  on-screen table.
- Writing `.xlsx` for 8 lakh rows is noticeably slower than `.csv`; if you
  don't specifically need Excel formatting, exporting to CSV is faster for
  files this large.

## 8. Deploying for free (so your team can use it without you running it locally)
1. Push this folder to a GitHub repo.
2. Go to https://share.streamlit.io (Streamlit Community Cloud — free tier).
3. Sign in with GitHub, "New app", point it at your repo and `app.py`.
4. Deploy — you get a shareable `*.streamlit.app` link.

No API keys are needed anywhere in this app — nothing to configure per user.

## Files
- `app.py` — main Streamlit app, Layers 1 and 3, orchestration, UI
- `pincode_lookup.py` — Layer 2, pincode API client with caching
- `ml_classifier.py` — Layer 4, TF-IDF + XGBoost/Logistic Regression/Naive Bayes classifier
- `feedback_store.py` — review-queue persistence
- `reviewer_feedback.csv` — created automatically after your first saved review
- `ml_training_data.csv` — created automatically after your first run; cumulative labeled address set for Layer 4
- `ml_address_model.pkl` — created automatically once Layer 4 has enough data to train; the saved model
