"""
Agreement Address Quality Checker - 4-layer pipeline
-----------------------------------------------------
Layer 1: Structural/mechanical rules (free, offline, instant)
Layer 2: Pincode master-data validation (free public API, needs internet)
Layer 3: Placeholder / gibberish / foreign-location dictionary (free, offline)
Layer 4: ML pattern classifier - TF-IDF + Logistic Regression/Naive Bayes (free, offline)
Layer 5: Optional AI semantic judge - Gemini free tier (needs your API key)

Run:
    pip install -r requirements.txt
    streamlit run app.py
"""

import io
import re
import time

import pandas as pd
import streamlit as st

from pincode_lookup import (
    lookup_pincodes_bulk, check_connectivity, circuit_status, reset_circuit, set_ssl_verify,
    DEFAULT_BULK_WORKERS,
)
from ai_review import gemini_check
from feedback_store import load_feedback, save_feedback, normalize, previously_cleared_addresses
from ml_classifier import (
    update_training_data, can_train, train_model, predict as ml_predict,
    evaluate_model, compare_algorithms, top_features,
    MIN_SAMPLES, MIN_PER_CLASS,
)

st.set_page_config(page_title="Agreement Address Quality Checker", layout="wide")

# ----------------------------------------------------------------------
# Reference data
# ----------------------------------------------------------------------
INDIAN_STATES = [
    "ANDHRA PRADESH", "ARUNACHAL PRADESH", "ASSAM", "BIHAR", "CHHATTISGARH",
    "GOA", "GUJARAT", "HARYANA", "HIMACHAL PRADESH", "JHARKHAND", "KARNATAKA",
    "KERALA", "MADHYA PRADESH", "MAHARASHTRA", "MANIPUR", "MEGHALAYA",
    "MIZORAM", "NAGALAND", "ODISHA", "PUNJAB", "RAJASTHAN", "SIKKIM",
    "TAMIL NADU", "TELANGANA", "TRIPURA", "UTTAR PRADESH", "UTTARAKHAND",
    "WEST BENGAL", "DELHI", "JAMMU AND KASHMIR", "LADAKH", "PUDUCHERRY",
    "CHANDIGARH", "ANDAMAN AND NICOBAR", "DADRA AND NAGAR HAVELI",
    "DAMAN AND DIU", "LAKSHADWEEP",
]

COMMON_SAFE_LONG_WORDS = {"MAHARASHTRA", "TELANGANA", "CHHATTISGARH", "PONDICHERRY", "VISAKHAPATNAM"}

PLACEHOLDER_PHRASES = {
    "NA", "N A", "N/A", "N.A", "N.A.", "NIL", "NONE", "XXX", "XXXX", "XYZ", "ABC",
    "TEST", "TESTING", "TBD", "PENDING", "DUMMY", "SAMPLE", "DEFAULT", "UNKNOWN",
    "NOT AVAILABLE", "ADDRESS NOT AVAILABLE", "SAME AS ABOVE", "SAME AS PREVIOUS",
    "ASDF", "ASDFGH", "QWERTY",
}
PLACEHOLDER_WORDS = {"TEST", "TESTING", "TBD", "DUMMY", "SAMPLE", "ASDF", "ASDFGH", "QWERTY", "XYZ", "NIL"}

FOREIGN_LOCATION_HINTS = {
    "DUBAI", "UAE", "ABU DHABI", "SHARJAH", "SINGAPORE", "LONDON", "USA",
    "UNITED STATES", "UNITED KINGDOM", "CANADA", "AUSTRALIA", "NEPAL", "DOHA", "QATAR",
}

CRITICAL_ISSUE_PREFIXES = {
    "EMPTY_ADDRESS", "MISSING_PINCODE", "PINCODE_NOT_FOUND_IN_INDIA",
    "PLACEHOLDER_ADDRESS", "ADDRESS_TOO_SHORT",
}

ISSUE_DESCRIPTIONS = {
    "EMPTY_ADDRESS": "Address field is blank",
    "DOUBLE_COMMA_EMPTY_FIELD": "Contains ',,' - an empty field between commas",
    "PINCODE_DUPLICATED": "6-digit pincode appears twice back-to-back",
    "PINCODE_GLUED_TO_TEXT": "Pincode is stuck directly to a word with no space",
    "MISSING_PINCODE": "No 6-digit pincode found",
    "STATE_NOT_FOUND": "No recognizable Indian state name in the address",
    "ADDRESS_TOO_SHORT": "Address has very few words - likely incomplete",
    "POSSIBLE_MERGED_WORDS": "A long word may be two+ words stuck together",
    "HOUSE_NO_ZERO_OR_PLACEHOLDER": "House/flat number looks like a placeholder",
    "PINCODE_NOT_FOUND_IN_INDIA": "Pincode doesn't exist in the official India Post database",
    "PINCODE_STATE_MISMATCH": "Pincode belongs to a different state than what's written",
    "PLACEHOLDER_ADDRESS": "Entire address is a placeholder value (NA, TEST, etc.)",
    "PLACEHOLDER_WORD": "Contains a placeholder/junk word",
    "FOREIGN_LOCATION_MENTIONED": "Mentions a location outside India",
    "REPEATED_CHARACTER_RUN": "Same character repeated 4+ times in a row (e.g. aaaa)",
    "POSSIBLE_GIBBERISH_TEXT": "Long run of consonants suggests random/gibberish text",
    "AI_FLAGGED": "AI reviewer flagged this address",
    "ML_FLAGGED_PATTERN": "ML classifier judged this address's text patterns as issue-like",
}

DEMO_DATA = [
    ("AGR001", "ABHISHEK BUNGALOW NO. ONEKALPATARU NAGAR ASHOKA MARG , 422011"),
    ("AGR002", "SECTOR NO-4,CBD BELAPUR , NAVI MUMBAI400206"),
    ("AGR003", "FLAT NO- X, 5 TH FLOOR, BEACON CHSSOUTH AVENUEOPP RAMKRISHNA MISSION HOSPITAL, , SANTACRUZ-W, MUMBAI- 400054400054"),
    ("AGR004", "# 0, INSIDE NEW MARKET BAGGA MARKET , ,JAGADHRI YAMUNA NAGAR HARYANA - 135001"),
    ("AGR005", "# INDUSTRIEL AREA, , NEAR JODI FNAST ROAD YAMUNA NAGAR HARYANA - 135002"),
    ("AGR006", "# CHHACHHROULI ROAD, JAGADHRI, , YAMUNA NAGAR HARYANA - 135002"),
    ("AGR007", "YELAMANCHILI ROADATCHUT,APURAM, MAIN ROAD , ,MAIN ROAD531011"),
    ("AGR008", "12, GREEN PARK EXTENSION, NEW DELHI, DELHI - 110016"),
    ("AGR009", "MAIN ROAD 1, DUBAI"),
    ("AGR010", "NA"),
    ("AGR011", "FLAT 302 SUNRISE APARTMENTS MG ROAD BANGALORE KARNATAKA - 999999"),
]


def describe_issue(issue: str) -> str:
    base = issue.split("(")[0]
    return ISSUE_DESCRIPTIONS.get(base, base)


def severity_for(issue_codes):
    if not issue_codes:
        return "Clean"
    if any(i.split("(")[0] in CRITICAL_ISSUE_PREFIXES for i in issue_codes):
        return "Critical"
    return "Warning"


# ----------------------------------------------------------------------
# Layer 1 - structural rules
# ----------------------------------------------------------------------
def layer1_structural(addr: str, tokens, min_words: int, merge_len_threshold: int):
    issues = []
    if re.search(r",\s*,", addr):
        issues.append("DOUBLE_COMMA_EMPTY_FIELD")
    if re.search(r"(\d{6})\1", addr):
        issues.append("PINCODE_DUPLICATED")
    glued_match = re.search(r"[A-Za-z](\d{6})\b", addr)
    if glued_match and "PINCODE_DUPLICATED" not in issues:
        issues.append("PINCODE_GLUED_TO_TEXT")

    if not any(state in addr.upper() for state in INDIAN_STATES):
        issues.append("STATE_NOT_FOUND")

    phrase_counts = {}
    for n in (2, 3):
        for i in range(len(tokens) - n + 1):
            phrase = " ".join(tokens[i:i + n])
            phrase_counts[phrase] = phrase_counts.get(phrase, 0) + 1
    repeated = [p for p, c in phrase_counts.items() if c > 1 and len(p) > 6]
    if repeated:
        issues.append(f"REPEATED_PHRASE({repeated[0]})")

    if len(tokens) < min_words:
        issues.append("ADDRESS_TOO_SHORT")

    long_tokens = [t for t in tokens if len(t) >= merge_len_threshold and t not in COMMON_SAFE_LONG_WORDS]
    if long_tokens:
        issues.append(f"POSSIBLE_MERGED_WORDS({long_tokens[0]})")

    if re.search(r"#\s*0\b", addr):
        issues.append("HOUSE_NO_ZERO_OR_PLACEHOLDER")

    return issues


# ----------------------------------------------------------------------
# Layer 2 - pincode master-data validation
# ----------------------------------------------------------------------
def extract_pins(addr: str):
    pins = set(re.findall(r"\b\d{6}\b", addr))
    glued_pins = set(re.findall(r"[A-Za-z](\d{6})\b", addr))
    return pins | glued_pins


def layer2_issues_from_results(addr_upper: str, pins: set, pin_results: dict):
    """
    Turns already-fetched pincode lookup results into issues for one row.
    Pure/offline - no network call happens here, so this is cheap to run
    per-row even for very large files. Returns (issues, network_status).
    """
    if not pins:
        return [], "skipped"
    issues = []
    network_status = "ok"
    for pin in sorted(pins):
        result = pin_results.get(pin, "ERROR")
        if result is None:
            issues.append("PINCODE_NOT_FOUND_IN_INDIA")
        elif result == "ERROR":
            network_status = "error"
        else:
            actual_state = str(result.get("state", "")).upper()
            if actual_state and actual_state not in addr_upper:
                other_states = [s for s in INDIAN_STATES if s in addr_upper and s != actual_state]
                if other_states:
                    issues.append(f"PINCODE_STATE_MISMATCH(pin={pin} actual={actual_state} stated={other_states[0]})")
    return issues, network_status


# ----------------------------------------------------------------------
# Layer 3 - placeholder / gibberish / foreign-location dictionary
# ----------------------------------------------------------------------
def layer3_placeholder_gibberish(addr_upper: str, tokens):
    issues = []
    stripped = re.sub(r"[^A-Z ]", " ", addr_upper)
    stripped = re.sub(r"\s+", " ", stripped).strip()
    if stripped in PLACEHOLDER_PHRASES:
        issues.append("PLACEHOLDER_ADDRESS")

    hit = set(tokens) & PLACEHOLDER_WORDS
    if hit and "PLACEHOLDER_ADDRESS" not in issues:
        issues.append(f"PLACEHOLDER_WORD({sorted(hit)[0]})")

    foreign_hit = [f for f in FOREIGN_LOCATION_HINTS if f in addr_upper]
    if foreign_hit:
        issues.append(f"FOREIGN_LOCATION_MENTIONED({foreign_hit[0]})")

    if re.search(r"([A-Za-z0-9])\1{3,}", addr_upper):
        issues.append("REPEATED_CHARACTER_RUN")

    if re.search(r"[BCDFGHJKLMNPQRSTVWXYZ]{6,}", addr_upper):
        issues.append("POSSIBLE_GIBBERISH_TEXT")

    return issues


# ----------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------
def analyze_address_local(addr, min_words, merge_len_threshold):
    """
    Everything that does NOT need the network: Layers 1 & 3, plus pincode
    extraction. Safe and cheap to run per-row even for huge files - all
    regex/string work, no I/O. Returns (issues, addr_upper, pins).
    """
    if not isinstance(addr, str) or not addr.strip():
        return ["EMPTY_ADDRESS"], "", set()

    addr = addr.strip()
    addr_upper = addr.upper()
    tokens = re.findall(r"[A-Za-z]+", addr_upper)
    pins = extract_pins(addr)

    issues = []
    issues += layer1_structural(addr, tokens, min_words, merge_len_threshold)
    issues += layer3_placeholder_gibberish(addr_upper, tokens)

    if not pins and "MISSING_PINCODE" not in issues:
        issues.append("MISSING_PINCODE")

    return issues, addr_upper, pins


def _dedupe(issues):
    seen = set()
    out = []
    for i in issues:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


def process_dataframe(df, address_col, agreement_col, min_words, merge_len_threshold,
                       layer2_enabled, cleared_addresses, pincode_progress_callback=None,
                       pincode_workers=DEFAULT_BULK_WORKERS):
    """
    Two-phase processing so Layer 2 never makes a network call per row:
      Phase 1: run Layers 1 & 3 on every row (pure/offline, cheap even at
               100k+ rows) and collect every unique pincode mentioned.
      Phase 2: resolve ALL unique pincodes in one concurrent batch (see
               pincode_lookup.lookup_pincodes_bulk), then apply those
               already-fetched results back onto each row - no further I/O.
    Uses plain list iteration (not iterrows/itertuples) for phase 1/3, which
    is meaningfully faster at this scale since it avoids building a pandas
    Series per row.
    """
    addresses = df[address_col].tolist()
    agreements = df[agreement_col].tolist() if agreement_col else [""] * len(df)

    # Phase 1: local analysis + pin collection
    local_issues_list = []
    addr_upper_list = []
    pins_list = []
    all_pins = set()
    for addr in addresses:
        issues, addr_upper, pins = analyze_address_local(addr, min_words, merge_len_threshold)
        local_issues_list.append(issues)
        addr_upper_list.append(addr_upper)
        pins_list.append(pins)
        if pins:
            all_pins |= pins

    # Phase 2: one concurrent, deduped batch for every pincode in the file
    pin_results = {}
    if layer2_enabled and all_pins:
        pin_results = lookup_pincodes_bulk(all_pins, progress_callback=pincode_progress_callback,
                                            max_workers=pincode_workers)

    rows = []
    net_error_count = 0
    for addr, agreement, local_issues, addr_upper, pins in zip(
        addresses, agreements, local_issues_list, addr_upper_list, pins_list
    ):
        if layer2_enabled:
            l2_issues, net_status = layer2_issues_from_results(addr_upper, pins, pin_results)
        else:
            l2_issues, net_status = [], "skipped"
        if net_status == "error":
            net_error_count += 1

        issues = _dedupe(local_issues + l2_issues)

        cleared = isinstance(addr, str) and normalize(addr) in cleared_addresses
        if cleared:
            issues = []

        rows.append({
            "Agreement No": agreement,
            "Address": addr,
            "Severity": severity_for(issues) if not cleared else "Clean (reviewer-cleared)",
            "Issue Count": len(issues),
            "Issues": "; ".join(issues) if issues else "",
            "Issue Details": "; ".join(describe_issue(i) for i in issues) if issues else "",
        })
    return pd.DataFrame(rows), net_error_count




# ----------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------
st.title("📋 Agreement Address Quality Checker")
st.caption("4-layer pipeline: structural rules -> pincode master-data check -> "
           "placeholder/gibberish dictionary -> optional AI semantic review.")

if "feedback_df" not in st.session_state:
    st.session_state.feedback_df = load_feedback()

with st.sidebar:
    st.header("⚙️ Layer 1 settings")
    min_words = st.slider("Minimum words for a valid address", 2, 10, 5)
    merge_len_threshold = st.slider("Merged-word length threshold", 8, 20, 12)

    st.divider()
    st.header("🌐 Layer 2: Pincode master-data")
    layer2_enabled = st.checkbox("Enable pincode validation (needs internet)", value=True)
    st.caption("Checks every pincode against the official India Post directory "
               "(free public API). Catches fake pincodes and state mismatches, "
               "e.g. 'Dubai' with no valid Indian pincode. Pincodes are deduplicated and looked up "
               "concurrently - a 1-lakh-row file has at most ~19,000 unique Indian pincodes to actually "
               "fetch, no matter how many rows share them.")
    pincode_workers = 40
    if layer2_enabled:
        pincode_workers = st.slider(
            "Lookup concurrency (parallel requests)", 10, 80, 40, 5,
            help="Higher = faster on large files, at the cost of more simultaneous connections. "
                 "40 is a good default; lower it if your network is unstable."
        )
        ignore_ssl = st.checkbox(
            "Ignore SSL errors (corporate proxy)", value=False,
            help="Turn this on only if 'Test connection' below fails with a TLS/SSL error. Some office "
                 "networks/VPNs/antivirus intercept HTTPS with their own certificate, which requests "
                 "correctly rejects by default. This switches off certificate checking for pincode "
                 "lookups only (a public, read-only geographic dataset) so the app can still reach it. "
                 "The proper fix is to get your org's root CA .pem from IT and set the REQUESTS_CA_BUNDLE "
                 "environment variable before launching the app - that keeps full verification."
        )
        set_ssl_verify(not ignore_ssl)
        if ignore_ssl:
            st.caption("⚠️ Certificate verification is off for the pincode API. Fine on a trusted office "
                       "network; avoid this on public wifi.")
        if st.button("🔌 Test connection to pincode API"):
            with st.spinner("Checking..."):
                result = check_connectivity()
            if result["ok"]:
                st.success(result["message"])
            else:
                st.error(result["message"])
                st.caption("If this keeps failing: check your internet, disable any VPN, or ask your network "
                           "team to allow `aniket-thapa.github.io` (used for Layer 2 only). Layers 1, 3, 4, and 5 "
                           "don't need this and will still work fully without it.")

    st.divider()
    st.header("📖 Layer 3: Placeholder dictionary")
    st.caption("Always on, free, offline. Catches NA/TEST/XXX-style junk, "
               "foreign city/country mentions, repeated characters, gibberish runs.")

    st.divider()
    st.header("🧠 Layer 4: ML pattern classifier")
    use_ml = st.checkbox("Enable ML semantic pattern check", value=True)
    ml_algorithm = "logreg"
    ml_threshold = 0.5
    ml_auto_select = True
    if use_ml:
        ml_auto_select = st.checkbox(
            "Auto-select best model (cross-validates both, picks higher F1)", value=True
        )
        if not ml_auto_select:
            ml_algo_label = st.radio("Model", ["Logistic Regression", "Naive Bayes"], index=0)
            ml_algorithm = "logreg" if ml_algo_label == "Logistic Regression" else "nb"
        ml_threshold = st.slider("Flag threshold (issue probability)", 0.3, 0.9, 0.5, 0.05)
        st.caption(
            "TF-IDF (word + char n-grams) plus hand-engineered stats (length, digit ratio, "
            "repeated-character runs, glued digits/letters, vowel ratio...) feeding Logistic "
            "Regression or Naive Bayes - trained offline in under a second, no API, no internet, no cost. "
            "It self-trains from every address this tool has ever scored: Layers 1-3's Severity supplies "
            "bootstrap labels, and any 'Confirmed Issue' / 'False Positive' decision you save in the review "
            "queue overrides the bootstrap label for that address. Needs at least "
            f"{MIN_SAMPLES} labeled addresses ({MIN_PER_CLASS}+ of each class) before it activates - "
            "until then it sits out and tells you why. Training data accumulates in "
            "`ml_training_data.csv` next to the app, same as `reviewer_feedback.csv`. Reported accuracy "
            "comes from stratified cross-validation on held-out folds, not the training fit."
        )

    st.divider()
    st.header("🤖 Layer 5: Optional AI review (Gemini)")
    use_ai = st.checkbox("Enable AI semantic review")
    api_key = ""
    ai_scope = "Flagged rows only"
    sample_size = 20
    if use_ai:
        api_key = st.text_input("Gemini API key", type="password",
                                 help="Free key: https://aistudio.google.com/app/apikey")
        ai_scope = st.radio("Run AI on", ["Flagged rows only", "Random sample", "All rows"], index=0)
        if ai_scope == "Random sample":
            sample_size = st.number_input("Sample size", 5, 200, 20)
        st.caption("Your address data is sent to Google's API for this step only. "
                   "Free tier has rate limits, so requests are paced automatically."
        )

st.subheader("1. Load data")
col_a, col_b = st.columns([2, 1])
with col_a:
    uploaded = st.file_uploader("Upload Excel file (.xlsx)", type=["xlsx", "xls"])
with col_b:
    st.write("")
    st.write("")
    use_demo = st.button("▶ Try demo data instead")

def read_excel_fast(uploaded_file) -> pd.DataFrame:
    """
    python-calamine reads .xlsx roughly 5-10x faster than the default
    openpyxl engine (openpyxl walks the XML DOM; calamine is a native
    Rust reader) - meaningful once files reach tens of thousands of rows.
    Falls back to openpyxl automatically if calamine isn't installed or
    the file format trips it up, so this never breaks reading a valid file.
    """
    try:
        uploaded_file.seek(0)
        return pd.read_excel(uploaded_file, engine="calamine")
    except Exception:
        uploaded_file.seek(0)
        return pd.read_excel(uploaded_file, engine="openpyxl")


df = None
if uploaded is not None:
    with st.spinner("Reading file..."):
        df = read_excel_fast(uploaded)
    st.caption(f"Loaded {len(df):,} rows.")
elif use_demo:
    df = pd.DataFrame(DEMO_DATA, columns=["Agreement No", "Address"])
    st.session_state["demo_loaded"] = True
elif st.session_state.get("demo_loaded"):
    df = pd.DataFrame(DEMO_DATA, columns=["Agreement No", "Address"])

if df is not None:
    st.success(f"Loaded {len(df)} rows.")
    st.dataframe(df.head(), use_container_width=True)

    st.subheader("2. Select columns")
    cols = list(df.columns)

    def guess(col_list, keyword):
        for c in col_list:
            if keyword in str(c).lower():
                return c
        return col_list[0]

    c1, c2 = st.columns(2)
    with c1:
        agreement_col = st.selectbox("Agreement No. column", cols, index=cols.index(guess(cols, "agree")))
    with c2:
        address_col = st.selectbox("Address column", cols, index=cols.index(guess(cols, "address")))

    if st.button("🔍 Run address check", type="primary"):
        cleared_addresses = previously_cleared_addresses(st.session_state.feedback_df)

        progress_bar = st.progress(0.0, text=f"Checking {len(df):,} addresses (Layers 1 & 3)...")

        def _pincode_progress(done, total):
            frac = done / total if total else 1.0
            progress_bar.progress(frac, text=f"Resolving pincodes: {done:,} / {total:,} unique pincodes")

        result_df, net_error_count = process_dataframe(
            df, address_col, agreement_col, min_words, merge_len_threshold,
            layer2_enabled, cleared_addresses,
            pincode_progress_callback=_pincode_progress if layer2_enabled else None,
            pincode_workers=pincode_workers,
        )
        progress_bar.empty()

        if net_error_count:
            pct = net_error_count / len(result_df) * 100 if len(result_df) else 0
            cstatus = circuit_status()
            st.warning(
                f"⚠️ Layer 2 (pincode API) couldn't be reached for {net_error_count} of {len(result_df)} "
                f"rows ({pct:.0f}%). Those rows were still checked with Layers 1 & 3 (and 4/5 if enabled) - "
                "only the pincode master-data check was skipped for them."
            )
            with st.expander("Why did this happen, and what can I do?"):
                if cstatus["open"]:
                    st.write(
                        f"The client detected {cstatus['consecutive_failures']} consecutive network failures "
                        "and is currently fast-failing (circuit breaker open) to avoid slowing the batch down "
                        f"further with repeated timeouts. It will retry automatically in "
                        f"~{cstatus['cooldown_seconds_left']:.0f}s, or you can force a retry now."
                    )
                st.write(
                    "Common causes: no internet access from this machine, a VPN or corporate proxy/firewall "
                    "blocking `aniket-thapa.github.io` (the free pincode API host), or a temporary API blip. "
                    "Rows that already succeeded are cached, so a retry will only re-check the ones that failed."
                )
                col_a, col_b = st.columns(2)
                with col_a:
                    if st.button("🔌 Test connection now"):
                        with st.spinner("Checking..."):
                            diag = check_connectivity()
                        if diag["ok"]:
                            st.success(diag["message"])
                        else:
                            st.error(diag["message"])
                with col_b:
                    if st.button("🔁 Reset & retry Layer 2"):
                        reset_circuit()
                        st.info("Connection state reset. Click '🔍 Run address check' again - "
                                 "previously successful pincode lookups are cached, so only the "
                                 "failed rows will be re-checked.")

        # ---------------- Layer 4: ML pattern classifier ----------------
        if use_ml:
            training_df = update_training_data(result_df, st.session_state.feedback_df)
            ok, reason = can_train(training_df)
            if not ok:
                st.info(f"🧠 Layer 4 (ML classifier) not active yet: {reason}")
            else:
                with st.spinner("Cross-validating and training Layer 4 classifier..."):
                    if ml_auto_select:
                        comparison = compare_algorithms(training_df)
                        ml_algorithm = max(comparison, key=lambda a: comparison[a]["f1"])
                        metrics = comparison[ml_algorithm]
                    else:
                        metrics = evaluate_model(training_df, ml_algorithm)
                        comparison = {ml_algorithm: metrics}

                    model = train_model(training_df, algorithm=ml_algorithm)
                    predictions = ml_predict(result_df["Address"].tolist(), model)

                ml_probs = [p for _, p in predictions]
                result_df["ML_Issue_Probability"] = [round(p, 3) for p in ml_probs]

                new_issues_col = []
                new_details_col = []
                new_severity_col = []
                new_count_col = []
                for i, row in result_df.iterrows():
                    issues = [x for x in str(row["Issues"]).split("; ") if x]
                    prob = ml_probs[i]
                    is_human_cleared = row["Severity"] == "Clean (reviewer-cleared)"
                    if (not is_human_cleared and prob >= ml_threshold
                            and not any(x.startswith("ML_FLAGGED_PATTERN") for x in issues)):
                        issues = issues + [f"ML_FLAGGED_PATTERN(p={prob:.2f})"]
                    new_issues_col.append("; ".join(issues))
                    new_details_col.append("; ".join(describe_issue(x) for x in issues) if issues else "")
                    if is_human_cleared:
                        new_severity_col.append(row["Severity"])  # human clearance always wins
                    else:
                        new_severity_col.append(severity_for(issues))
                    new_count_col.append(len(issues))

                result_df["Issues"] = new_issues_col
                result_df["Issue Details"] = new_details_col
                result_df["Severity"] = new_severity_col
                result_df["Issue Count"] = new_count_col

                algo_label = "Logistic Regression" if ml_algorithm == "logreg" else "Naive Bayes"
                st.caption(
                    f"🧠 Layer 4 trained on {len(training_df)} labeled addresses "
                    f"({(training_df['Label'] == 0).sum()} clean, {(training_df['Label'] == 1).sum()} flagged) "
                    f"using {algo_label}"
                    + (" (auto-selected over Naive Bayes)" if ml_auto_select and ml_algorithm == "logreg" else "")
                    + (" (auto-selected over Logistic Regression)" if ml_auto_select and ml_algorithm == "nb" else "")
                    + "."
                )

                with st.expander("📊 Layer 4 model performance (cross-validated, out-of-fold)"):
                    if len(comparison) > 1:
                        cmp_rows = []
                        for alg, m in comparison.items():
                            cmp_rows.append({
                                "Model": "Logistic Regression" if alg == "logreg" else "Naive Bayes",
                                "Accuracy": round(m["accuracy"], 3),
                                "Precision": round(m["precision"], 3),
                                "Recall": round(m["recall"], 3),
                                "F1": round(m["f1"], 3),
                                "Selected": "✅" if alg == ml_algorithm else "",
                            })
                        st.dataframe(pd.DataFrame(cmp_rows), use_container_width=True, hide_index=True)
                    else:
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Accuracy", f"{metrics['accuracy']:.1%}")
                        c2.metric("Precision", f"{metrics['precision']:.1%}")
                        c3.metric("Recall", f"{metrics['recall']:.1%}")
                        c4.metric("F1", f"{metrics['f1']:.1%}")

                    st.caption(
                        f"Estimated via {metrics['n_splits']}-fold stratified cross-validation on "
                        f"{metrics['n_samples']} labeled addresses - every prediction used to compute these "
                        "numbers came from a fold the model did NOT see during that fold's training, so this "
                        "isn't the model grading its own homework. Precision = of the rows it flagged, how "
                        "many were actually issues. Recall = of the actual issues, how many it caught."
                    )
                    if metrics["n_samples"] < 50:
                        st.caption(
                            "⚠️ Small sample size - treat these numbers as a rough signal, not a guarantee. "
                            "They'll stabilize as more labeled addresses accumulate."
                        )

                    cm = metrics["confusion_matrix"]
                    cm_df = pd.DataFrame(
                        cm, index=["Actual: Clean", "Actual: Issue"],
                        columns=["Predicted: Clean", "Predicted: Issue"]
                    )
                    st.write("**Confusion matrix**")
                    st.dataframe(cm_df, use_container_width=True)

                    st.write("**What the model actually learned (top predictive features)**")
                    top_issue, top_clean = top_features(model, n=10)
                    fc1, fc2 = st.columns(2)
                    with fc1:
                        st.caption("Pushes toward **Issue**")
                        st.dataframe(
                            pd.DataFrame(top_issue, columns=["Feature", "Weight"]).assign(
                                Weight=lambda d: d["Weight"].round(3)
                            ),
                            use_container_width=True, hide_index=True,
                        )
                    with fc2:
                        st.caption("Pushes toward **Clean**")
                        st.dataframe(
                            pd.DataFrame(top_clean, columns=["Feature", "Weight"]).assign(
                                Weight=lambda d: d["Weight"].round(3)
                            ),
                            use_container_width=True, hide_index=True,
                        )
                    st.caption(
                        "`stat__...` features are the hand-engineered numeric signals (length, digit ratio, "
                        "repeated-character runs, etc). Everything else is a word or character n-gram the "
                        "model found predictive in your own data."
                    )

        flagged_df = result_df[result_df["Severity"].isin(["Critical", "Warning"])].reset_index(drop=True)
        clean_count = len(result_df) - len(flagged_df)

        st.subheader("3. Results")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total records", len(result_df))
        m2.metric("Critical", int((result_df["Severity"] == "Critical").sum()))
        m3.metric("Warning", int((result_df["Severity"] == "Warning").sum()))
        m4.metric("Clean", clean_count)

        if len(flagged_df) > 0:
            issue_counts = {}
            for issues in flagged_df["Issues"]:
                for i in issues.split("; "):
                    if not i:
                        continue
                    base = i.split("(")[0]
                    issue_counts[base] = issue_counts.get(base, 0) + 1
            st.bar_chart(pd.Series(issue_counts, name="Count"))

        # ---------------- Layer 5: AI review ----------------
        if use_ai and api_key and len(result_df) > 0:
            if ai_scope == "Flagged rows only":
                target_df = flagged_df
            elif ai_scope == "Random sample":
                target_df = result_df.sample(min(int(sample_size), len(result_df)))
            else:
                target_df = result_df
            ai_targets = target_df.index.tolist()

            if len(ai_targets) > 0:
                st.subheader("🤖 Layer 5: AI review")
                progress = st.progress(0.0, text="Calling Gemini...")
                ai_results = {}
                for n, idx in enumerate(ai_targets):
                    addr = target_df.loc[idx, "Address"]
                    if isinstance(addr, str) and addr.strip():
                        res = gemini_check(addr, api_key)
                    else:
                        res = {"verdict": "issue", "severity": "Critical", "category": "empty", "reason": "empty address"}
                    ai_results[idx] = res
                    progress.progress((n + 1) / len(ai_targets), text=f"Calling Gemini... {n+1}/{len(ai_targets)}")
                    time.sleep(1.1)  # pacing for free-tier rate limits
                progress.empty()

                ai_df = pd.DataFrame.from_dict(ai_results, orient="index")
                ai_df.columns = [f"AI_{c}" for c in ai_df.columns]
                result_df = result_df.join(ai_df)
                flagged_df = result_df[
                    result_df["Severity"].isin(["Critical", "Warning"]) |
                    (result_df.get("AI_verdict") == "issue")
                ].reset_index(drop=True)

        st.markdown("**Flagged addresses**")
        st.dataframe(flagged_df, use_container_width=True)

        st.markdown("**Full results**")
        st.dataframe(result_df, use_container_width=True)

        st.session_state["last_flagged_df"] = flagged_df
        st.session_state["last_result_df"] = result_df

        def to_excel_bytes(d):
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                d.to_excel(writer, index=False, sheet_name="Results")
            return buf.getvalue()

        dl1, dl2 = st.columns(2)
        with dl1:
            st.download_button("⬇ Download flagged only (.xlsx)", data=to_excel_bytes(flagged_df),
                                file_name="flagged_addresses.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        with dl2:
            st.download_button("⬇ Download full results (.xlsx)", data=to_excel_bytes(result_df),
                                file_name="all_address_results.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    # ---------------- Review queue ----------------
    if "last_flagged_df" in st.session_state and len(st.session_state["last_flagged_df"]) > 0:
        st.subheader("4. Review queue (human-in-the-loop)")
        st.caption("Mark false positives here. They're remembered on future runs so you "
                   "don't have to re-review the same address twice.")

        review_input = st.session_state["last_flagged_df"][["Agreement No", "Address", "Severity", "Issue Details"]].copy()
        review_input["Decision"] = "Pending"
        edited = st.data_editor(
            review_input,
            column_config={
                "Decision": st.column_config.SelectboxColumn(
                    options=["Pending", "Confirmed Issue", "False Positive"]
                )
            },
            use_container_width=True,
            key="review_editor",
        )

        if st.button("💾 Save reviewer decisions"):
            decided = edited[edited["Decision"] != "Pending"].copy()
            decided["Notes"] = ""
            new_feedback = decided[["Agreement No", "Address", "Decision", "Notes"]]
            combined = pd.concat([st.session_state.feedback_df, new_feedback], ignore_index=True)
            combined = combined.drop_duplicates(subset=["Address", "Decision"], keep="last")
            st.session_state.feedback_df = combined
            save_feedback(combined)
            st.success(f"Saved {len(new_feedback)} decisions. They'll auto-apply on future runs.")
else:
    st.info("Upload an Excel file above, or click 'Try demo data' to see it in action.")
