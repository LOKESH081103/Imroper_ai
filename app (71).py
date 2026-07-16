"""
Agreement Address Quality Checker - 4-layer pipeline
-----------------------------------------------------
Layer 1: Structural/mechanical rules (free, offline, instant)
Layer 2: Pincode master-data validation (free public API, needs internet)
Layer 3: Placeholder / gibberish / foreign-location dictionary (free, offline)
Layer 4: ML pattern classifier - TF-IDF + XGBoost/Logistic Regression/Naive
         Bayes (free, offline). This is the only "intelligence" layer now -
         there is no external AI/LLM call anywhere in this pipeline, so
         nothing here is rate-limited or capped by a free-tier quota. It
         scales to hundreds of thousands of rows in seconds.

Run:
    pip install -r requirements.txt
    streamlit run app.py
"""

import io
import re

import pandas as pd
import streamlit as st

from pincode_lookup import (
    lookup_pincodes_bulk, check_connectivity, circuit_status, reset_circuit,
    DEFAULT_BULK_WORKERS,
)
from feedback_store import load_feedback, save_feedback, normalize, previously_cleared_addresses
from ml_classifier import (
    update_training_data, can_train, train_model, predict as ml_predict,
    evaluate_model, compare_algorithms, top_features, import_labeled_dataset,
    training_data_summary, load_training_data,
    MIN_SAMPLES, MIN_PER_CLASS, ALGORITHMS, DEFAULT_ALGORITHM,
)
from rule_engine import (
    INDIAN_STATES, COMMON_SAFE_LONG_WORDS, PLACEHOLDER_PHRASES, PLACEHOLDER_WORDS,
    FOREIGN_LOCATION_HINTS, CRITICAL_ISSUE_PREFIXES, ISSUE_DESCRIPTIONS, DEMO_DATA,
    describe_issue, severity_for, layer1_structural, extract_pins,
    layer2_issues_from_results, layer3_placeholder_gibberish, analyze_address_local,
    dedupe,
)
from dashboard_builder import summarize_from_result_df, render_html

st.set_page_config(page_title="Agreement Address Quality Checker", layout="wide")

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

        issues = dedupe(local_issues + l2_issues)

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
           "placeholder/gibberish dictionary -> self-trained ML classifier (XGBoost). "
           "100% offline after upload - no external AI calls, no rate limits, no quotas.")

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
        if st.button("🔌 Test connection to pincode API"):
            with st.spinner("Checking..."):
                result = check_connectivity()
            if result["ok"]:
                st.success(result["message"])
            else:
                st.error(result["message"])
                st.caption("If this keeps failing: check your internet, disable any VPN, or ask your network "
                           "team to allow `aniket-thapa.github.io` (used for Layer 2 only). Layers 1, 3, and 4 "
                           "don't need this and will still work fully without it.")

    st.divider()
    st.header("📖 Layer 3: Placeholder dictionary")
    st.caption("Always on, free, offline. Catches NA/TEST/XXX-style junk, "
               "foreign city/country mentions, repeated characters, gibberish runs.")

    st.divider()
    st.header("🧠 Layer 4: ML pattern classifier")
    use_ml = st.checkbox("Enable ML semantic pattern check", value=True)
    ml_algorithm = DEFAULT_ALGORITHM
    ml_threshold = 0.5
    ml_auto_select = True
    ALGO_LABELS = {"xgb": "XGBoost", "logreg": "Logistic Regression", "nb": "Naive Bayes"}
    if use_ml:
        ml_auto_select = st.checkbox(
            f"Auto-select best model (cross-validates all {len(ALGORITHMS)}, picks higher F1)", value=True
        )
        if not ml_auto_select:
            ml_algo_label = st.radio("Model", [ALGO_LABELS[a] for a in ALGORITHMS], index=0)
            ml_algorithm = {v: k for k, v in ALGO_LABELS.items()}[ml_algo_label]
        ml_threshold = st.slider("Flag threshold (issue probability)", 0.3, 0.9, 0.5, 0.05)
        st.caption(
            "TF-IDF (word + char n-grams) plus hand-engineered stats (length, digit ratio, "
            "repeated-character runs, glued digits/letters, vowel ratio...) feeding XGBoost "
            "(gradient-boosted trees), Logistic Regression, or Naive Bayes - trained offline in "
            "seconds, no API, no internet, no cost. XGBoost is the default: it captures nonlinear "
            "combinations of these signals (e.g. 'short AND foreign word AND low digit ratio') that "
            "a linear model can't express directly, which usually means better F1 on this kind of "
            "messy, rule-adjacent text. It self-trains from every address this tool has ever scored: "
            "Layers 1-3's Severity supplies bootstrap labels, and any 'Confirmed Issue' / "
            "'False Positive' decision you save in the review queue overrides the bootstrap label for "
            "that address. Needs at least "
            f"{MIN_SAMPLES} labeled addresses ({MIN_PER_CLASS}+ of each class) before it activates - "
            "until then it sits out and tells you why. Training data accumulates in "
            "`ml_training_data.csv` next to the app, same as `reviewer_feedback.csv`. Reported accuracy "
            "comes from stratified cross-validation on held-out folds, not the training fit. Prediction "
            "on the full uploaded file is a single vectorized pass - it stays fast even at 8+ lakh rows."
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

with st.expander("📥 Import a labeled dataset to train Layer 4 (e.g. a 'type' column of proper/improper)"):
    st.caption(
        "Already have a file where each address is labeled proper/improper (or clean/issue, "
        "valid/invalid - any wording works)? Upload it here to feed those labels straight into "
        "Layer 4's training data as ground truth. Unlike the automatic bootstrap labels Layers 1-3 "
        "generate, these are treated as authoritative: never overwritten by a later bootstrap pass, "
        "and never trimmed by the training-set size cap - a reviewer decision on one specific address "
        "in the review queue below is the only thing that can still override an imported label, for "
        "that address."
    )
    label_file = st.file_uploader(
        "Labeled dataset (.xlsx, .xls, or .csv)", type=["xlsx", "xls", "csv"], key="label_import_uploader"
    )
    label_df = None
    if label_file is not None:
        try:
            if label_file.name.lower().endswith(".csv"):
                label_df = pd.read_csv(label_file, keep_default_na=False, na_values=[])
            else:
                label_df = pd.read_excel(label_file, keep_default_na=False, na_values=[])
        except Exception as e:
            st.error(f"Couldn't read that file: {e}")

    if label_df is not None and len(label_df) > 0:
        label_cols = list(label_df.columns)
        lc1, lc2 = st.columns(2)
        # best-effort default guesses so the common case needs zero clicks
        addr_guess = next((c for c in label_cols if "address" in c.lower()), label_cols[0])
        type_guess = next(
            (c for c in label_cols if c.lower() in ("type", "label", "status", "decision", "category")),
            label_cols[min(1, len(label_cols) - 1)],
        )
        addr_col = lc1.selectbox("Address column", label_cols, index=label_cols.index(addr_guess))
        type_col = lc2.selectbox("Label column", label_cols, index=label_cols.index(type_guess))

        distinct_vals = sorted(label_df[type_col].astype(str).str.strip().unique().tolist())
        default_improper = [
            v for v in distinct_vals
            if v.lower() in ("improper", "issue", "invalid", "bad", "flagged", "problem", "1", "true", "yes")
        ]
        improper_vals = st.multiselect(
            "Which value(s) mean 'improper / issue'? (everything else in this column is treated as 'proper / clean')",
            distinct_vals,
            default=default_improper,
        )

        if improper_vals:
            n_improper = label_df[type_col].astype(str).str.strip().isin(improper_vals).sum()
            st.caption(f"{len(label_df):,} rows -> {n_improper:,} would be labeled **improper/issue**, "
                       f"{len(label_df) - n_improper:,} would be labeled **proper/clean**.")
        else:
            st.warning("Select at least one value that means 'improper' - otherwise every row would "
                       "import as 'clean', which is almost certainly not what you want.")

        if st.button("📥 Import into Layer 4 training data", type="primary", disabled=not improper_vals):
            updated = import_labeled_dataset(label_df, addr_col, type_col, positive_values=improper_vals)
            s = training_data_summary(updated)
            st.success(
                f"Imported. Layer 4's training set now has {s['total']:,} labeled addresses total: "
                f"{s['imported']:,} imported, {s['human']:,} from the review queue, {s['rule']:,} bootstrap "
                f"({s['clean']:,} clean / {s['issue']:,} issue overall). Run a check below and Layer 4 will "
                "train on this immediately."
            )

    existing_summary = training_data_summary(load_training_data())
    if existing_summary["total"] > 0:
        st.caption(
            f"Current Layer 4 training set: {existing_summary['total']:,} labeled addresses "
            f"({existing_summary['imported']:,} imported, {existing_summary['human']:,} reviewed, "
            f"{existing_summary['rule']:,} bootstrap)."
        )

if df is not None:
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
        comparison = None
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

                algo_label = ALGO_LABELS.get(ml_algorithm, ml_algorithm)
                auto_note = ""
                if ml_auto_select and len(comparison) > 1:
                    others = ", ".join(ALGO_LABELS.get(a, a) for a in comparison if a != ml_algorithm)
                    auto_note = f" (auto-selected over {others})"
                st.caption(
                    f"🧠 Layer 4 trained on {len(training_df)} labeled addresses "
                    f"({(training_df['Label'] == 0).sum()} clean, {(training_df['Label'] == 1).sum()} flagged) "
                    f"using {algo_label}{auto_note}."
                )

                with st.expander("📊 Layer 4 model performance (cross-validated, out-of-fold)"):
                    if len(comparison) > 1:
                        cmp_rows = []
                        for alg, m in comparison.items():
                            cmp_rows.append({
                                "Model": ALGO_LABELS.get(alg, alg),
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

        # ---------------- Visual dashboard ----------------
        st.markdown("#### 📊 Dashboard")
        dash_c1, dash_c2 = st.columns(2)
        with dash_c1:
            st.caption("Severity breakdown")
            sev_series = result_df["Severity"].replace(
                "Clean (reviewer-cleared)", "Clean"
            ).value_counts().reindex(["Critical", "Warning", "Clean"]).fillna(0)
            st.bar_chart(sev_series)
        with dash_c2:
            if len(flagged_df) > 0:
                st.caption("Why addresses got flagged")
                issue_counts = {}
                for issues in flagged_df["Issues"]:
                    for i in issues.split("; "):
                        if not i:
                            continue
                        base = i.split("(")[0]
                        issue_counts[base] = issue_counts.get(base, 0) + 1
                st.bar_chart(pd.Series(issue_counts, name="Count").sort_values(ascending=False).head(10))
            else:
                st.caption("Why addresses got flagged")
                st.info("Nothing flagged - every address came back Clean.")

        if comparison and len(comparison) > 0:
            st.caption("Layer 4 model comparison (cross-validated)")
            model_cmp_df = pd.DataFrame({
                ALGO_LABELS.get(alg, alg): {
                    "Accuracy": m["accuracy"] * 100, "Precision": m["precision"] * 100,
                    "Recall": m["recall"] * 100, "F1": m["f1"] * 100,
                }
                for alg, m in comparison.items()
            }).T
            st.bar_chart(model_cmp_df)

        dash_data = summarize_from_result_df(result_df, model_comparison=comparison, layer2_included=layer2_enabled)
        dash_html = render_html(dash_data, source_name=uploaded.name if uploaded else "demo data",
                                 title="Address Quality Report")
        st.download_button(
            "⬇ Download presentation dashboard (.html)", data=dash_html,
            file_name="address_quality_dashboard.html", mime="text/html",
            help="A single self-contained HTML file with these same charts, full-screen and presentation-ready "
                 "- open it in any browser, no Streamlit needed."
        )

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
