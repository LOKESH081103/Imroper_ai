"""
dashboard_builder.py - shared rendering code for the presentation dashboard.

Used by:
  - app.py (Streamlit): builds the same dashboard inline after a run, from
    the already-computed result_df (so Layer 2 results, if enabled, are
    included for free - no re-running any rules).
  - dashboard_report.py (CLI): builds it from a raw address file with no
    Streamlit involved, for offline/scheduled use.

Both produce the exact same HTML/JS - one template, no drift between the
in-app version and the standalone file version.
"""

import json
from collections import Counter
from datetime import datetime
from typing import Optional

import pandas as pd

ALGO_LABELS = {"xgb": "XGBoost", "logreg": "Logistic Regression", "nb": "Naive Bayes"}


def summarize_from_result_df(result_df: pd.DataFrame, model_comparison: Optional[dict] = None,
                              layer2_included: bool = False) -> dict:
    """
    Build the same summary `data` dict that dashboard_report.py's
    build_report_data() produces, but from an already-processed result_df
    (columns: Severity, Issues) coming out of app.py's process_dataframe -
    so Layer 2 (and Layer 4, if it ran) are naturally reflected, with no
    re-computation and no risk of drifting from what the app actually found.
    """
    total = len(result_df)
    sev_counts = result_df["Severity"].value_counts()
    # "Clean (reviewer-cleared)" is still clean for this rollup
    clean = int(sev_counts.get("Clean", 0)) + int(sev_counts.get("Clean (reviewer-cleared)", 0))
    critical = int(sev_counts.get("Critical", 0))
    warning = int(sev_counts.get("Warning", 0))

    issue_counts = Counter()
    for issues in result_df["Issues"].fillna(""):
        for code in str(issues).split("; "):
            if code:
                issue_counts[code.split("(")[0]] += 1
    top_issues = issue_counts.most_common(10)

    model_comparison_out = None
    if model_comparison:
        model_comparison_out = {
            alg: {k: round(v * 100, 2) for k, v in r.items() if k in ("accuracy", "precision", "recall", "f1")}
            for alg, r in model_comparison.items()
        }

    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "total": total,
        "layer2_included": layer2_included,
        "has_ground_truth": False,
        "ground_truth": None,
        "confusion": None,
        "severity": {"critical": critical, "warning": warning, "clean": clean},
        "rule_flagged": critical + warning,
        "top_issues": top_issues,
        "model_comparison": model_comparison_out,
    }


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<style>
  :root {{
    --bg: #fcfcfb; --card: #ffffff; --border: #e1e0d9; --text: #0b0b0b;
    --muted: #52514e; --faint: #898781; --red: #e34948; --blue: #2a78d6;
    --amber: #eda100; --purple: #4a3aa7; --green: #008300;
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{
      --bg: #1a1a19; --card: #232322; --border: #383835; --text: #ffffff;
      --muted: #c3c2b7; --faint: #898781;
    }}
  }}
  body {{ font-family: -apple-system, "Segoe UI", Roboto, sans-serif; background: var(--bg); color: var(--text);
          margin: 0; padding: 32px 40px 60px; }}
  h1 {{ font-size: 22px; font-weight: 500; margin: 0 0 4px; }}
  h2 {{ font-size: 16px; font-weight: 500; margin: 32px 0 12px; }}
  .meta {{ font-size: 13px; color: var(--faint); margin: 0 0 20px; }}
  .banner {{ display: flex; gap: 10px; align-items: center; padding: 10px 14px; background: rgba(237,161,0,0.12);
             border: 0.5px solid var(--amber); border-radius: 8px; font-size: 13px; color: var(--amber); margin-bottom: 24px; }}
  .kpis {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px,1fr)); gap: 12px; }}
  .card {{ background: var(--card); border: 0.5px solid var(--border); border-radius: 12px; padding: 16px; }}
  .card .label {{ font-size: 13px; color: var(--muted); margin: 0 0 6px; }}
  .card .value {{ font-size: 26px; font-weight: 500; margin: 0; }}
  .charts-row {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }}
  .chart-wrap {{ position: relative; }}
  table {{ width: 100%; font-size: 13px; border-collapse: collapse; }}
  td, th {{ padding: 8px; text-align: center; border-bottom: 0.5px solid var(--border); }}
  td:first-child, th:first-child {{ text-align: left; color: var(--muted); }}
  .legend {{ display: flex; flex-wrap: wrap; gap: 16px; font-size: 12px; color: var(--muted); margin-bottom: 8px; }}
  .dot {{ width: 10px; height: 10px; border-radius: 2px; display: inline-block; margin-right: 4px; }}
  .note {{ font-size: 12px; color: var(--faint); }}
</style>
</head>
<body>
  <h1>{title}</h1>
  <p class="meta">Generated {generated_at} - source file: {source_name} ({total} rows)</p>
  {banner}

  <div class="kpis">
    <div class="card"><p class="label">Total addresses</p><p class="value">{total_fmt}</p></div>
    {gt_cards}
    <div class="card"><p class="label">Flagged{flagged_label_suffix}</p><p class="value">{rule_flagged_fmt} <span style="font-size:14px;color:var(--muted)">({rule_flagged_pct}%)</span></p></div>
    {accuracy_card}
  </div>

  <div class="charts-row" style="margin-top:24px;">
    <div>
      <div class="legend">{severity_legend}</div>
      <div class="chart-wrap" style="height:220px;"><canvas id="sevChart"></canvas></div>
    </div>
    {truth_chart_block}
  </div>

  <h2>Why addresses got flagged</h2>
  <div class="chart-wrap" style="height:{issue_chart_height}px;"><canvas id="issueChart"></canvas></div>

  {confusion_block}
  {model_block}

<script>
const isDark = matchMedia('(prefers-color-scheme: dark)').matches;
const muted = isDark ? '#898781' : '#898781';
const grid = isDark ? '#383835' : '#e1e0d9';
const ring = isDark ? '#232322' : '#ffffff';
const common = {{ responsive: true, maintainAspectRatio: false }};

new Chart(document.getElementById('sevChart'), {{
  type: 'doughnut',
  data: {{ labels: ['Critical','Warning','Clean'],
    datasets: [{{ data: {sev_data}, backgroundColor: ['#e34948','#eda100','#2a78d6'], borderWidth: 2, borderColor: ring }}] }},
  options: {{ ...common, plugins: {{ legend: {{ display: false }} }}, cutout: '65%' }}
}});

new Chart(document.getElementById('issueChart'), {{
  type: 'bar',
  data: {{ labels: {issue_labels}, datasets: [{{ data: {issue_values}, backgroundColor: '#4a3aa7', borderRadius: 4, barThickness: 20 }}] }},
  options: {{ ...common, indexAxis: 'y', plugins: {{ legend: {{ display: false }} }},
    scales: {{ x: {{ grid: {{ color: grid }}, ticks: {{ color: muted }} }}, y: {{ grid: {{ display: false }}, ticks: {{ color: muted }} }} }} }}
}});

{truth_chart_js}
{model_chart_js}
</script>
</body>
</html>
"""


def render_html(data: dict, source_name: str, title: str) -> str:
    total = data["total"]
    sev = data["severity"]

    if data.get("layer2_included"):
        banner = ""
        flagged_label_suffix = " (all 4 layers)"
    else:
        banner = ('<div class="banner"><span>&#9888;</span><span>'
                  'Layer 2 (pincode master-data check) is not included in this report - it needs a live network '
                  'call per unique pincode. Run the Streamlit app with Layer 2 enabled for full 4-layer numbers.'
                  '</span></div>')
        flagged_label_suffix = " by rule engine (Layers 1+3)"

    gt_cards = ""
    truth_chart_block = ""
    truth_chart_js = ""
    accuracy_card = ""
    confusion_block = ""
    model_block = ""

    if data["has_ground_truth"]:
        gt = data["ground_truth"]
        gt_pct = round(100 * gt["improper"] / max(total, 1), 1)
        gt_cards = (f'<div class="card"><p class="label">Ground-truth improper</p>'
                    f'<p class="value">{gt["improper"]:,} <span style="font-size:14px;color:var(--muted)">'
                    f'({gt_pct}%)</span></p></div>')
        truth_chart_block = (
            '<div><div class="legend">'
            '<span><span class="dot" style="background:#e34948"></span>Improper</span>'
            '<span><span class="dot" style="background:#2a78d6"></span>Proper</span>'
            '</div><div class="chart-wrap" style="height:220px;"><canvas id="truthChart"></canvas></div></div>'
        )
        truth_chart_js = f"""
new Chart(document.getElementById('truthChart'), {{
  type: 'doughnut',
  data: {{ labels: ['Improper','Proper'], datasets: [{{ data: [{gt["improper"]}, {gt["proper"]}],
    backgroundColor: ['#e34948','#2a78d6'], borderWidth: 2, borderColor: ring }}] }},
  options: {{ ...common, plugins: {{ legend: {{ display: false }} }}, cutout: '65%' }}
}});"""

        c = data["confusion"]
        accuracy_card = (f'<div class="card"><p class="label">Rule-engine accuracy</p>'
                          f'<p class="value">{c["accuracy"]}%</p></div>')
        confusion_block = f"""
<h2>Rule engine vs ground truth</h2>
<table>
  <tr><th></th><th>Predicted improper</th><th>Predicted proper</th></tr>
  <tr><td>Actually improper</td><td>{c["tp"]:,}</td><td>{c["fn"]:,}</td></tr>
  <tr><td>Actually proper</td><td>{c["fp"]:,}</td><td>{c["tn"]:,}</td></tr>
</table>"""

    mcmp = data.get("model_comparison")
    if mcmp and "skipped_reason" not in mcmp:
        algs = list(mcmp.keys())
        colors = ["#2a78d6", "#eda100", "#008300"]
        datasets_js = ",\n      ".join(
            f'{{ label: "{ALGO_LABELS.get(a,a)}", data: {json.dumps([mcmp[a]["accuracy"], mcmp[a]["precision"], mcmp[a]["recall"], mcmp[a]["f1"]])}, backgroundColor: "{colors[i % 3]}", borderRadius: 4 }}'
            for i, a in enumerate(algs)
        )
        legend_html = "".join(
            f'<span><span class="dot" style="background:{colors[i % 3]}"></span>{ALGO_LABELS.get(a,a)}</span>'
            for i, a in enumerate(algs)
        )
        model_block = f"""
<h2>Layer 4 model comparison (real, cross-validated on your data)</h2>
<div class="legend">{legend_html}</div>
<div class="chart-wrap" style="height:260px;"><canvas id="modelChart"></canvas></div>
"""
        model_chart_js = f"""
new Chart(document.getElementById('modelChart'), {{
  type: 'bar',
  data: {{ labels: ['Accuracy','Precision','Recall','F1'], datasets: [
      {datasets_js}
  ] }},
  options: {{ ...common, plugins: {{ legend: {{ display: false }} }},
    scales: {{ y: {{ min: 0, max: 100, grid: {{ color: grid }}, ticks: {{ color: muted }} }}, x: {{ grid: {{ display: false }}, ticks: {{ color: muted }} }} }} }}
}});"""
    elif mcmp and "skipped_reason" in mcmp:
        model_block = f'<h2>Layer 4 model comparison</h2><p class="note">Skipped: {mcmp["skipped_reason"]}</p>'
        model_chart_js = ""
    else:
        model_chart_js = ""

    top_issues = data["top_issues"]
    issue_labels = json.dumps([i[0].replace("_", " ").title() for i in top_issues])
    issue_values = json.dumps([i[1] for i in top_issues])

    severity_legend = (
        f'<span><span class="dot" style="background:#e34948"></span>Critical {sev["critical"]:,}</span>'
        f'<span><span class="dot" style="background:#eda100"></span>Warning {sev["warning"]:,}</span>'
        f'<span><span class="dot" style="background:#2a78d6"></span>Clean {sev["clean"]:,}</span>'
    )

    return HTML_TEMPLATE.format(
        title=title,
        generated_at=data["generated_at"],
        source_name=source_name,
        total=total,
        total_fmt=f"{total:,}",
        banner=banner,
        flagged_label_suffix=flagged_label_suffix,
        gt_cards=gt_cards,
        rule_flagged_fmt=f'{data["rule_flagged"]:,}',
        rule_flagged_pct=round(100 * data["rule_flagged"] / max(total, 1), 1),
        accuracy_card=accuracy_card,
        severity_legend=severity_legend,
        truth_chart_block=truth_chart_block,
        issue_chart_height=max(200, len(top_issues) * 32 + 60),
        confusion_block=confusion_block,
        model_block=model_block,
        sev_data=json.dumps([sev["critical"], sev["warning"], sev["clean"]]),
        issue_labels=issue_labels,
        issue_values=issue_values,
        truth_chart_js=truth_chart_js,
        model_chart_js=model_chart_js,
    )
