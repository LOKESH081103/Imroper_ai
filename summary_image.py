"""
summary_image.py - render the agreement-level summary as a single
presentation-ready PNG image, so it can be dropped straight into an email,
a Slack message, or a slide without opening the app or Excel.

Pure presentation layer: takes the dict returned by
agreement_summary.summarize_agreements() and draws it. No analysis logic
lives here.
"""

import io

import matplotlib
matplotlib.use("Agg")  # headless backend - no display needed, safe inside Streamlit
import matplotlib.pyplot as plt

SEVERITY_COLORS = {"Clean": "#2ecc71", "Warning": "#f5b041", "Critical": "#e74c3c"}


def build_agreement_summary_png(agr_summary: dict, title: str = "Agreement Address Quality Summary",
                                 source_name: str = "") -> bytes:
    """
    Returns PNG bytes (300 DPI) with:
      - a headline strip: total agreements / total addresses
      - a donut chart of agreements by severity (Clean/Warning/Critical)
      - a donut chart of addresses, split by their agreement's severity
      - a KPI table underneath both, spelling out the exact counts so
        nothing has to be read off the chart by eye
    """
    total_agreements = agr_summary["total_agreements"]
    total_addresses = agr_summary["total_addresses"]

    labels = ["Clean", "Warning", "Critical"]
    agr_counts = [agr_summary["clean_agreements"], agr_summary["warning_agreements"],
                  agr_summary["critical_agreements"]]
    addr_counts = [agr_summary["clean_addresses"], agr_summary["warning_addresses"],
                   agr_summary["critical_addresses"]]
    colors = [SEVERITY_COLORS[l] for l in labels]

    fig = plt.figure(figsize=(11, 7.5), dpi=300)
    fig.patch.set_facecolor("white")

    # ---- header ----
    fig.text(0.5, 0.96, title, ha="center", va="top", fontsize=18, fontweight="bold", color="#1a1a1a")
    subtitle = f"{total_agreements:,} agreements  •  {total_addresses:,} addresses"
    if source_name:
        subtitle += f"  •  {source_name}"
    fig.text(0.5, 0.915, subtitle, ha="center", va="top", fontsize=11, color="#666666")

    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1.1], top=0.86, bottom=0.06, left=0.06, right=0.94,
                           wspace=0.35, hspace=0.35)

    def _donut(ax, counts, center_label):
        nonzero = [(l, c, col) for l, c, col in zip(labels, counts, colors) if c > 0]
        if not nonzero:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=12, color="#999")
            ax.axis("off")
            return
        n_labels, n_counts, n_colors = zip(*nonzero)
        total_here = sum(n_counts)
        wedges, _, autotexts = ax.pie(
            n_counts, colors=n_colors, startangle=90, counterclock=False,
            wedgeprops=dict(width=0.42, edgecolor="white", linewidth=2),
            autopct=lambda p: f"{p:.0f}%" if p >= 6 else "",
            pctdistance=0.79,
        )
        for t in autotexts:
            t.set_color("white")
            t.set_fontsize(10)
            t.set_fontweight("bold")
        ax.text(0, 0, f"{total_here:,}", ha="center", va="center", fontsize=17, fontweight="bold", color="#1a1a1a")
        ax.set_title(center_label, fontsize=12, fontweight="bold", color="#333333", pad=12)

    ax1 = fig.add_subplot(gs[0, 0])
    _donut(ax1, agr_counts, "Agreements by severity")

    ax2 = fig.add_subplot(gs[0, 1])
    _donut(ax2, addr_counts, "Addresses by agreement severity")

    # shared legend under both donuts
    handles = [plt.Rectangle((0, 0), 1, 1, color=SEVERITY_COLORS[l]) for l in labels]
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.52), ncol=3, frameon=False, fontsize=11)

    # ---- KPI table strip ----
    ax3 = fig.add_subplot(gs[1, :])
    ax3.axis("off")
    col_labels = ["", "Agreements", "Addresses in them"]
    rows = [
        ["Clean", f"{agr_summary['clean_agreements']:,}", f"{agr_summary['clean_addresses']:,}"],
        ["Warning", f"{agr_summary['warning_agreements']:,}", f"{agr_summary['warning_addresses']:,}"],
        ["Critical", f"{agr_summary['critical_agreements']:,}", f"{agr_summary['critical_addresses']:,}"],
        ["Total", f"{total_agreements:,}", f"{total_addresses:,}"],
    ]
    table = ax3.table(cellText=rows, colLabels=col_labels, cellLoc="center", loc="center",
                       colWidths=[0.34, 0.33, 0.33])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.9)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#e0e0e0")
        if r == 0:
            cell.set_facecolor("#f2f2f2")
            cell.set_text_props(fontweight="bold", color="#333333")
        elif r == len(rows):
            cell.set_facecolor("#fafafa")
            cell.set_text_props(fontweight="bold")
        else:
            cell.set_facecolor("white")
            if c == 0:
                cell.set_text_props(fontweight="bold", color=SEVERITY_COLORS[rows[r - 1][0]])

    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor="white", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()
