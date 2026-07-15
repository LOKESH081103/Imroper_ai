from PIL import Image, ImageDraw, ImageFont
import io
import streamlit as st
import pandas as pd
import base64
from datetime import date

st.set_page_config(page_title="Daily Collections Update", layout="wide", page_icon="📊")

# ── Global page styling ───────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Base ── */
[data-testid="stAppViewContainer"] {background: #f5f7fa;}
[data-testid="stHeader"] {background: transparent;}

/* ── App header banner ── */
.app-banner {
    background: linear-gradient(135deg, #0f2a5f 0%, #1a4d8f 50%, #2563c4 100%);
    border-radius: 18px;
    padding: 32px 40px 26px;
    color: #fff;
    margin-bottom: 32px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    box-shadow: 0 8px 32px rgba(15, 42, 95, 0.25);
    border: 1px solid rgba(255,255,255,.08);
}
.app-banner .banner-title {font-size: 28px; font-weight: 800; letter-spacing: .8px; margin-bottom: 6px;}
.app-banner .banner-sub   {font-size: 14px; opacity: .85; font-weight: 400; letter-spacing: .3px;}
.app-banner .banner-badge {
    background: rgba(255,255,255,.15);
    border: 1.5px solid rgba(255,255,255,.4);
    border-radius: 12px;
    padding: 10px 20px;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: .4px;
    backdrop-filter: blur(10px);
}

/* ── Upload card ── */
.upload-card {
    background: #ffffff;
    border-radius: 16px;
    padding: 26px 28px 22px;
    box-shadow: 0 4px 16px rgba(0,0,0,.06);
    margin-bottom: 8px;
    border: 1px solid #e8ecf1;
    transition: all .3s ease;
}
.upload-card:hover {
    box-shadow: 0 8px 24px rgba(0,0,0,.08);
    border-color: #d0dce6;
}
.upload-card .card-title {
    font-size: 12px;
    font-weight: 800;
    color: #1a4d8f;
    text-transform: uppercase;
    letter-spacing: .7px;
    margin-bottom: 16px;
    padding-bottom: 10px;
    border-bottom: 2.5px solid #e8f1f9;
}

/* ── Scope selector card ── */
.scope-card {
    background: #ffffff;
    border-radius: 16px;
    padding: 26px 28px;
    box-shadow: 0 4px 16px rgba(0,0,0,.06);
    margin-bottom: 28px;
    border: 1px solid #e8ecf1;
}
.scope-card .scope-title {
    font-size: 14px;
    font-weight: 800;
    color: #0f2a5f;
    margin-bottom: 18px;
    letter-spacing: .5px;
}

/* ── Radio & Select styling ── */
[data-testid="stRadio"] label {
    font-weight: 600 !important;
    font-size: 14px !important;
}
[data-testid="stSelectbox"] {
    margin-top: 2px;
}

/* ── Generate button ── */
[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #0f2a5f 0%, #1a4d8f 50%, #2563c4 100%) !important;
    color: #fff !important;
    font-weight: 800 !important;
    font-size: 16px !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 14px 0 !important;
    letter-spacing: .5px !important;
    box-shadow: 0 6px 20px rgba(25, 77, 143, 0.3) !important;
    transition: all .3s ease !important;
    border: 1px solid rgba(255,255,255,.1) !important;
}
[data-testid="stButton"] > button:hover {
    opacity: .92 !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(25, 77, 143, 0.4) !important;
}
[data-testid="stButton"] > button:active {
    transform: translateY(0) !important;
}

/* ── Report section headers ── */
.rpt-section-head {
    background: linear-gradient(90deg, #0f2a5f 0%, #1a4d8f 100%);
    border-radius: 10px;
    padding: 12px 20px;
    color: #fff;
    font-size: 15px;
    font-weight: 800;
    letter-spacing: .5px;
    margin: 24px 0 14px;
    display: flex;
    align-items: center;
    gap: 12px;
}
.rpt-section-head .snum {
    background: rgba(255,255,255,.3);
    border-radius: 50%;
    width: 28px; height: 28px;
    display: inline-flex;
    align-items: center; justify-content: center;
    font-size: 13px; font-weight: 900;
    flex-shrink: 0;
}

/* ── Report title ── */
.rpt-title {
    background: linear-gradient(135deg, #0f2a5f 0%, #1a4d8f 50%, #2563c4 100%);
    border-radius: 16px;
    padding: 28px 32px;
    color: #fff;
    margin-bottom: 24px;
    display: flex;
    justify-content: space-between;
    align-items: flex-end;
    box-shadow: 0 8px 24px rgba(15, 42, 95, 0.15);
    border: 1px solid rgba(255,255,255,.08);
}
.rpt-title .rt-main  {font-size: 21px; font-weight: 900; letter-spacing: .5px;}
.rpt-title .rt-sub   {font-size: 13px; opacity: .85; margin-top: 4px; letter-spacing: .3px;}
.rpt-title .rt-badge {
    background: rgba(255,255,255,.15);
    border: 1.5px solid rgba(255,255,255,.35);
    border-radius: 10px;
    padding: 8px 16px;
    font-size: 13px;
    font-weight: 700;
    backdrop-filter: blur(10px);
}

/* ── Dataframe tweaks ── */
[data-testid="stDataFrame"] {
    border-radius: 12px; 
    overflow: hidden;
    box-shadow: 0 4px 12px rgba(0,0,0,.05);
}
</style>
""", unsafe_allow_html=True)

# ── EXACT column indices (0-based, arrow columns included) ────────────────────
COL_ZONE      = 0
COL_FR_FLOW   = 2;  COL_FR_EFF  = 3;  COL_FR_LMTD  = 6
COL_129_FLOW  = 9;  COL_129_EFF = 11; COL_129_LMTD = 14
COL_NORM_EFF  = 21; COL_NORM_LMTD = 24
COL_3059_FLOW = 29; COL_3059_EFF = 31; COL_3059_LMTD = 34
COL_S3_FLOW   = 41; COL_S3_EFF  = 43; COL_S3_LMTD   = 46

# ── Zone configuration: parent key → list of sub-region keys for bottom ranking
ZONE_CONFIG = {
    'PAN_INDIA': {
        'label': 'PAN INDIA',
        'subzones': ['NORTH_1','NORTH_2','NORTH_3','EAST_1','EAST_2',
                     'SOUTH_1','SOUTH_2','WEST_1','WEST_2'],
        'display': {
            'NORTH_1':'North 1','NORTH_2':'North 2','NORTH_3':'North 3',
            'EAST_1':'East 1','EAST_2':'East 2',
            'SOUTH_1':'South 1','SOUTH_2':'South 2',
            'WEST_1':'West 1','WEST_2':'West 2',
        },
    },
    'NORTH_1': {
        'label': 'North 1',
        'subzones': ['DELHI_1','DELHI_2','DELHI_3','DELHI_4','DELHI_5','DELHI_6','DELHI_7',
                     'DELHI_WEST','GURUGRAM','UTTAR_PRADESH_1','UTTAR_PRADESH_2'],
        'display': {
            'DELHI_1':'Delhi 1','DELHI_2':'Delhi 2','DELHI_3':'Delhi 3',
            'DELHI_4':'Delhi 4','DELHI_5':'Delhi 5','DELHI_6':'Delhi 6','DELHI_7':'Delhi 7',
            'DELHI_WEST':'Delhi West','GURUGRAM':'Gurugram',
            'UTTAR_PRADESH_1':'UP 1','UTTAR_PRADESH_2':'UP 2',
        },
    },
    'NORTH_2': {
        'label': 'North 2',
        'subzones': ['CHANDIGARH','HARYANA','HIMACHAL','PUNJAB_1','PUNJAB_2','UTTARAKHAND'],
        'display': {
            'CHANDIGARH':'Chandigarh','HARYANA':'Haryana','HIMACHAL':'Himachal',
            'PUNJAB_1':'Punjab 1','PUNJAB_2':'Punjab 2','UTTARAKHAND':'Uttarakhand',
        },
    },
    'NORTH_3': {
        'label': 'North 3',
        'subzones': ['BHILWARA','JAIPUR_1','JAIPUR_2','JAIPUR_3','JAIPUR_4',
                     'BIKANER','JODHPUR','UDAIPUR'],
        'display': {
            'BHILWARA':'Bhilwara','JAIPUR_1':'Jaipur 1','JAIPUR_2':'Jaipur 2',
            'JAIPUR_3':'Jaipur 3','JAIPUR_4':'Jaipur 4',
            'BIKANER':'Bikaner','JODHPUR':'Jodhpur','UDAIPUR':'Udaipur',
        },
    },
    'EAST_1': {
        'label': 'East 1',
        'subzones': ['WEST_BENGAL','CHATTISGARH_1','CHATTISGARH_2','CHATTISGARH_3','ODISHA'],
        'display': {
            'WEST_BENGAL':'West Bengal',
            'CHATTISGARH_1':'Chattisgarh 1','CHATTISGARH_2':'Chattisgarh 2','CHATTISGARH_3':'Chattisgarh 3',
            'ODISHA':'Odisha',
        },
    },
    'EAST_2': {
        'label': 'East 2',
        'subzones': ['BIHAR','JHARKHAND','NORTH_EAST'],
        'display': {'BIHAR':'Bihar','JHARKHAND':'Jharkhand','NORTH_EAST':'North East'},
    },
    'SOUTH_1': {
        'label': 'South 1',
        'subzones': ['ANDHRA_PRADESH_1','ANDHRA_PRADESH_2','ANDHRA_PRADESH_3',
                     'KERALA',
                     'TAMIL_NADU_1','TAMIL_NADU_2','TAMIL_NADU_3','TAMIL_NADU_4',
                     'TAMIL_NADU_5','TAMIL_NADU_6','TAMIL_NADU_7',
                     'TAMIL_NADU_1,4 & 7','TAMIL_NADU_2 & 5 ','TAMIL_NADU_3 & 6'],
        'display': {
            'ANDHRA_PRADESH_1':'AP 1','ANDHRA_PRADESH_2':'AP 2','ANDHRA_PRADESH_3':'AP 3',
            'KERALA':'Kerala',
            'TAMIL_NADU_1':'Tamil Nadu 1','TAMIL_NADU_2':'Tamil Nadu 2','TAMIL_NADU_3':'Tamil Nadu 3',
            'TAMIL_NADU_4':'Tamil Nadu 4','TAMIL_NADU_5':'Tamil Nadu 5','TAMIL_NADU_6':'Tamil Nadu 6',
            'TAMIL_NADU_7':'Tamil Nadu 7',
            'TAMIL_NADU_1,4 & 7':'TN 1,4&7','TAMIL_NADU_2 & 5 ':'TN 2&3','TAMIL_NADU_3 & 6':'TN 5&6',
        },
    },
    'SOUTH_2': {
        'label': 'South 2',
        'subzones': ['KARNATAKA_1','KARNATAKA_2','KARNATAKA_3','KARNATAKA_4',
                     'KARNATAKA_5','KARNATAKA_6','KARNATAKA_7',
                     'TELANGANA_1','TELANGANA_2','TELANGANA_3',
                     'TELANGANA_4','TELANGANA_5','TELANGANA_6'],
        'display': {
            'KARNATAKA_1':'Karnataka 1','KARNATAKA_2':'Karnataka 2','KARNATAKA_3':'Karnataka 3',
            'KARNATAKA_4':'Karnataka 4','KARNATAKA_5':'Karnataka 5','KARNATAKA_6':'Karnataka 6',
            'KARNATAKA_7':'Karnataka 7',
            'TELANGANA_1':'Telangana 1','TELANGANA_2':'Telangana 2','TELANGANA_3':'Telangana 3',
            'TELANGANA_4':'Telangana 4','TELANGANA_5':'Telangana 5','TELANGANA_6':'Telangana 6',
        },
    },
    'WEST_1': {
        'label': 'West 1',
        'subzones': ['GUJARAT_1','GUJARAT_2','GUJARAT_3','GUJARAT_4','GUJARAT_5','GUJARAT_6',
                     'MAHARASTRA_1','MAHARASTRA_2','MAHARASTRA_3','MAHARASTRA_4','MAHARASTRA_5'],
        'display': {
            'GUJARAT_1':'Gujarat 1','GUJARAT_2':'Gujarat 2','GUJARAT_3':'Gujarat 3',
            'GUJARAT_4':'Gujarat 4','GUJARAT_5':'Gujarat 5','GUJARAT_6':'Gujarat 6',
            'MAHARASTRA_1':'Maharastra 1','MAHARASTRA_2':'Maharastra 2','MAHARASTRA_3':'Maharastra 3',
            'MAHARASTRA_4':'Maharastra 4','MAHARASTRA_5':'Maharastra 5',
        },
    },
    'WEST_2': {
        'label': 'West 2',
        'subzones': ['MADHYA_PRADESH_1','MADHYA_PRADESH_2','GOA',
                     'MUMBAI_1','MUMBAI_2','MUMBAI_3','MUMBAI_4','MUMBAI_5'],
        'display': {
            'MADHYA_PRADESH_1':'MP 1','MADHYA_PRADESH_2':'MP 2','GOA':'Goa',
            'MUMBAI_1':'Mumbai 1','MUMBAI_2':'Mumbai 2','MUMBAI_3':'Mumbai 3',
            'MUMBAI_4':'Mumbai 4','MUMBAI_5':'Mumbai 5',
        },
    },
    # ── Custom combined zones ─────────────────────────────────────────────────
    'MP_AND_GUJARAT': {
        'label': 'MP & Gujarat',
        'subzones': ['MADHYA_PRADESH_1','MADHYA_PRADESH_2',
                     'GUJARAT_1','GUJARAT_2','GUJARAT_3','GUJARAT_4','GUJARAT_5','GUJARAT_6'],
        'display': {
            'MADHYA_PRADESH_1':'MP 1','MADHYA_PRADESH_2':'MP 2',
            'GUJARAT_1':'Gujarat 1','GUJARAT_2':'Gujarat 2','GUJARAT_3':'Gujarat 3',
            'GUJARAT_4':'Gujarat 4','GUJARAT_5':'Gujarat 5','GUJARAT_6':'Gujarat 6',
        },
    },
    'ROMH_AND_MUMBAI': {
        'label': 'ROMH & Mumbai',
        'subzones': ['MAHARASTRA_1','MAHARASTRA_2','MAHARASTRA_3','MAHARASTRA_4','MAHARASTRA_5',
                     'MUMBAI_1','MUMBAI_2','MUMBAI_3','MUMBAI_4','MUMBAI_5'],
        'display': {
            'MAHARASTRA_1':'Maharastra 1','MAHARASTRA_2':'Maharastra 2','MAHARASTRA_3':'Maharastra 3',
            'MAHARASTRA_4':'Maharastra 4','MAHARASTRA_5':'Maharastra 5',
            'MUMBAI_1':'Mumbai 1','MUMBAI_2':'Mumbai 2','MUMBAI_3':'Mumbai 3',
            'MUMBAI_4':'Mumbai 4','MUMBAI_5':'Mumbai 5',
        },
    },
}

# All zone keys that need to be extracted from the dashboard
ALL_ZONE_KEYS = [
    'NORTH','NORTH_1','NORTH_2','NORTH_3',
    'EAST','EAST_1','EAST_2',
    'SOUTH','SOUTH_1','SOUTH_2',
    'WEST','WEST_1','WEST_2',
    'PAN_INDIA',
    # Sub-regions for NORTH_1
    'DELHI','DELHI_1','DELHI_2','DELHI_3','DELHI_4','DELHI_5','DELHI_6','DELHI_7',
    'DELHI_WEST','GURUGRAM','UTTAR_PRADESH','UTTAR_PRADESH_1','UTTAR_PRADESH_2',
    # Sub-regions for NORTH_2
    'CHANDIGARH','HARYANA','HIMACHAL','PUNJAB_1','PUNJAB_2','UTTARAKHAND',
    # Sub-regions for NORTH_3
    'RAJASTHAN_1','RAJASTHAN_2',
    'BHILWARA','JAIPUR_1','JAIPUR_2','JAIPUR_3','JAIPUR_4','BIKANER','JODHPUR','UDAIPUR',
    # Sub-regions for EAST_1
    'CHATTISGARH','CHATTISGARH_1','CHATTISGARH_2','CHATTISGARH_3','ODISHA','WEST_BENGAL',
    # Sub-regions for EAST_2
    'BIHAR','JHARKHAND','NORTH_EAST',
    # Sub-regions for SOUTH_1
    'ANDHRA_PRADESH','ANDHRA_PRADESH_1','ANDHRA_PRADESH_2','ANDHRA_PRADESH_3',
    'KERALA',
    'TAMIL_NADU','TAMIL_NADU_1','TAMIL_NADU_2','TAMIL_NADU_3',
    'TAMIL_NADU_4','TAMIL_NADU_5','TAMIL_NADU_6','TAMIL_NADU_7',
    'TAMIL_NADU_1,4 & 7','TAMIL_NADU_2 & 5 ','TAMIL_NADU_3 & 6',
    # Sub-regions for SOUTH_2
    'KARNATAKA','KARNATAKA_1','KARNATAKA_2','KARNATAKA_3','KARNATAKA_4',
    'KARNATAKA_5','KARNATAKA_6','KARNATAKA_7',
    'TELANGANA','TELANGANA_1','TELANGANA_2','TELANGANA_3',
    'TELANGANA_4','TELANGANA_5','TELANGANA_6',
    # Sub-regions for WEST_1
    'GUJARAT','GUJARAT_1','GUJARAT_2','GUJARAT_3','GUJARAT_4','GUJARAT_5','GUJARAT_6',
    'ROMH','MAHARASTRA_1','MAHARASTRA_2','MAHARASTRA_3','MAHARASTRA_4','MAHARASTRA_5',
    # Sub-regions for WEST_2
    'MADHYA_PRADESH','MADHYA_PRADESH_1','MADHYA_PRADESH_2',
    'GOA','MUMBAI_1','MUMBAI_2','MUMBAI_3','MUMBAI_4','MUMBAI_5','MUMBAI_&_GOA',
    # Custom combined zones
    'MP_AND_GUJARAT','ROMH_AND_MUMBAI',
]

# ── Helpers ───────────────────────────────────────────────────────────────────
def pct(val):
    if val is None: return None
    if isinstance(val, float) and pd.isna(val): return None
    s = str(val).strip()
    if s in ('', 'nan', 'None', '-'): return None
    has_pct_sign = s.endswith('%')
    s_clean = s.replace('%', '').replace(',', '')
    try: f = float(s_clean)
    except ValueError: return None
    if has_pct_sign: return f
    else:
        if abs(f) < 1.5: return round(f * 100, 4)
        return f

def num(val):
    if val is None: return None
    if isinstance(val, float) and pd.isna(val): return None
    s = str(val).strip().replace(',', '').replace('%', '')
    if s in ('', 'nan', 'None', '-'): return None
    try: return float(s)
    except ValueError: return None

def fmt_pct(v): return f"{v:.2f}%" if v is not None else '---'
def fmt_num(v): return f"{v:,.2f}" if v is not None else '---'

def safe_get(row_list, idx):
    if row_list is None or idx >= len(row_list): return None
    return row_list[idx]

def bottom2_eff(all_metrics, eff_key, subzones, display):
    vals = [(display.get(sz, sz), all_metrics.get(sz, {}).get(eff_key))
            for sz in subzones if all_metrics.get(sz, {}).get(eff_key) is not None]
    sorted_vals = sorted(vals, key=lambda x: x[1])
    grouped = {}
    for name, val in sorted_vals:
        grouped.setdefault(val, []).append(name)
    distinct = sorted(grouped.items(), key=lambda x: x[0])[:2]
    return [(", ".join(names), val) for val, names in distinct]

def bottom2_flow(all_metrics, flow_key, subzones, display):
    vals = [(display.get(sz, sz), all_metrics.get(sz, {}).get(flow_key))
            for sz in subzones if all_metrics.get(sz, {}).get(flow_key) is not None]
    sorted_vals = sorted(vals, key=lambda x: x[1], reverse=True)
    grouped = {}
    for name, val in sorted_vals:
        grouped.setdefault(val, []).append(name)
    distinct = sorted(grouped.items(), key=lambda x: x[0], reverse=True)[:2]
    return [(", ".join(names), val) for val, names in distinct]

# ── File loading ──────────────────────────────────────────────────────────────
def load_all_sheets(file):
    if file is None: return {}
    name = file.name.lower()
    file.seek(0)
    try:
        if name.endswith('.csv'):
            df = pd.read_csv(file, header=None, dtype=str)
            return {"Sheet1": df}
        elif name.endswith('.xlsb'):
            # Try pyxlsb first; fall back to LibreOffice conversion
            try:
                return pd.read_excel(file, sheet_name=None, engine="pyxlsb", header=None)
            except Exception:
                import tempfile, subprocess, os
                file.seek(0)
                with tempfile.TemporaryDirectory() as tmpdir:
                    xlsb_path = os.path.join(tmpdir, file.name)
                    with open(xlsb_path, 'wb') as f:
                        f.write(file.read())
                    result = subprocess.run(
                        ['libreoffice', '--headless', '--convert-to', 'xlsx',
                         xlsb_path, '--outdir', tmpdir],
                        capture_output=True, timeout=60
                    )
                    xlsx_path = xlsb_path.replace('.xlsb', '.xlsx')
                    if os.path.exists(xlsx_path):
                        return pd.read_excel(xlsx_path, sheet_name=None, header=None)
                    raise RuntimeError(f"LibreOffice conversion failed: {result.stderr.decode()}")
        else:
            return pd.read_excel(file, sheet_name=None, header=None)
    except Exception as e:
        st.error(f"Error loading {file.name}: {e}")
        return {}

# ── Extraction Helpers ────────────────────────────────────────────────────────
def _detect_zone_col(df):
    """Scan first 30 rows across columns to find which column holds zone names."""
    targets = set(k.upper() for k in ALL_ZONE_KEYS)
    best_col, best_count = 1, 0
    for col_idx in range(min(10, len(df.columns))):
        count = 0
        for _, row in df.head(30).iterrows():
            try:
                cell = str(row.iloc[col_idx]).strip().upper().replace(' ', '_')
                if cell in targets:
                    count += 1
            except Exception:
                pass
        if count > best_count:
            best_count, best_col = count, col_idx
    return best_col

import re as _re

def _norm_header_text(v):
    """Lowercase + collapse whitespace, for fuzzy header matching."""
    if v is None:
        return ""
    s = str(v).strip().lower()
    s = _re.sub(r"[\s\n]+", " ", s)
    return s

def _find_group_col(df, group_row, group_label, search_cols=90):
    """Find the starting column of a top-level group header (e.g. '1-29 NORM')
    by scanning a header row for matching text."""
    target = _norm_header_text(group_label)
    limit = min(search_cols, len(df.columns))
    for c in range(limit):
        if target in _norm_header_text(df.iloc[group_row, c]):
            return c
    return None

def _find_sub_col(df, header_row, start_col, sub_label, width=15, exact=False, exclude=None):
    """Within [start_col, start_col+width) find a sub-header column by text match.
    exact=True requires the normalized header to equal sub_label exactly (used for
    'Flow' so it doesn't accidentally match 'Flow Projection').
    exclude, if given, skips any header containing that substring (used to keep
    'May'26 LMTD' lookups from matching 'Performance : AOD Vs LMTD%')."""
    if start_col is None:
        return None
    target = _norm_header_text(sub_label)
    end = min(start_col + width, len(df.columns))
    for c in range(start_col, end):
        txt = _norm_header_text(df.iloc[header_row, c])
        if exclude and exclude in txt:
            continue
        if exact:
            if txt == target:
                return c
        else:
            if target in txt:
                return c
    return None

def _find_all_sub_cols(df, header_row, start_col, sub_label, width=15, exclude=None):
    """Like _find_sub_col but returns ALL matching columns in range, not just
    the first. Used when a header phrase (e.g. 'LMTD') legitimately appears
    more than once in a block (a % version and an absolute-value version)."""
    if start_col is None:
        return []
    target = _norm_header_text(sub_label)
    end = min(start_col + width, len(df.columns))
    found = []
    for c in range(start_col, end):
        txt = _norm_header_text(df.iloc[header_row, c])
        if exclude and exclude in txt:
            continue
        if target in txt:
            found.append(c)
    return found

def _row_boundaries(df, row_idx, total_cols):
    """Column indices where `row_idx` has a non-null value — marks the start
    of each top-level column group on that row."""
    return [c for c in range(total_cols) if pd.notna(df.iloc[row_idx, c])]

def _group_span(boundaries, start_col, total_cols):
    """Width from start_col up to (but not including) the next group boundary,
    so block-local header search never overruns into the next block."""
    later = [b for b in boundaries if b > start_col]
    end = later[0] if later else total_cols
    return max(end - start_col, 1)

def extract_dashboard_metrics(df, zone_name="PAN_INDIA"):
    """
    Extract Norm metrics from the dashboard.

    Columns are located DYNAMICALLY by matching header text ("1-29 NORM" group ->
    "Overall Norm" / "Norm Projection" / the May'26 LMTD reference column, and the
    "STAGE_3" group -> "Allocation Value") instead of hardcoded absolute column
    numbers. This keeps extraction correct even if upstream column layout shifts
    (e.g. an extra leading zone-type column gets added/removed).

    The "1-29 NORM" block has TWO columns whose header text both contain
    "LMTD" — one holding a percentage (e.g. 0.056 = 5.6%) and one holding an
    absolute Crores value (e.g. 21.7). Since norm_129 / norm_129_proj are
    absolute Crores values, norm_129_lmtd must use the absolute-scale twin,
    not the percentage one — otherwise it shows a tiny, out-of-scale number
    next to Today/Projection. We resolve this per-row using the same
    convention as pct(): a raw value with |v| < 1.5 is treated as a
    percentage-style decimal, so we skip it in favour of the larger one.

    Falls back to the old fixed offsets only if headers can't be located at all.
    """
    metrics = {'norm_129': None, 'norm_129_lmtd': None, 'norm_129_proj': None}
    if df is None or df.empty:
        return metrics
    try:
        zone_col  = _detect_zone_col(df)
        target    = zone_name.upper().replace(" ", "_")
        found_idx = None
        for idx, row in df.iterrows():
            cell = str(row.iloc[zone_col]).strip().upper().replace(" ", "_")
            if cell == target:
                found_idx = idx
                break
        # Fallback: PAN INDIA is usually around row 16
        if found_idx is None:
            found_idx = 16
        if len(df) <= found_idx:
            return metrics

        row = df.iloc[found_idx].tolist()

        # ── Locate columns by header text (scan a few candidate header rows,
        #    since group titles / sub-titles can sit on slightly different
        #    rows depending on the export) ──────────────────────────────────
        norm_group_col = None
        s3_group_col   = None
        norm_header_row = None
        for hr in range(0, min(6, len(df))):
            if norm_group_col is None:
                gc = _find_group_col(df, hr, "1-29 NORM")
                if gc is not None:
                    norm_group_col, norm_header_row = gc, hr
            if s3_group_col is None:
                gc3 = _find_group_col(df, hr, "STAGE_3")
                if gc3 is not None:
                    s3_group_col = gc3

        # Bound the search to this block only (up to the next group's start
        # column) so it can't wander into the next block (e.g. picking up a
        # "May'26 LMTD%" that actually belongs to "30-59", not "1-29 NORM").
        norm_width = 18
        if norm_group_col is not None and norm_header_row is not None:
            boundaries  = _row_boundaries(df, norm_header_row, len(df.columns))
            norm_width  = _group_span(boundaries, norm_group_col, len(df.columns))

        c_norm = c_proj = c_s3rb = None
        lmtd_candidates = []
        if norm_group_col is not None:
            # sub-headers sit one row below the group title, typically
            for sr in range(norm_header_row + 1, min(norm_header_row + 3, len(df))):
                if c_norm is None:
                    c_norm = _find_sub_col(df, sr, norm_group_col, "Overall Norm", width=norm_width)
                if c_proj is None:
                    c_proj = _find_sub_col(df, sr, norm_group_col, "Norm Projection", width=norm_width)
                if not lmtd_candidates:
                    # Collect every "...LMTD..." header in this block, excluding
                    # the "Performance : AOD Vs LMTD%" delta column.
                    lmtd_candidates = _find_all_sub_cols(
                        df, sr, norm_group_col, "lmtd", width=norm_width, exclude="performance"
                    )
        if s3_group_col is not None:
            for sr in range(0, min(6, len(df))):
                c_s3rb = _find_sub_col(df, sr, s3_group_col, "Allocation Value", width=18)
                if c_s3rb is not None:
                    break

        # Pick the absolute-scale LMTD candidate over the percentage-scale
        # one. |v| >= 1.5 is a reliable signal when present; for small zones
        # where even the absolute value dips under 1.5, the absolute-scale
        # column is still consistently the LAST "...LMTD..." match in the
        # block (the percentage version sits right after "Performance...",
        # the absolute version right before "EOM-Norm Value") — so fall back
        # to the rightmost candidate rather than the first.
        c_lmtd = None
        if lmtd_candidates:
            vals = [(c, num(safe_get(row, c))) for c in lmtd_candidates]
            vals = [(c, v) for c, v in vals if v is not None]
            abs_scale = [(c, v) for c, v in vals if abs(v) >= 1.5]
            if abs_scale:
                c_lmtd = abs_scale[0][0]
            elif vals:
                c_lmtd = vals[-1][0]
            else:
                c_lmtd = lmtd_candidates[-1]

        # ── Read values; fall back to legacy hardcoded offsets if a given
        #    column couldn't be resolved by header text ────────────────────
        metrics['norm_129']      = num(safe_get(row, c_norm)) if c_norm is not None else num(safe_get(row, 15))
        metrics['norm_129_proj'] = num(safe_get(row, c_proj)) if c_proj is not None else num(safe_get(row, 17))
        metrics['norm_129_lmtd'] = num(safe_get(row, c_lmtd)) if c_lmtd is not None else num(safe_get(row, 23))
        # NOTE: 's3_rb' (Stage 3 Roll Back) is intentionally NOT read from the
        # main dashboard anymore. It must come exclusively from the dedicated
        # Stage 3 Roll Back file (see extract_s3_rb_metrics). If that file is
        # not uploaded, Section 2's "Stage 3 Roll Back" row should show '---'
        # rather than silently falling back to a dashboard-derived value.
    except Exception as e:
        st.warning(f"Error extracting dashboard metrics for {zone_name}: {e}")
    return metrics

def extract_s2_metrics(dfs_dict, zone_name=None):
    """
    Extract Stage 2 metrics from the 'New Summary' sheet.
    Zone name is in column 2 (0-based). Columns:
      col2 = zone, col4 = RB Projection, col6 = Actual RB (today), col8 = LMTD RB
    """
    metrics = {'s2_rb': None, 's2_lmtd': None, 's2_proj': None}
    if not dfs_dict:
        return metrics
    try:
        # Prefer 'New Summary' sheet; fall back to first sheet
        df = dfs_dict.get('New Summary')
        if df is None:
            df = list(dfs_dict.values())[0] if dfs_dict else None
        if df is None or df.empty:
            return metrics

        # Normalise the zone target
        target_key = (zone_name or 'PAN_INDIA').upper().strip()

        found_idx = None
        for idx, row in df.iterrows():
            # Zone name lives in column 2
            cell = str(row.iloc[2]).strip().upper().replace(' ', '_')
            if cell == target_key or cell == target_key.replace('_', ' '):
                found_idx = idx
                break

        # If zone not found, fall back to PAN INDIA row (index 4)
        if found_idx is None:
            found_idx = 4

        if len(df) > found_idx:
            row = df.iloc[found_idx].tolist()
            # col4 = RB Projection, col6 = Actual RB, col8 = LMTD RB
            metrics['s2_proj'] = num(safe_get(row, 4))
            metrics['s2_rb']   = num(safe_get(row, 6))
            metrics['s2_lmtd'] = num(safe_get(row, 8))
    except Exception as e:
        st.warning(f"Error extracting Stage 2 metrics: {e}")
    return metrics

def extract_s3_metrics(dfs_dict, zone_name=None):
    """
    Extract Stage 3 metrics.
    S3 Concern summary sheet: zone in col0, col4=flow, col8=LMTD flow.
    Top Sheet: PAN INDIA only (row 12, col2 = current-month flow projection).
    """
    metrics = {'s3_concern_flow': None, 's3_concern_lmtd': None, 's3_concern_proj': None}
    if not dfs_dict:
        return metrics
    try:
        # ── S3 Concern summary ──────────────────────────────────────────
        df_sum = dfs_dict.get('S3 Concern summary')
        if df_sum is None:
            for k in dfs_dict.keys():
                if 'concern summary' in k.lower():
                    df_sum = dfs_dict[k]; break

        if df_sum is not None and not df_sum.empty:
            target_key = (zone_name or 'PAN_INDIA').upper().strip()
            found_idx = None

            for idx, row in df_sum.iterrows():
                # Zone name is in column 0
                cell = str(row.iloc[0]).strip().upper().replace(' ', '_')
                if cell == target_key or cell == target_key.replace('_', ' '):
                    found_idx = idx
                    break

            # Fallback to PAN INDIA (row 4)
            if found_idx is None:
                found_idx = 4

            if len(df_sum) > found_idx:
                row_sum = df_sum.iloc[found_idx].tolist()
                # col4 = AS ON DATE FLOW, col8 = LMTD FLOW, col2 = Collected (Projection)
                metrics['s3_concern_flow'] = num(safe_get(row_sum, 4))
                metrics['s3_concern_lmtd'] = num(safe_get(row_sum, 8))
                metrics['s3_concern_proj'] = num(safe_get(row_sum, 2))

    except Exception as e:
        st.warning(f"Error extracting Stage 3 metrics: {e}")
    return metrics


def extract_s3_rb_metrics(rb_file, zone_name=None):
    """
    Extract Stage 3 Roll Back metrics from the dedicated S3 RB file.
    File structure:
      col0 = Zone/Region name
      col4 (E) = RB Achieved (Today or Yesterday value)
      col7 (H) = LMTD RB value
    Zone name aliases: MUMBAI -> MUMBAI_&_GOA, MAHARASTRA -> ROMH
    """
    metrics = {'s3_rb': None, 's3_rb_lmtd': None, 's3_rb_proj': None}
    if rb_file is None:
        return metrics
    try:
        import io
        rb_file.seek(0)
        df = pd.read_excel(rb_file, header=None)

        # Zone name aliases: S3 RB file uses different names for some zones
        ALIASES = {
            'MUMBAI':      'MUMBAI_&_GOA',
            'MAHARASTRA':  'ROMH',
        }

        target_raw = (zone_name or 'PAN_INDIA').upper().strip()
        # Reverse alias: if we're looking for MUMBAI_&_GOA, accept MUMBAI too
        reverse_aliases = {v: k for k, v in ALIASES.items()}

        found_row = None
        for _, row in df.iterrows():
            c0 = str(row.iloc[0]).strip()
            if not c0 or c0 == 'nan':
                continue
            normalized = c0.upper().replace(' ', '_')
            # Apply alias mapping (file -> app key)
            mapped = ALIASES.get(normalized, normalized)
            if mapped == target_raw or normalized == target_raw:
                found_row = row
                break

        # For combined zones, sum their sub-zones
        if found_row is None:
            COMBINED_MAP = {
                'MP_AND_GUJARAT':  ['MADHYA_PRADESH', 'GUJARAT'],
                'ROMH_AND_MUMBAI': ['MAHARASTRA', 'MUMBAI'],
            }
            sub_keys = COMBINED_MAP.get(target_raw)
            if sub_keys:
                rb_total = 0.0; lmtd_total = 0.0; found_any = False
                for _, row in df.iterrows():
                    c0 = str(row.iloc[0]).strip().upper().replace(' ', '_')
                    if c0 in sub_keys:
                        try: rb_total   += float(row.iloc[4]); found_any = True
                        except: pass
                        try: lmtd_total += float(row.iloc[7]); found_any = True
                        except: pass
                proj_total = 0.0
                for _, row in df.iterrows():
                    c0 = str(row.iloc[0]).strip().upper().replace(' ', '_')
                    if c0 in sub_keys:
                        try: proj_total += float(row.iloc[2])
                        except: pass
                if found_any:
                    metrics['s3_rb']      = rb_total
                    metrics['s3_rb_lmtd'] = lmtd_total
                    metrics['s3_rb_proj'] = proj_total
                return metrics

        if found_row is not None:
            try: metrics['s3_rb']      = float(found_row.iloc[4])
            except: pass
            try: metrics['s3_rb_lmtd'] = float(found_row.iloc[7])
            except: pass
            try: metrics['s3_rb_proj'] = float(found_row.iloc[2])
            except: pass

    except Exception as e:
        import streamlit as st
        st.warning(f"Error reading S3 RB file: {e}")
    return metrics

def find_data_rows(df):
    targets  = set(k.upper() for k in ALL_ZONE_KEYS)
    zone_col = _detect_zone_col(df)
    result   = []
    for _, row in df.iterrows():
        try:
            cell = str(row.iloc[zone_col]).strip().upper().replace(' ', '_')
        except Exception:
            continue
        if cell in targets:
            result.append((zone_col, row.tolist()))
    return result

def get_row(rows, name):
    norm = name.upper().replace(' ', '_')
    for zone_col, r in rows:
        try:
            cell = str(r[zone_col]).strip().upper().replace(' ', '_')
        except Exception:
            continue
        if cell == norm:
            return r
    return None

def _resolve_columns(df):
    """
    Dynamically resolve every metric column extract_all() needs, by matching
    header text within each top-level block (FRESH / 1-29 / 1-29 NORM / 30-59
    / S3 CONCERN) instead of relying on fixed absolute column numbers.

    This is the fix for values silently coming out wrong (or blank) only for
    SOME uploads: the fixed COL_* constants assume the zone name sits in a
    specific column with a specific number of columns before each block. If
    one file has an extra leading column (e.g. a zone-type marker) that the
    other doesn't, every fixed offset shifts by one and starts reading the
    wrong header's data for that file — most visibly in the S3 CONCERN block,
    where the shifted column can land on a blank/arrow cell.

    Falls back to the historical fixed offsets for any column that can't be
    located by header text (e.g. a sheet with no group headers at all).
    """
    cols = {
        'fresh_flow': COL_FR_FLOW, 'fresh_eff': COL_FR_EFF, 'fresh_lmtd': COL_FR_LMTD,
        '129_flow': COL_129_FLOW, '129_eff': COL_129_EFF, '129_lmtd': COL_129_LMTD,
        'norm_eff': COL_NORM_EFF, 'norm_lmtd': COL_NORM_LMTD,
        '3059_flow': COL_3059_FLOW, '3059_eff': COL_3059_EFF, '3059_lmtd': COL_3059_LMTD,
        's3_flow': COL_S3_FLOW, 's3_eff': COL_S3_EFF, 's3_lmtd': COL_S3_LMTD,
    }
    if df is None or df.empty:
        return cols

    GROUP_ROW  = 2  # row holding 'FRESH' / '1-29' / '1-29 NORM' / '30-59' / 'S3 CONCERN'
    total_cols = len(df.columns)
    if len(df) <= GROUP_ROW:
        return cols
    boundaries = _row_boundaries(df, GROUP_ROW, total_cols)

    block_defs = [
        ('FRESH',      ['fresh_flow', 'fresh_eff', 'fresh_lmtd'], "aod eff"),
        ('1-29',       ['129_flow',   '129_eff',   '129_lmtd'],   "aod eff"),
        ('1-29 NORM',  ['norm_eff',   'norm_lmtd'],                "aod norm"),
        ('30-59',      ['3059_flow',  '3059_eff',  '3059_lmtd'],  "aod eff"),
        ('S3 CONCERN', ['s3_flow',    's3_eff',    's3_lmtd'],    "aod eff"),
    ]

    for group_label, metric_names, eff_term in block_defs:
        group_col = _find_group_col(df, GROUP_ROW, group_label, search_cols=total_cols)
        if group_col is None:
            continue  # keep fallback offsets for this whole block
        width = _group_span(boundaries, group_col, total_cols)

        resolved = {}
        for sr in range(GROUP_ROW + 1, min(GROUP_ROW + 3, len(df))):
            for mname in metric_names:
                if mname in resolved:
                    continue
                if mname.endswith('_flow'):
                    c = _find_sub_col(df, sr, group_col, "flow", width=width, exact=True)
                elif mname.endswith('_eff'):
                    c = _find_sub_col(df, sr, group_col, eff_term, width=width)
                elif mname.endswith('_lmtd'):
                    # the raw May'26 LMTD reference column, not the
                    # "Performance : AOD Vs LMTD%" delta column
                    c = _find_sub_col(df, sr, group_col, "lmtd", width=width, exclude="performance")
                else:
                    c = None
                if c is not None:
                    resolved[mname] = c
        cols.update(resolved)

    return cols

def extract_all(df, rows):
    metrics = {}
    cmap = _resolve_columns(df)
    for z in ALL_ZONE_KEYS:
        r = get_row(rows, z)
        if r is None: continue
        metrics[z] = {
            'fresh_flow' : num(safe_get(r, cmap['fresh_flow'])),
            'fresh_eff'  : pct(safe_get(r, cmap['fresh_eff'])),
            'fresh_lmtd' : pct(safe_get(r, cmap['fresh_lmtd'])),
            '129_flow'   : num(safe_get(r, cmap['129_flow'])),
            '129_eff'    : pct(safe_get(r, cmap['129_eff'])),
            '129_lmtd'   : pct(safe_get(r, cmap['129_lmtd'])),
            'norm_eff'   : pct(safe_get(r, cmap['norm_eff'])),
            'norm_lmtd'  : pct(safe_get(r, cmap['norm_lmtd'])),
            '3059_flow'  : num(safe_get(r, cmap['3059_flow'])),
            '3059_eff'   : pct(safe_get(r, cmap['3059_eff'])),
            '3059_lmtd'  : pct(safe_get(r, cmap['3059_lmtd'])),
            's3_flow'    : num(safe_get(r, cmap['s3_flow'])),
            's3_eff'     : pct(safe_get(r, cmap['s3_eff'])),
            's3_lmtd'    : pct(safe_get(r, cmap['s3_lmtd'])),
        }
    return metrics

# ── Streamlit render ──────────────────────────────────────────────────────────
def render_report(today_m, yest_m, extra_t, extra_y, date_str, view_key):
    cfg      = ZONE_CONFIG[view_key]
    region_t = today_m.get(view_key, {})
    region_y = yest_m.get(view_key, {})
    subzones = cfg['subzones']
    display  = cfg['display']
    label    = cfg['label']

    st.markdown(
        f"""<div class="rpt-title">
  <div>
    <div class="rt-main">📋 DAILY COLLECTIONS UPDATE</div>
    <div class="rt-sub">{label} — Zone / Region Efficiency Summary</div>
  </div>
  <div class="rt-badge">📅 {date_str}</div>
</div>""",
        unsafe_allow_html=True,
    )

    def section(num, title):
        st.markdown(
            f'<div class="rpt-section-head"><span class="snum">{num}</span>{title}</div>',
            unsafe_allow_html=True,
        )

    # ── Section 1 & 2 (hidden for combined zones) ───────────────────────────
    if view_key not in {'MP_AND_GUJARAT', 'ROMH_AND_MUMBAI'}:
        # ── Section 1 ────────────────────────────────────────────────────────
        section(1, "Efficiency %")
        buckets = [
            ("Fresh",              'fresh_eff', 'fresh_lmtd', 'fresh_flow'),
            ("1–29",               '129_eff',   '129_lmtd',   '129_flow'),
            ("1–29 Norm%",         'norm_eff',  'norm_lmtd',  None),
            ("30-59",              '3059_eff',  '3059_lmtd',  '3059_flow'),
            ("60-89 (S3 CONCERN)", 's3_eff',    's3_lmtd',    's3_flow'),
        ]
        rows1 = []
        for lbl, eff_k, lmtd_k, flow_k in buckets:
            t_eff_raw  = region_t.get(eff_k)
            t_lmtd_raw = region_t.get(lmtd_k)
            t_eff_str  = fmt_pct(t_eff_raw)
            t_lmtd_str = fmt_pct(t_lmtd_raw)
            if t_eff_raw is not None and t_lmtd_raw is not None:
                if t_eff_raw > t_lmtd_raw:
                    t_eff_str += " ↑"; t_lmtd_str += " ↓"
                elif t_eff_raw < t_lmtd_raw:
                    t_eff_str += " ↓"; t_lmtd_str += " ↑"
            rows1.append({
                'Bucket':                   lbl,
                'Today':                    t_eff_str,
                'Yesterday':                fmt_pct(region_y.get(eff_k)),
                'LMTD':                     t_lmtd_str,
                'AOD Flow Value (GA Crs.)': fmt_num(region_t.get(flow_k)) if flow_k and region_t.get(flow_k) is not None else '---',
            })
        st.dataframe(pd.DataFrame(rows1).set_index('Bucket'), use_container_width=True)

        # ── Section 2 ────────────────────────────────────────────────────────
        section(2, "Norm / RB / RF Metrics")
        def _s2_arrow(t_val, l_val):
            """Return (today_suffix, lmtd_suffix) arrows based on raw values."""
            if t_val is None or l_val is None: return "", ""
            if t_val > l_val: return " ↑", " ↓"
            if t_val < l_val: return " ↓", " ↑"
            return "", ""
        _s2_pairs = [
            ("1-29 Norm",           'norm_129',        'norm_129_lmtd',   'norm_129',       'norm_129_proj'),
            ("Stage 2 Roll Back",   's2_rb',           's2_lmtd',         's2_rb',          's2_proj'),
            ("Stage 3 Concern Flow",'s3_concern_flow', 's3_concern_lmtd', 's3_concern_flow','s3_concern_proj'),
            ("Stage 3 Roll Back",   's3_rb',           's3_rb_lmtd',      's3_rb',          's3_rb_proj'),
        ]
        rows2 = []
        for _lbl, _t_k, _l_k, _y_k, _p_k in _s2_pairs:
            _t_raw = extra_t.get(_t_k); _l_raw = extra_t.get(_l_k)
            _ta, _la = _s2_arrow(_t_raw, _l_raw)
            rows2.append({
                "Bucket":          _lbl,
                "Today":           fmt_num(_t_raw) + _ta,
                "Yesterday":       fmt_num(extra_y.get(_y_k)),
                "LMTD":            fmt_num(_l_raw) + _la,
                "Projection Crs.": fmt_num(extra_t.get(_p_k)),
            })
        st.dataframe(pd.DataFrame(rows2).set_index('Bucket'), use_container_width=True)
        st.caption("(a) * Agreement level S3 concern,  (b) Efficiency% excluding hold cases.")

    # ── Section 3 ────────────────────────────────────────────────────────────
    section(3, "Bottom Sub-Regions | Efficiency %")
    eff_b2 = [
        ("Fresh",              'fresh_eff'),
        ("1-29 (S2 Concern)",  '129_eff'),
        ("1-29 Norm",          'norm_eff'),
        ("30-59",              '3059_eff'),
        ("60-89 (S3 Concern)", 's3_eff'),
    ]
    rows3 = []
    for lbl, eff_k in eff_b2:
        b = bottom2_eff(today_m, eff_k, subzones, display)
        rows3.append({
            'Bucket':   lbl,
            'Bottom 1': b[0][0] if len(b) > 0 else '---',
            'Eff %':    fmt_pct(b[0][1]) if len(b) > 0 else '---',
            'Bottom 2': b[1][0] if len(b) > 1 else '---',
            'Eff % ':   fmt_pct(b[1][1]) if len(b) > 1 else '---',
        })
    st.dataframe(pd.DataFrame(rows3).set_index('Bucket'), use_container_width=True)

    # ── Section 4 ────────────────────────────────────────────────────────────
    section(4, "Bottom Sub-Regions | AOD Flow Value Crs.")
    flow_b2 = [
        ("Fresh",              'fresh_flow'),
        ("1-29 (S2 Concern)",  '129_flow'),
        ("30-59",              '3059_flow'),
        ("60-89 (S3 Concern)", 's3_flow'),
    ]
    rows4 = []
    for lbl, flow_k in flow_b2:
        b = bottom2_flow(today_m, flow_k, subzones, display)
        rows4.append({
            'Bucket':      lbl,
            'Bottom 1':    b[0][0] if len(b) > 0 else '---',
            'Flow Value':  fmt_num(b[0][1]) if len(b) > 0 else '---',
            'Bottom 2':    b[1][0] if len(b) > 1 else '---',
            'Flow Value ': fmt_num(b[1][1]) if len(b) > 1 else '---',
        })
    st.dataframe(pd.DataFrame(rows4).set_index('Bucket'), use_container_width=True)


# ── PNG Image Generator ───────────────────────────────────────────────────────
def create_report_image(today_m, yest_m, extra_t, extra_y, report_date, view_key):
    cfg      = ZONE_CONFIG[view_key]
    region_t = today_m.get(view_key, {})
    region_y = yest_m.get(view_key, {})
    subzones = cfg['subzones']
    display  = cfg['display']
    label    = cfg['label']

    width = 3000; height = 8500
    img  = Image.new("RGB", (width, height), "#ffffff")
    draw = ImageDraw.Draw(img)

    COLOR_HEADER_BG      = "#0f2a5f"
    COLOR_HEADER_TEXT    = "#ffffff"
    COLOR_SECTION_BG     = "#1a4d8f"
    COLOR_SECTION_TEXT   = "#ffffff"
    COLOR_GRID_HEADER_BG = "#e8f1f9"
    COLOR_GRID_HEADER_T  = "#1a4d8f"
    COLOR_GRID_BORDER    = "#dce5f0"
    COLOR_GRID_ROW_ALT   = "#f7fbff"
    COLOR_TEXT           = "#0f2a5f"
    COLOR_TEXT_SEC       = "#4a5568"
    COLOR_SUBZONE        = "#7a3d0d"

    def _load_font(size):
        # Try real TrueType fonts first (cross-platform candidates), and only
        # fall back to PIL's built-in font as an absolute last resort. Without
        # this, "arial.ttf" fails to load on most Linux servers (it simply
        # isn't installed there) and PIL silently substitutes a tiny, fixed
        # bitmap font — which is why text used to render small/misaligned
        # regardless of the size numbers below.
        candidates = [
            "arial.ttf", "Arial.ttf", "arialbd.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/Library/Fonts/Arial.ttf",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "C:\\Windows\\Fonts\\arial.ttf",
        ]
        for path in candidates:
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
        try:
            return ImageFont.load_default(size=size)  # Pillow >= 9.2 scalable fallback
        except TypeError:
            return ImageFont.load_default()

    font_title    = _load_font(190)
    font_subtitle = _load_font(88)
    font_section  = _load_font(100)
    font_header   = _load_font(72)
    font_body     = _load_font(66)
    font_small    = _load_font(56)

    y = 0
    # Header
    draw.rectangle([0, 0, width, 460], fill=COLOR_HEADER_BG)
    draw.text((95, 60), "DAILY COLLECTIONS UPDATE", fill=COLOR_HEADER_TEXT, font=font_title)
    draw.text((95, 286), f"{label}  |  {report_date}", fill=COLOR_HEADER_TEXT, font=font_subtitle)
    y = 590

    row_height = 170
    BODY_OFFSET = 46

    def draw_section_header(title):
        nonlocal y
        draw.rectangle([60, y, width - 60, y + 172], fill=COLOR_SECTION_BG)
        draw.text((95, y + 36), title, fill=COLOR_SECTION_TEXT, font=font_section)
        y += 202

    def draw_table_header(headers, col_x):
        nonlocal y
        draw.rectangle([60, y, width - 60, y + 175], fill=COLOR_GRID_HEADER_BG)
        for i, h in enumerate(headers):
            draw.text((col_x[i] + 18, y + 50), h, fill=COLOR_GRID_HEADER_T, font=font_header)
        draw.line([60, y + 175, width - 60, y + 175], fill=COLOR_GRID_BORDER, width=3)
        y += 181

    def draw_row(cells, col_x, colors, idx):
        nonlocal y
        if idx % 2 == 0:
            draw.rectangle([60, y, width - 60, y + row_height], fill=COLOR_GRID_ROW_ALT)
        for x in col_x[1:] + [width - 60]:
            draw.line([x, y, x, y + row_height], fill=COLOR_GRID_BORDER, width=2)
        for i, (text, color) in enumerate(zip(cells, colors)):
            draw.text((col_x[i] + 18, y + BODY_OFFSET), str(text), fill=color, font=font_body)
        y += row_height

    def get_w(txt):
        try: return draw.textlength(txt, font=font_body)
        except:
            try: return font_body.getlength(txt)
            except: return len(txt) * 8

    def wrap_names(text, max_width):
        """Wrap a comma-separated list of region names to fit within max_width,
        keeping each region name whole (never splitting a name mid-word)."""
        parts = [p.strip() for p in str(text).split(",")]
        lines, current = [], ""
        for p in parts:
            candidate = p if not current else current + ", " + p
            if get_w(candidate) <= max_width or not current:
                current = candidate
            else:
                lines.append(current)
                current = p
        if current:
            lines.append(current)
        return lines if lines else [str(text)]

    def draw_row_wrapped(cells, col_x, colors, idx, wrap_cols=None, wrap_width=600):
        """Like draw_row, but cells listed in wrap_cols get wrapped onto
        multiple lines (instead of overflowing into neighbouring columns)
        when the text is too long — e.g. multiple tied region names."""
        nonlocal y
        wrap_cols = wrap_cols or set()
        line_height = 78

        cell_lines = []
        max_lines = 1
        for i, text in enumerate(cells):
            if i in wrap_cols:
                lines = wrap_names(text, wrap_width)
            else:
                lines = [str(text)]
            cell_lines.append(lines)
            max_lines = max(max_lines, len(lines))

        this_row_height = max(row_height, BODY_OFFSET + 12 + max_lines * line_height)

        if idx % 2 == 0:
            draw.rectangle([60, y, width - 60, y + this_row_height], fill=COLOR_GRID_ROW_ALT)
        for x in col_x[1:] + [width - 60]:
            draw.line([x, y, x, y + this_row_height], fill=COLOR_GRID_BORDER, width=2)
        for i, lines in enumerate(cell_lines):
            for li, line in enumerate(lines):
                draw.text((col_x[i] + 18, y + BODY_OFFSET + li * line_height), line,
                           fill=colors[i], font=font_body)
        y += this_row_height

    # ── Section 1 & 2 (hidden for combined zones) ───────────────────────────
    if view_key not in {'MP_AND_GUJARAT', 'ROMH_AND_MUMBAI'}:
        # ── Section 1 ────────────────────────────────────────────────────────
        draw_section_header("1. Efficiency %")
        col_x1 = [95, 750, 1150, 1550, 1950]
        draw_table_header(["Bucket", "Today", "Yesterday", "LMTD", "AOD Flow (GA Crs.)"], col_x1)
        buckets_s1 = [
            ("Fresh",              'fresh_eff', 'fresh_lmtd', 'fresh_flow'),
            ("1-29",               '129_eff',   '129_lmtd',   '129_flow'),
            ("1-29 Norm%",         'norm_eff',  'norm_lmtd',  None),
            ("30-59",              '3059_eff',  '3059_lmtd',  '3059_flow'),
            ("60-89 (S3 Concern)", 's3_eff',    's3_lmtd',    's3_flow'),
        ]
        for idx, (lbl, eff_k, lmtd_k, flow_k) in enumerate(buckets_s1):
            t_eff_raw  = region_t.get(eff_k)
            t_lmtd_raw = region_t.get(lmtd_k)
            t_eff_s    = fmt_pct(t_eff_raw)
            t_lmtd_s   = fmt_pct(t_lmtd_raw)
            t_flow_s   = fmt_num(region_t.get(flow_k)) if flow_k and region_t.get(flow_k) is not None else "---"
            draw_row([lbl, t_eff_s, fmt_pct(region_y.get(eff_k)), t_lmtd_s, t_flow_s],
                     col_x1, [COLOR_TEXT, COLOR_TEXT, COLOR_TEXT_SEC, COLOR_TEXT_SEC, COLOR_SUBZONE], idx)
            if t_eff_raw is not None and t_lmtd_raw is not None:
                arrow_y = y - row_height + BODY_OFFSET
                w_eff   = get_w(t_eff_s)
                w_lmtd  = get_w(t_lmtd_s)
                if t_eff_raw > t_lmtd_raw:
                    draw.text((col_x1[1] + 18 + w_eff + 8, arrow_y), "↑", fill="#38a169", font=font_body)
                    draw.text((col_x1[3] + 18 + w_lmtd + 8, arrow_y), "↓", fill="#e53e3e", font=font_body)
                elif t_eff_raw < t_lmtd_raw:
                    draw.text((col_x1[1] + 18 + w_eff + 8, arrow_y), "↓", fill="#e53e3e", font=font_body)
                    draw.text((col_x1[3] + 18 + w_lmtd + 8, arrow_y), "↑", fill="#38a169", font=font_body)
        draw.line([60, y, width - 60, y], fill=COLOR_GRID_BORDER, width=3)
        y += 95

        # ── Section 2 ────────────────────────────────────────────────────────
        draw_section_header("2. Norm / RB / RF Metrics")
        col_x2 = [95, 750, 1150, 1550, 1950]
        draw_table_header(["Bucket", "Today", "Yesterday", "LMTD", "Projection Crs."], col_x2)
        _s2_pairs_png = [
            ("1-29 Norm",           'norm_129',        'norm_129_lmtd',   'norm_129',       'norm_129_proj'),
            ("Stage 2 Roll Back",   's2_rb',           's2_lmtd',         's2_rb',          's2_proj'),
            ("Stage 3 Concern Flow",'s3_concern_flow', 's3_concern_lmtd', 's3_concern_flow','s3_concern_proj'),
            ("Stage 3 Roll Back",   's3_rb',           's3_rb_lmtd',      's3_rb',          's3_rb_proj'),
        ]
        for idx, (_lbl, _t_k, _l_k, _y_k, _p_k) in enumerate(_s2_pairs_png):
            _t_raw = extra_t.get(_t_k); _l_raw = extra_t.get(_l_k)
            _tv = fmt_num(_t_raw); _lv = fmt_num(_l_raw)
            draw_row([_lbl, _tv, fmt_num(extra_y.get(_y_k)), _lv, fmt_num(extra_t.get(_p_k))],
                     col_x2, [COLOR_TEXT, COLOR_TEXT, COLOR_TEXT_SEC, COLOR_TEXT_SEC, COLOR_SUBZONE], idx)
            if _t_raw is not None and _l_raw is not None and _t_raw != _l_raw:
                _arrow_y = y - row_height + BODY_OFFSET
                _w_t = get_w(_tv); _w_l = get_w(_lv)
                if _t_raw > _l_raw:
                    draw.text((col_x2[1] + 18 + _w_t + 8, _arrow_y), "↑", fill="#38a169", font=font_body)
                    draw.text((col_x2[3] + 18 + _w_l + 8, _arrow_y), "↓", fill="#e53e3e", font=font_body)
                else:
                    draw.text((col_x2[1] + 18 + _w_t + 8, _arrow_y), "↓", fill="#e53e3e", font=font_body)
                    draw.text((col_x2[3] + 18 + _w_l + 8, _arrow_y), "↑", fill="#38a169", font=font_body)
        draw.line([60, y, width - 60, y], fill=COLOR_GRID_BORDER, width=3)
        y += 38
        draw.text((95, y), "(a) * Agreement level S3 concern,  (b) Efficiency% excluding hold cases.",
                  fill=COLOR_TEXT_SEC, font=font_small)
        y += 95

    # ── Section 3 ─────────────────────────────────────────────────────────────
    draw_section_header("3. Bottom Sub-Regions | Efficiency %")
    col_x3 = [95, 750, 1350, 1770, 2350]
    draw_table_header(["Bucket", "Bottom 1", "Eff %", "Bottom 2", "Eff %"], col_x3)
    buckets_s3 = [
        ("Fresh",              'fresh_eff'),
        ("1-29 (S2 Concern)",  '129_eff'),
        ("1-29 Norm",          'norm_eff'),
        ("30-59",              '3059_eff'),
        ("60-89 (S3 Concern)", 's3_eff'),
    ]
    for idx, (lbl, eff_k) in enumerate(buckets_s3):
        b = bottom2_eff(today_m, eff_k, subzones, display)
        b1n = b[0][0] if len(b) > 0 else "---"; b1v = fmt_pct(b[0][1]) if len(b) > 0 else "---"
        b2n = b[1][0] if len(b) > 1 else "---"; b2v = fmt_pct(b[1][1]) if len(b) > 1 else "---"
        draw_row_wrapped([lbl, b1n, b1v, b2n, b2v], col_x3,
                 [COLOR_TEXT, COLOR_SUBZONE, COLOR_TEXT, COLOR_SUBZONE, COLOR_TEXT_SEC], idx,
                 wrap_cols={1, 3}, wrap_width=540)
    draw.line([60, y, width - 60, y], fill=COLOR_GRID_BORDER, width=3)
    y += 95

    # ── Section 4 ─────────────────────────────────────────────────────────────
    draw_section_header("4. Bottom Sub-Regions | AOD Flow Value (Crs.)")
    col_x4 = [95, 750, 1350, 1770, 2350]
    draw_table_header(["Bucket", "Bottom 1", "Flow Value", "Bottom 2", "Flow Value"], col_x4)
    buckets_s4 = [
        ("Fresh",              'fresh_flow'),
        ("1-29 (S2 Concern)",  '129_flow'),
        ("30-59",              '3059_flow'),
        ("60-89 (S3 Concern)", 's3_flow'),
    ]
    for idx, (lbl, flow_k) in enumerate(buckets_s4):
        b = bottom2_flow(today_m, flow_k, subzones, display)
        b1n = b[0][0] if len(b) > 0 else "---"; b1v = fmt_num(b[0][1]) if len(b) > 0 else "---"
        b2n = b[1][0] if len(b) > 1 else "---"; b2v = fmt_num(b[1][1]) if len(b) > 1 else "---"
        draw_row_wrapped([lbl, b1n, b1v, b2n, b2v], col_x4,
                 [COLOR_TEXT, COLOR_SUBZONE, COLOR_TEXT, COLOR_SUBZONE, COLOR_TEXT_SEC], idx,
                 wrap_cols={1, 3}, wrap_width=540)
    draw.line([60, y, width - 60, y], fill=COLOR_GRID_BORDER, width=3)
    y += 150

    # Footer
    draw.text((95, y), f"Generated on {date.today().strftime('%d %b %Y')} | Collections MIS Report | Confidential",
              fill=COLOR_TEXT_SEC, font=font_small)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


# ── HTML Report Generator ─────────────────────────────────────────────────────
def build_html_report(today_m, yest_m, extra_t, extra_y, date_str, view_key):
    cfg      = ZONE_CONFIG[view_key]
    region_t = today_m.get(view_key, {})
    region_y = yest_m.get(view_key, {})
    subzones = cfg['subzones']
    display  = cfg['display']
    label    = cfg['label']

    def arrow(t, lmtd):
        if t is None or lmtd is None: return "", ""
        if t > lmtd:
            return ' <span style="color:#38a169;font-weight:bold;">↑</span>', \
                   ' <span style="color:#e53e3e;font-weight:bold;">↓</span>'
        if t < lmtd:
            return ' <span style="color:#e53e3e;font-weight:bold;">↓</span>', \
                   ' <span style="color:#38a169;font-weight:bold;">↑</span>'
        return "", ""

    def s1_rows():
        buckets = [
            ("Fresh",              'fresh_eff','fresh_lmtd','fresh_flow'),
            ("1–29",               '129_eff',  '129_lmtd',  '129_flow'),
            ("1–29 Norm%",         'norm_eff', 'norm_lmtd', None),
            ("30-59",              '3059_eff', '3059_lmtd', '3059_flow'),
            ("60-89 (S3 CONCERN)", 's3_eff',   's3_lmtd',   's3_flow'),
        ]
        html = ""
        for lbl, eff_k, lmtd_k, flow_k in buckets:
            t = region_t.get(eff_k); l = region_t.get(lmtd_k)
            ta, la = arrow(t, l)
            html += f"""<tr>
<td class="label">{lbl}</td>
<td class="val today">{fmt_pct(t)}{ta}</td>
<td class="val">{fmt_pct(region_y.get(eff_k))}</td>
<td class="val">{fmt_pct(l)}{la}</td>
<td class="val flow">{fmt_num(region_t.get(flow_k)) if flow_k and region_t.get(flow_k) is not None else '---'}</td>
</tr>"""
        return html

    def s2_rows():
        _pairs = [
            ("1-29 Norm",           'norm_129',        'norm_129_lmtd',   'norm_129',       'norm_129_proj'),
            ("Stage 2 Roll Back",   's2_rb',           's2_lmtd',         's2_rb',          's2_proj'),
            ("Stage 3 Concern Flow",'s3_concern_flow', 's3_concern_lmtd', 's3_concern_flow','s3_concern_proj'),
            ("Stage 3 Roll Back",   's3_rb',           's3_rb_lmtd',      's3_rb',          's3_rb_proj'),
        ]
        html = ""
        for _lbl, _t_k, _l_k, _y_k, _p_k in _pairs:
            _t_raw = extra_t.get(_t_k); _l_raw = extra_t.get(_l_k)
            _tv = fmt_num(_t_raw); _lv = fmt_num(_l_raw)
            if _t_raw is not None and _l_raw is not None and _t_raw != _l_raw:
                if _t_raw > _l_raw:
                    _tv += ' <span style="color:#38a169;font-weight:bold;">↑</span>'
                    _lv += ' <span style="color:#e53e3e;font-weight:bold;">↓</span>'
                else:
                    _tv += ' <span style="color:#e53e3e;font-weight:bold;">↓</span>'
                    _lv += ' <span style="color:#38a169;font-weight:bold;">↑</span>'
            html += f'''<tr>
<td class="label">{_lbl}</td><td class="val today">{_tv}</td>
<td class="val">{fmt_num(extra_y.get(_y_k))}</td><td class="val">{_lv}</td>
<td class="val flow">{fmt_num(extra_t.get(_p_k))}</td>
</tr>'''
        return html

    def s3_rows():
        buckets = [
            ("Fresh",              'fresh_eff'),
            ("1-29 (S2 Concern)",  '129_eff'),
            ("1-29 Norm",          'norm_eff'),
            ("30-59",              '3059_eff'),
            ("60-89 (S3 Concern)", 's3_eff'),
        ]
        html = ""
        for lbl, eff_k in buckets:
            b = bottom2_eff(today_m, eff_k, subzones, display)
            b1n = b[0][0] if len(b)>0 else '---'; b1v = fmt_pct(b[0][1]) if len(b)>0 else '---'
            b2n = b[1][0] if len(b)>1 else '---'; b2v = fmt_pct(b[1][1]) if len(b)>1 else '---'
            html += f"""<tr>
<td class="label">{lbl}</td><td class="val subzone">{b1n}</td><td class="val today">{b1v}</td>
<td class="val subzone">{b2n}</td><td class="val">{b2v}</td></tr>"""
        return html

    def s4_rows():
        buckets = [
            ("Fresh",              'fresh_flow'),
            ("1-29 (S2 Concern)",  '129_flow'),
            ("30-59",              '3059_flow'),
            ("60-89 (S3 Concern)", 's3_flow'),
        ]
        html = ""
        for lbl, flow_k in buckets:
            b = bottom2_flow(today_m, flow_k, subzones, display)
            b1n = b[0][0] if len(b)>0 else '---'; b1v = fmt_num(b[0][1]) if len(b)>0 else '---'
            b2n = b[1][0] if len(b)>1 else '---'; b2v = fmt_num(b[1][1]) if len(b)>1 else '---'
            html += f"""<tr>
<td class="label">{lbl}</td><td class="val subzone">{b1n}</td><td class="val today">{b1v}</td>
<td class="val subzone">{b2n}</td><td class="val">{b2v}</td></tr>"""
        return html

    css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Inter',sans-serif;background:#f5f7fa;color:#0f2a5f;padding:36px 24px}
.report-wrapper{max-width:880px;margin:0 auto}
.report-header{background:linear-gradient(135deg,#0f2a5f 0%,#1a4d8f 50%,#2563c4 100%);
  border-radius:18px 18px 0 0;padding:32px 40px 28px;color:#fff;display:flex;justify-content:space-between;
  align-items:flex-end;box-shadow:0 8px 24px rgba(15,42,95,.15);border:1px solid rgba(255,255,255,.08)}
.report-header .title{font-size:22px;font-weight:800;letter-spacing:.6px}
.report-header .subtitle{font-size:13px;opacity:.85;margin-top:5px;letter-spacing:.3px}
.report-header .badge{background:rgba(255,255,255,.15);border:1.5px solid rgba(255,255,255,.4);
  border-radius:10px;padding:8px 16px;font-size:13px;font-weight:700;backdrop-filter:blur(10px)}
.report-body{background:#fff;border-radius:0 0 18px 18px;padding:36px 40px 40px;
  box-shadow:0 4px 20px rgba(0,0,0,.07);border:1px solid #e8ecf1}
.section{margin-bottom:40px}.section:last-child{margin-bottom:0}
.section-footer-note{font-size:11px;color:#718096;margin-top:10px;padding-left:4px}
.section-header{display:flex;align-items:center;gap:12px;margin-bottom:16px}
.section-number{background:linear-gradient(135deg,#0f2a5f 0%,#1a4d8f 100%);color:#fff;font-size:12px;
  font-weight:800;width:28px;height:28px;border-radius:50%;display:flex;align-items:center;
  justify-content:center;flex-shrink:0}
.section-title{font-size:15px;font-weight:800;color:#0f2a5f;text-transform:uppercase;letter-spacing:.6px}
table{width:100%;border-collapse:separate;border-spacing:0;border-radius:12px;overflow:hidden;
  border:1px solid #dce5f0;font-size:13px}
thead tr{background:linear-gradient(90deg,#e8f1f9 0%,#f0f5fb 100%)}
thead th{padding:12px 14px;text-align:center;font-size:11px;font-weight:800;color:#1a4d8f;
  text-transform:uppercase;letter-spacing:.5px;border-bottom:2px solid #bee3f8}
thead th:first-child{text-align:left}
tbody tr:nth-child(even){background:#f7fbff}
tbody tr:hover{background:#e8f1f9}
td{padding:12px 14px;border-bottom:1px solid #dce5f0}
tbody tr:last-child td{border-bottom:none}
td.label{font-weight:700;color:#0f2a5f;text-align:left;white-space:nowrap}
td.val{text-align:center;color:#4a5568;font-variant-numeric:tabular-nums}
td.today{font-weight:800;color:#0f2a5f}
td.flow{color:#276749;font-weight:700}
td.subzone{font-weight:700;color:#7a3d0d}
.report-footer{text-align:center;margin-top:24px;font-size:11px;color:#a0aec0;letter-spacing:.3px}
</style>"""

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"/>
<title>Daily Collections Update | {date_str} | {label}</title>{css}</head>
<body><div class="report-wrapper">
<div class="report-header">
  <div><div class="title">DAILY COLLECTIONS UPDATE</div>
  <div class="subtitle">{label} — Efficiency Summary</div></div>
  <div class="badge">📅 {date_str}</div>
</div>
<div class="report-body">

("" if view_key in {'MP_AND_GUJARAT', 'ROMH_AND_MUMBAI'} else ('<div class="section">\n<div class="section-header"><div class="section-number">1</div>\n<div class="section-title">Efficiency %</div></div>\n<table><thead><tr><th>Bucket</th><th>Today</th><th>Yesterday</th><th>LMTD</th>\n<th>AOD Flow Value (GA Crs.)</th></tr></thead>\n<tbody>' + s1_rows() + '</tbody></table></div>\n\n<div class="section">\n<div class="section-header"><div class="section-number">2</div>\n<div class="section-title">Norm / RB / RF Metrics</div></div>\n<table><thead><tr><th>Bucket</th><th>Today</th><th>Yesterday</th><th>LMTD</th>\n<th>Projection Crs.</th></tr></thead>\n<tbody>' + s2_rows() + '</tbody></table>\n<div class="section-footer-note">(a) * Agreement level S3 concern, (b) Efficiency% excluding hold cases.</div>\n</div>\n\n'))<div class="section">
<div class="section-header"><div class="section-number">3</div>
<div class="section-title">Bottom Sub-Regions | Efficiency %</div></div>
<table><thead><tr><th>Bucket</th><th>Bottom 1</th><th>Eff %</th><th>Bottom 2</th>
<th>Eff %</th></tr></thead>
<tbody>{s3_rows()}</tbody></table></div>

<div class="section">
<div class="section-header"><div class="section-number">4</div>
<div class="section-title">Bottom Sub-Regions | AOD Flow Value Crs.</div></div>
<table><thead><tr><th>Bucket</th><th>Bottom 1</th><th>Flow Value</th><th>Bottom 2</th>
<th>Flow Value</th></tr></thead>
<tbody>{s4_rows()}</tbody></table></div>

</div>
<div class="report-footer">Generated on {date.today().strftime("%d %b %Y")} &nbsp;|&nbsp;
Collections MIS Report &nbsp;|&nbsp; Confidential</div>
</div></body></html>"""


# ── UI ────────────────────────────────────────────────────────────────────────

# App banner
st.markdown("""
<div class="app-banner">
  <div>
    <div class="banner-title">📊 Daily Collections Update Generator</div>
    <div class="banner-sub">Upload files · Choose region · Generate professional report</div>
  </div>
  <div class="banner-badge">MIS Report Tool</div>
</div>
""", unsafe_allow_html=True)

# ── File uploads ──────────────────────────────────────────────────────────────
col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown('<div class="upload-card"><div class="card-title">📅 Today\'s Files</div>', unsafe_allow_html=True)
    today_file      = st.file_uploader("Dashboard (Main)",      type=['xlsx','xls','xlsb','csv'], key='today',       label_visibility="visible")
    today_s2_file   = st.file_uploader("Stage 2 Summary",       type=['xlsx','xls','xlsb','csv'], key='today_s2',    label_visibility="visible")
    today_s3_file   = st.file_uploader("Stage 3 Concern",       type=['xlsx','xls','xlsb','csv'], key='today_s3',    label_visibility="visible")
    today_s3rb_file = st.file_uploader("Stage 3 Roll Back",     type=['xlsx','xls','xlsb','csv'], key='today_s3rb',  label_visibility="visible")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="upload-card"><div class="card-title">📅 Yesterday\'s Files</div>', unsafe_allow_html=True)
    yest_file      = st.file_uploader("Dashboard (Main)",      type=['xlsx','xls','xlsb','csv'], key='yest',        label_visibility="visible")
    yest_s2_file   = st.file_uploader("Stage 2 Summary",       type=['xlsx','xls','xlsb','csv'], key='yest_s2',     label_visibility="visible")
    yest_s3_file   = st.file_uploader("Stage 3 Concern",       type=['xlsx','xls','xlsb','csv'], key='yest_s3',     label_visibility="visible")
    yest_s3rb_file = st.file_uploader("Stage 3 Roll Back",     type=['xlsx','xls','xlsb','csv'], key='yest_s3rb',   label_visibility="visible")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Region selector & date ────────────────────────────────────────────────────
st.markdown('<div class="scope-card"><div class="scope-title">🗺️ Report Scope & Date</div>', unsafe_allow_html=True)

cfg_col1, cfg_col2, cfg_col3 = st.columns([2, 3, 2], gap="medium")

with cfg_col1:
    scope = st.radio(
        "View type",
        ["🌐  PAN INDIA", "📍  Sub-Region"],
        label_visibility="collapsed",
    )

region_options = {
    "North 1  —  Delhi / Gurugram / UP":         "NORTH_1",
    "North 2  —  PHC: Punjab / Haryana / UK":    "NORTH_2",
    "North 3  —  Rajasthan":                     "NORTH_3",
    "East 1   —  Chattisgarh / Odisha / WB":     "EAST_1",
    "East 2   —  Bihar / Jharkhand / NE":        "EAST_2",
    "South 1  —  AP / Kerala / Tamil Nadu":      "SOUTH_1",
    "South 2  —  Karnataka / Telangana":         "SOUTH_2",
    "West 1   —  Gujarat / ROMH":                "WEST_1",
    "West 2   —  MP / Mumbai & Goa":             "WEST_2",
    "West    —  MP & Gujarat":                   "MP_AND_GUJARAT",
    "West    —  ROMH & Mumbai":                  "ROMH_AND_MUMBAI",
}

with cfg_col2:
    if "Sub-Region" in scope:
        chosen   = st.selectbox("Choose sub-region", list(region_options.keys()), label_visibility="collapsed")
        view_key = region_options[chosen]
    else:
        st.markdown("<div style='color:#718096;font-size:13px;padding-top:8px;'>All 9 sub-zones included</div>", unsafe_allow_html=True)
        view_key = "PAN_INDIA"

with cfg_col3:
    report_date = st.text_input("Report Date", value="22-Jun-26", placeholder="e.g. 22-Jun-26", label_visibility="collapsed")
    st.markdown("<div style='color:#718096;font-size:11px;margin-top:-8px;'>Report date (e.g. 22-Jun-26)</div>", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

generate = st.button("⚡  Generate Report", type="primary", use_container_width=True)

if generate:
    if not today_file or not yest_file:
        st.error("⚠️ Please upload at least both dashboard files — Today and Yesterday.")
    else:
        with st.spinner("Parsing files and building report…"):
            try:
                today_dfs    = load_all_sheets(today_file)
                yest_dfs     = load_all_sheets(yest_file)
                today_s2_dfs = load_all_sheets(today_s2_file) if today_s2_file else {}
                yest_s2_dfs  = load_all_sheets(yest_s2_file)  if yest_s2_file  else {}
                today_s3_dfs = load_all_sheets(today_s3_file) if today_s3_file else {}
                yest_s3_dfs  = load_all_sheets(yest_s3_file)  if yest_s3_file  else {}

                today_df_main = today_dfs.get("PROJ OVERALL",
                                list(today_dfs.values())[0] if today_dfs else pd.DataFrame())
                yest_df_main  = yest_dfs.get("PROJ OVERALL",
                                list(yest_dfs.values())[0]  if yest_dfs  else pd.DataFrame())

                today_rows = find_data_rows(today_df_main)
                yest_rows  = find_data_rows(yest_df_main)

                if not today_rows or not yest_rows:
                    which = "Today's" if not today_rows else "Yesterday's"
                    df_bad = today_df_main if not today_rows else yest_df_main
                    detected_col = _detect_zone_col(df_bad) if not df_bad.empty else "N/A"
                    sample_vals = []
                    if not df_bad.empty:
                        for ci in range(min(5, len(df_bad.columns))):
                            vals = [str(df_bad.iloc[r, ci]).strip() for r in range(min(20, len(df_bad)))
                                    if str(df_bad.iloc[r, ci]).strip() not in ("nan","NaT","")]
                            if vals:
                                sample_vals.append(f"col{ci}: {vals[:3]}")
                    st.error(
                        f"⚠️ Could not find zone data in **{which} Dashboard** file. "
                        f"Best column detected: col[{detected_col}]. "
                        f"Sample values found: {'; '.join(sample_vals[:4]) if sample_vals else 'none'}. "
                        f"Make sure you are uploading the correct **Dashboard (Main)** file — "
                        f"not the Stage 2 or Stage 3 file."
                    )
                else:
                    today_m = extract_all(today_df_main, today_rows)
                    yest_m  = extract_all(yest_df_main, yest_rows)

                    dash_t_extra = extract_dashboard_metrics(today_df_main, view_key)
                    dash_y_extra = extract_dashboard_metrics(yest_df_main, view_key)
                    s2_t_extra   = extract_s2_metrics(today_s2_dfs, view_key)
                    s2_y_extra   = extract_s2_metrics(yest_s2_dfs, view_key)
                    s3_t_extra   = extract_s3_metrics(today_s3_dfs, view_key)
                    s3_y_extra   = extract_s3_metrics(yest_s3_dfs, view_key)
                    s3rb_t_extra = extract_s3_rb_metrics(today_s3rb_file, view_key)
                    s3rb_y_extra = extract_s3_rb_metrics(yest_s3rb_file, view_key)

                    # Stage 3 Roll Back metrics come EXCLUSIVELY from the dedicated
                    # S3 Roll Back file. If that file isn't uploaded, these stay
                    # None (rendered as '---') — they must NOT fall back to any
                    # value from the main dashboard or other files.
                    extra_t = {**dash_t_extra, **s2_t_extra, **s3_t_extra, **s3rb_t_extra}
                    extra_y = {**dash_y_extra, **s2_y_extra, **s3_y_extra, **s3rb_y_extra}

                    # Combined zones have no single parent row — that's expected
                    COMBINED_ZONES = {'MP_AND_GUJARAT', 'ROMH_AND_MUMBAI'}
                    if view_key not in today_m and view_key not in COMBINED_ZONES:
                        st.warning(
                            f"⚠️ No data found for **{ZONE_CONFIG[view_key]['label']}** in today's file. "
                            "Check that the zone row exists in column A."
                        )

                    # ── Render inline report ───────────────────────────────
                    st.markdown("---")
                    render_report(today_m, yest_m, extra_t, extra_y, report_date, view_key)

                    # ── Debug expander ─────────────────────────────────────
                    with st.expander("🔍 Debug: Section 2 raw values (click to verify zone data)"):
                        st.write(f"**View key (zone):** `{view_key}`")
                        st.write("**Today extra_t:**", extra_t)
                        st.write("**Yesterday extra_y:**", extra_y)

                    # ── Downloads ─────────────────────────────────────────
                    st.markdown("---")

                    safe_label = ZONE_CONFIG[view_key]['label'].replace(' ', '_')

                    png_buf = create_report_image(today_m, yest_m, extra_t, extra_y, report_date, view_key)
                    st.download_button(
                        label="📷  Download Report Image (PNG) — Mobile Ready",
                        data=png_buf,
                        file_name=f"Collections_{safe_label}_{report_date}.png",
                        mime="image/png",
                        use_container_width=True,
                    )

            except Exception as e:
                st.error(f"Error: {e}")
                st.exception(e)

st.markdown("<br>", unsafe_allow_html=True)
with st.expander("ℹ️  How to use this tool"):
    st.markdown("""
| Step | Action |
|------|--------|
| 1 | Upload **Today's** Dashboard, Stage 2 Summary, and Stage 3 Concern files on the left |
| 2 | Upload the matching **Yesterday's** files on the right |
| 3 | Choose **PAN INDIA** for the overall view, or pick a specific **Sub-Region** |
| 4 | Enter the report date and click **Generate Report** |
| 5 | Download the **PNG** image (optimized for mobile & WhatsApp) |

> Stage 2 and Stage 3 files are optional but recommended — without them Section 2 (Norm / RB / RF Metrics) will show `---`.
""")