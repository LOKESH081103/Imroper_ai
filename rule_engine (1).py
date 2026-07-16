"""
Layers 1 & 3 - structural rules + placeholder/gibberish dictionary.

Pulled out of app.py into its own module so the exact same rule logic can be
reused outside Streamlit - e.g. by dashboard_report.py to build a standalone
presentation dashboard from the command line, or by tests, notebooks, or a
future batch/cron job. app.py imports everything it needs from here; there
is no second copy of this logic anywhere in the project.

Layer 2 (pincode API) and Layer 4 (ML) live in pincode_lookup.py and
ml_classifier.py respectively, for the same reason.
"""

import re

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

# Recognized city/town names. An address that names a real Indian city is
# clearly an Indian address even if the state itself is never spelled out
# (e.g. "...Chennai - 600 037" - everyone knows Chennai is in Tamil Nadu).
# So STATE_NOT_FOUND only fires when NEITHER a state NOR one of these
# cities appears anywhere in the address. Not exhaustive - India has
# thousands of towns - but covers the major metros/state capitals/big
# cities that show up constantly in real address data. Extend this set as
# you find more false positives in your own data.
INDIAN_CITY_HINTS = {
    "MUMBAI", "DELHI", "NEW DELHI", "BANGALORE", "BENGALURU", "HYDERABAD", "CHENNAI",
    "KOLKATA", "PUNE", "AHMEDABAD", "SURAT", "JAIPUR", "LUCKNOW", "KANPUR", "NAGPUR",
    "INDORE", "THANE", "BHOPAL", "VISAKHAPATNAM", "VIZAG", "PATNA", "VADODARA",
    "GHAZIABAD", "LUDHIANA", "AGRA", "NASHIK", "FARIDABAD", "MEERUT", "RAJKOT",
    "VARANASI", "SRINAGAR", "AMRITSAR", "NAVI MUMBAI", "PRAYAGRAJ", "ALLAHABAD",
    "RANCHI", "HOWRAH", "COIMBATORE", "JABALPUR", "GWALIOR", "VIJAYAWADA", "JODHPUR",
    "MADURAI", "RAIPUR", "KOTA", "GUWAHATI", "CHANDIGARH", "SOLAPUR", "HUBLI",
    "MYSORE", "MYSURU", "TIRUCHIRAPPALLI", "TRICHY", "BAREILLY", "ALIGARH",
    "MORADABAD", "SALEM", "THIRUVANANTHAPURAM", "TRIVANDRUM", "BHIWANDI",
    "SAHARANPUR", "GORAKHPUR", "GUNTUR", "BIKANER", "AMRAVATI", "NOIDA", "GREATER NOIDA",
    "JAMSHEDPUR", "BHILAI", "WARANGAL", "CUTTACK", "KOCHI", "COCHIN", "NELLORE",
    "BHAVNAGAR", "DEHRADUN", "DURGAPUR", "ASANSOL", "ROURKELA", "NANDED", "KOLHAPUR",
    "AJMER", "AKOLA", "GULBARGA", "JAMNAGAR", "UJJAIN", "SILIGURI", "JHANSI", "JAMMU",
    "MANGALORE", "MANGALURU", "ERODE", "BELGAUM", "TIRUNELVELI", "MALEGAON", "GAYA",
    "JALANDHAR", "BHUBANESWAR", "TIRUPUR", "DAVANAGERE", "KOZHIKODE", "CALICUT",
    "KURNOOL", "BOKARO", "RAJAHMUNDRY", "BELLARY", "PATIALA", "AGARTALA", "BHAGALPUR",
    "MUZAFFARNAGAR", "LATUR", "DHULE", "TIRUPATI", "ROHTAK", "KORBA", "BHILWARA",
    "BRAHMAPUR", "MUZAFFARPUR", "AHMEDNAGAR", "MATHURA", "KOLLAM", "AVADI", "KADAPA",
    "SAMBALPUR", "BILASPUR", "SATARA", "BIJAPUR", "RAMPUR", "SHIVAMOGGA", "CHANDRAPUR",
    "JUNAGADH", "THRISSUR", "ALWAR", "BARDHAMAN", "KAKINADA", "NIZAMABAD", "PARBHANI",
    "TUMKUR", "KHAMMAM", "PANIPAT", "DARBHANGA", "KARNAL", "BATHINDA", "JALNA",
    "ELURU", "BARABANKI", "PURNIA", "SATNA", "MAU", "SONIPAT", "FARRUKHABAD", "SAGAR",
    "DURG", "IMPHAL", "RATLAM", "HAPUR", "ARRAH", "KARIMNAGAR", "ANANTAPUR", "ETAWAH",
    "AMBERNATH", "BHARATPUR", "BEGUSARAI", "GANDHINAGAR", "PUDUCHERRY", "PONDICHERRY",
    "SIKAR", "THOOTHUKUDI", "TUTICORIN", "REWA", "MIRZAPUR", "RAICHUR", "PALI",
    "HARIDWAR", "KATIHAR", "NAGERCOIL", "THANJAVUR", "BULANDSHAHR", "KATNI",
    "SAMBHAL", "SINGRAULI", "NADIAD", "SECUNDERABAD", "YAMUNANAGAR", "PANCHKULA",
    "BURHANPUR", "KHARAGPUR", "DINDIGUL", "GANDHIDHAM", "HOSPET", "AMBALA", "MEHSANA",
    "JORHAT", "MANSA", "SILCHAR", "TEZPUR", "SHIMLA", "MANALI", "GANGTOK", "AIZAWL",
    "KOHIMA", "ITANAGAR", "DISPUR", "PANAJI", "PANJIM", "DAMAN", "DIU", "SILVASSA",
    "LEH", "KARGIL", "PORT BLAIR", "KAVARATTI", "MOGAPPAIR",
}

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
def layer1_structural(addr: str, tokens, min_words: int = 5, merge_len_threshold: int = 15):
    issues = []
    if re.search(r",\s*,", addr):
        issues.append("DOUBLE_COMMA_EMPTY_FIELD")
    if re.search(r"(\d{6})\1", addr):
        issues.append("PINCODE_DUPLICATED")
    glued_match = re.search(r"[A-Za-z](\d{6})\b", addr)
    if glued_match and "PINCODE_DUPLICATED" not in issues:
        issues.append("PINCODE_GLUED_TO_TEXT")

    state_found = any(state in addr.upper() for state in INDIAN_STATES)
    city_found = any(city in addr.upper() for city in INDIAN_CITY_HINTS)
    if not state_found and not city_found:
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
# Layer 2 helpers - pincode extraction only (the network call itself lives
# in pincode_lookup.py)
# ----------------------------------------------------------------------
_PIN_PATTERN = re.compile(r"\b(\d{3})[ -](\d{3})\b|\b(\d{6})\b")


def extract_pins(addr: str):
    """
    Pulls out 6-digit Indian pincodes. Handles the plain, glued-together
    form (400001) as well as the equally common "split" form some people
    write it in - a space or hyphen between the two halves, e.g.
    "600 037" or "600-037" (both mean pincode 600037). Both normalize to
    the same 6-digit string so Layer 2's pincode lookup treats them
    identically to the plain form.
    """
    pins = set()
    for m in _PIN_PATTERN.finditer(addr):
        pins.add(m.group(3) if m.group(3) else m.group(1) + m.group(2))
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
def analyze_address_local(addr, min_words: int = 5, merge_len_threshold: int = 15):
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


def dedupe(issues):
    seen = set()
    out = []
    for i in issues:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out
