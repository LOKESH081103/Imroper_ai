"""
Layer 2 - Pincode master-data validation.

Uses the free, keyless "All India Pincode API" - static JSON served via
GitHub Pages, CORS-enabled, no server, no rate limit:
    https://aniket-thapa.github.io/india-pincode-api

Underlying data source: Dept. of Posts, via data.gov.in.
License: CC BY-NC 4.0 (non-commercial use, with attribution). If this tool
is ever used in a strictly commercial product, either get written
permission from the API author or swap in the official data.gov.in
All-India Pincode Directory download instead - see README for notes.
"""

import requests

BASE_URL = "https://aniket-thapa.github.io/india-pincode-api"

# In-memory cache so repeated pincodes in the same run only hit the network once.
_cache = {}


def lookup_pincode(pincode: str, timeout: int = 6):
    """
    Look up a 6-digit Indian pincode.

    Returns:
        dict  -> {"state": ..., "district": ..., "offices": [...]} if found
        None  -> pincode does not exist in the dataset (likely fake/foreign)
        "ERROR" -> network/API problem; caller should treat this as
                   "couldn't verify" rather than "invalid"
    """
    if pincode in _cache:
        return _cache[pincode]

    try:
        resp = requests.get(f"{BASE_URL}/pincodes/{pincode}.json", timeout=timeout)
        if resp.status_code == 404:
            _cache[pincode] = None
            return None
        resp.raise_for_status()
        data = resp.json()
        _cache[pincode] = data
        return data
    except Exception:
        # Network unavailable, API down, timeout, etc. Don't punish the
        # address for our own connectivity problem.
        return "ERROR"


def clear_cache():
    _cache.clear()
