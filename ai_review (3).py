"""
Layer 4 - optional AI semantic judge, using the user's own free-tier
Google Gemini API key. Never required for the tool to work; only called
for rows the caller explicitly sends to it (kept small to respect free
daily/per-minute quotas).
"""

import json
import re

import requests

GEMINI_URL_TMPL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"

PROMPT_TEMPLATE = (
    "You are validating Indian customer postal addresses for a data-quality "
    "audit. Judge ONLY the address given below - do not invent facts about "
    "it. Reply with STRICT JSON only, no markdown fences, no commentary, in "
    "exactly this shape:\n"
    '{{"verdict": "clean" or "issue", "severity": "Critical" or "Warning" or "None", '
    '"category": "short category name, e.g. gibberish / incomplete / foreign_location / '
    'merged_words / placeholder / plausible", "reason": "under 20 words"}}\n\n'
    "Address: {address}"
)


def gemini_check(address: str, api_key: str, model: str = "gemini-2.0-flash", timeout: int = 20):
    """Send one address to Gemini and return a parsed dict result.

    On any failure, returns a dict with verdict="error" so callers can
    display it without crashing the batch.
    """
    url = GEMINI_URL_TMPL.format(model=model, key=api_key)
    payload = {"contents": [{"parts": [{"text": PROMPT_TEMPLATE.format(address=address)}]}]}

    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
        text = re.sub(r"^```(json)?", "", text.strip())
        text = re.sub(r"```$", "", text.strip())
        parsed = json.loads(text.strip())
        # Normalize keys defensively in case the model drifts slightly
        parsed.setdefault("verdict", "issue")
        parsed.setdefault("severity", "Warning")
        parsed.setdefault("category", "unspecified")
        parsed.setdefault("reason", "")
        return parsed
    except Exception as e:
        return {"verdict": "error", "severity": "None", "category": "AI_ERROR", "reason": str(e)}
