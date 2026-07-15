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

Robustness & scale notes
-------------------------
Real-world networks (corporate proxies, VPNs, patchy wifi) are unreliable
in ways that show up as "some rows couldn't be verified" if the client is
naive - and calling the API once per ROW is a non-starter for large files:
India has ~19,000 unique pincodes total, so a 1-lakh-row file has at most
~19,000 distinct pincodes to actually look up no matter how many rows
share them. This module is built around that:
  - `lookup_pincodes_bulk()` dedupes the pincodes first, then fetches the
    unique ones concurrently with a thread pool (network I/O, not CPU-bound,
    so threads are the right tool). This turns "100,000 sequential round
    trips" into "~19,000 deduped lookups split across N workers".
  - A single `requests.Session` is reused so TLS/TCP connections are kept
    alive instead of renegotiating per pincode.
  - Each request gets 2 automatic retries (3 attempts total) with backoff
    for connection hiccups and 5xx responses.
  - A thread-safe circuit breaker: after a few consecutive network-level
    failures (not 404s - those are legitimate "pincode doesn't exist"
    answers), it's almost always a local connectivity problem, not this
    specific pincode. Rather than making the batch wait out a timeout on
    every remaining pincode, the client "opens" and fails fast for a short
    cooldown, then quietly tries again.
  - `check_connectivity()` gives a one-shot, specific diagnosis (DNS/
    firewall/timeout/etc.) the UI can surface on demand instead of a vague
    "no internet or API down".
  - A stall watchdog on the bulk path: `timeout=` on a `requests` call does
    NOT reliably bound DNS resolution on every platform/proxy setup, so a
    handful of pincodes can occasionally hang past their nominal timeout.
    Waiting on ALL futures (the naive `as_completed()` usage) means even
    one such straggler blocks the entire batch forever - and even worse,
    exiting a `with ThreadPoolExecutor() as executor:` block calls
    `shutdown(wait=True)`, which blocks again on the same stuck threads.
    `lookup_pincodes_bulk()` instead polls with a bounded per-round timeout;
    if a full round passes with zero new completions, everything still
    pending is marked "ERROR" and the executor is shut down with
    `wait=False` - the stuck thread(s) are abandoned in the background
    rather than allowed to freeze the whole run.
"""

import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FutureTimeoutError

import requests
from requests.adapters import HTTPAdapter

try:
    from urllib3.util.retry import Retry
    _HAS_RETRY = True
except ImportError:  # pragma: no cover - urllib3 is a hard dependency of requests,
    # so this should never actually happen; if it somehow does, degrade to a
    # plain session with no automatic retries rather than reaching for the
    # deprecated `requests.packages.urllib3` alias, which just emits a
    # DeprecationWarning for no benefit - the top-level `urllib3` package
    # IS `requests.packages.urllib3` in every modern requests version.
    _HAS_RETRY = False

BASE_URL = "https://aniket-thapa.github.io/india-pincode-api"

# In-memory cache so repeated pincodes in the same run only hit the network
# once. Deliberately only caches successes / confirmed-not-found results -
# never "ERROR" - so a transient failure gets retried on a later call
# instead of being stuck forever. Shared across threads; protected by
# _cache_lock since lookup_pincodes_bulk() hits it concurrently.
_cache = {}
_cache_lock = threading.Lock()

# ---- circuit breaker state (thread-safe) ----
_CIRCUIT_FAIL_THRESHOLD = 3
_CIRCUIT_COOLDOWN_SECONDS = 20
_consecutive_failures = 0
_circuit_opened_at = None
_circuit_lock = threading.Lock()

DEFAULT_BULK_WORKERS = 40


def _build_session() -> requests.Session:
    session = requests.Session()
    max_retries = 0
    if _HAS_RETRY:
        try:
            max_retries = Retry(
                total=2,                 # up to 2 retries (3 attempts total) per call
                backoff_factor=0.5,      # ~0.5s, 1s between retries
                status_forcelist=[500, 502, 503, 504],
                allowed_methods=["GET"],
                raise_on_status=False,
            )
        except TypeError:
            # urllib3 < 1.26 used `method_whitelist` instead of `allowed_methods`
            max_retries = Retry(
                total=2,
                backoff_factor=0.5,
                status_forcelist=[500, 502, 503, 504],
                method_whitelist=["GET"],
                raise_on_status=False,
            )
    # Pool sized for concurrent bulk lookups, not just one-at-a-time calls.
    adapter = HTTPAdapter(max_retries=max_retries, pool_connections=DEFAULT_BULK_WORKERS,
                           pool_maxsize=DEFAULT_BULK_WORKERS)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({"User-Agent": "agreement-address-checker/1.0 (+data-quality-tool)"})
    return session


_session = _build_session()


def _circuit_is_open() -> bool:
    global _circuit_opened_at
    with _circuit_lock:
        if _circuit_opened_at is None:
            return False
        if time.time() - _circuit_opened_at > _CIRCUIT_COOLDOWN_SECONDS:
            _circuit_opened_at = None  # cooldown elapsed, allow one more attempt
            return False
        return True


def _record_success():
    global _consecutive_failures, _circuit_opened_at
    with _circuit_lock:
        _consecutive_failures = 0
        _circuit_opened_at = None


def _record_failure():
    global _consecutive_failures, _circuit_opened_at
    with _circuit_lock:
        _consecutive_failures += 1
        if _consecutive_failures >= _CIRCUIT_FAIL_THRESHOLD and _circuit_opened_at is None:
            _circuit_opened_at = time.time()


def circuit_status() -> dict:
    """For UI diagnostics: is the circuit currently open, and why."""
    with _circuit_lock:
        cooldown_left = 0.0
        if _circuit_opened_at:
            cooldown_left = max(0.0, _CIRCUIT_COOLDOWN_SECONDS - (time.time() - _circuit_opened_at))
        return {
            "open": _circuit_is_open(),
            "consecutive_failures": _consecutive_failures,
            "cooldown_seconds_left": round(cooldown_left, 1),
        }


def reset_circuit():
    """Manually clear the circuit breaker, e.g. when the user hits 'Retry'."""
    global _consecutive_failures, _circuit_opened_at
    with _circuit_lock:
        _consecutive_failures = 0
        _circuit_opened_at = None


def check_connectivity(timeout: int = 6) -> dict:
    """
    One-shot diagnostic the UI can trigger on demand (a 'Test connection'
    button) - looks up a known-good pincode and reports specifically what
    happened, instead of a generic 'no internet or API down'.
    """
    test_pin = "110001"
    start = time.time()
    try:
        resp = _session.get(f"{BASE_URL}/pincodes/{test_pin}.json", timeout=(4, timeout))
        elapsed = time.time() - start
        if resp.status_code == 200:
            return {"ok": True, "message": f"Reached the pincode API successfully ({elapsed:.1f}s response time).",
                     "elapsed": elapsed}
        return {"ok": False, "message": f"API responded with HTTP {resp.status_code} for a known-good pincode.",
                 "elapsed": elapsed}
    except requests.exceptions.SSLError as e:
        return {"ok": False,
                 "message": f"TLS/SSL error - a proxy or firewall may be intercepting HTTPS traffic. ({e})",
                 "elapsed": time.time() - start}
    except requests.exceptions.ConnectTimeout:
        return {"ok": False,
                 "message": "Connection timed out reaching aniket-thapa.github.io - it may be blocked by a "
                             "firewall/VPN/corporate proxy, or there's no internet access from this machine.",
                 "elapsed": time.time() - start}
    except requests.exceptions.ConnectionError as e:
        return {"ok": False,
                 "message": f"Could not connect at all - check internet access, VPN, or whether your network "
                             f"blocks GitHub Pages (aniket-thapa.github.io). ({e})",
                 "elapsed": time.time() - start}
    except requests.exceptions.Timeout:
        return {"ok": False, "message": "Request timed out - network may be slow or the API briefly unresponsive.",
                 "elapsed": time.time() - start}
    except Exception as e:
        return {"ok": False, "message": f"Unexpected error: {e}", "elapsed": time.time() - start}


def _fetch_one(pincode: str, timeout: int):
    """Single network fetch, no cache/circuit bookkeeping other than recording result."""
    try:
        resp = _session.get(f"{BASE_URL}/pincodes/{pincode}.json", timeout=(4, timeout))
        if resp.status_code == 404:
            _record_success()
            return pincode, None
        resp.raise_for_status()
        data = resp.json()
        _record_success()
        return pincode, data
    except Exception:
        _record_failure()
        return pincode, "ERROR"


def lookup_pincode(pincode: str, timeout: int = 8):
    """
    Look up a single 6-digit Indian pincode (used outside the bulk path,
    e.g. connectivity tests or one-off lookups).

    Returns:
        dict  -> {"state": ..., "district": ..., "offices": [...]} if found
        None  -> pincode does not exist in the dataset (likely fake/foreign)
        "ERROR" -> network/API problem (including a fast-fail from the
                   circuit breaker); caller should treat this as "couldn't
                   verify" rather than "invalid"
    """
    with _cache_lock:
        if pincode in _cache:
            return _cache[pincode]

    if _circuit_is_open():
        return "ERROR"

    _, result = _fetch_one(pincode, timeout)
    if result != "ERROR":
        with _cache_lock:
            _cache[pincode] = result
    return result


STALL_TIMEOUT_SECONDS = 25  # if a full round passes with zero completions, stop waiting


def lookup_pincodes_bulk(pincodes, timeout: int = 8, max_workers: int = DEFAULT_BULK_WORKERS,
                          progress_callback=None) -> dict:
    """
    Look up many pincodes concurrently. This is the path large files should
    use: dedupe your pincodes down to the unique set first (there are only
    ~19,000 possible in all of India), then call this once instead of
    calling lookup_pincode() per row.

    progress_callback, if given, is called as progress_callback(done, total)
    after each pincode resolves (cached or fetched), so the UI can show real
    progress instead of an indefinite spinner.

    This function is guaranteed to return - it never waits indefinitely.
    `timeout=` on the underlying request does not reliably bound DNS
    resolution on every platform/proxy, so a small number of lookups can
    occasionally hang past their nominal timeout. Waiting on every future
    (as a plain `as_completed()` loop does) means even one such straggler
    blocks the whole batch forever. Here, if a full STALL_TIMEOUT_SECONDS
    window passes with zero new completions, everything still outstanding
    is marked "ERROR" and abandoned - the executor is shut down with
    `wait=False` so any genuinely stuck thread doesn't block the return.

    Returns: {pincode: dict | None | "ERROR"} for every pincode requested.
    """
    unique_pins = sorted(set(pincodes))
    results = {}
    to_fetch = []

    with _cache_lock:
        for pin in unique_pins:
            if pin in _cache:
                results[pin] = _cache[pin]
            else:
                to_fetch.append(pin)

    total = len(unique_pins)
    done = len(results)
    if progress_callback and done:
        progress_callback(done, total)

    if not to_fetch:
        return results

    executor = ThreadPoolExecutor(max_workers=max_workers)
    try:
        futures = {}
        for pin in to_fetch:
            if _circuit_is_open():
                # Fail fast for everything still queued once the breaker
                # trips, rather than spinning up threads that will just
                # time out one by one.
                results[pin] = "ERROR"
                done += 1
                if progress_callback:
                    progress_callback(done, total)
                continue
            futures[executor.submit(_fetch_one, pin, timeout)] = pin

        pending = set(futures.keys())
        while pending:
            try:
                completed_this_round = list(as_completed(pending, timeout=STALL_TIMEOUT_SECONDS))
            except FutureTimeoutError:
                completed_this_round = [f for f in pending if f.done()]

            if not completed_this_round:
                # Zero progress for a full stall window - one or more
                # requests are stuck past their nominal timeout (typically
                # a DNS/connect hang the requests timeout doesn't cover).
                # Give up on everything still pending rather than wait
                # forever; the underlying thread(s) are abandoned when the
                # executor shuts down below.
                for f in pending:
                    pin = futures[f]
                    results[pin] = "ERROR"
                    _record_failure()
                    done += 1
                    if progress_callback:
                        progress_callback(done, total)
                break

            for f in completed_this_round:
                pin, result = f.result()
                results[pin] = result
                if result != "ERROR":
                    with _cache_lock:
                        _cache[pin] = result
                done += 1
                if progress_callback:
                    progress_callback(done, total)
                pending.discard(f)
    finally:
        # wait=False: never block returning to the caller on a thread that
        # may be stuck forever - let it finish or die in the background.
        try:
            executor.shutdown(wait=False, cancel_futures=True)
        except TypeError:
            # cancel_futures was added in Python 3.9
            executor.shutdown(wait=False)

    return results


def clear_cache():
    with _cache_lock:
        _cache.clear()

