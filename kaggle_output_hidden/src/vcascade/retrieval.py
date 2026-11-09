"""Evidence retrieval for the retrieval-grounded verifier (V3).

Uses the public Wikipedia API: no index to download, works from a Kaggle /
Colab notebook with internet enabled. The search string is the question plus
the SLM's candidate answer, so the passages bear on the claim being checked.

Wikimedia allows 200 requests/minute to a client whose User-Agent carries contact
info (set WIKI_CONTACT) and at most 3 concurrent requests. We use 3 workers behind
a shared limiter of 3 requests/second, retry 429s with backoff (honouring
Retry-After), and append progress to a resume file so an interrupted run picks up
where it stopped.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

API = "https://en.wikipedia.org/w/api.php"
# Wikimedia's User-Agent policy wants contact info (an email or a URL); without it requests
# fall in the "unidentified" tier, limited to 10 per minute instead of 200.
CONTACT = os.environ.get("WIKI_CONTACT", "")
HEADERS = {"User-Agent": f"vcascade-research-bot/0.1 ({CONTACT}) python-requests" if CONTACT
           else "vcascade-research-bot/0.1 python-requests"}


_rate_lock = threading.Lock()
_last_request = [0.0]
MIN_INTERVAL = 1 / 3  # seconds between requests across all workers: 180/min, under the 200/min limit


def _throttle():
    with _rate_lock:
        wait = _last_request[0] + MIN_INTERVAL - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _last_request[0] = time.monotonic()


def _get(params: dict, retries: int = 8) -> dict:
    for attempt in range(retries):
        _throttle()
        try:
            r = requests.get(API, params={**params, "format": "json"}, headers=HEADERS, timeout=20)
            if r.status_code == 429 or r.status_code >= 500:
                wait = float(r.headers.get("Retry-After", 0) or 0) or min(2 ** attempt * 2, 120)
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            time.sleep(min(2 ** attempt * 2, 120))
    raise RuntimeError(f"Wikipedia API kept failing after {retries} attempts")


def clean_query(text: str, max_words: int = 30) -> str:
    """Drop quotes and search-operator punctuation; long or quoted queries otherwise match nothing."""
    text = re.sub(r"[\"“”‘’'`?!:;()\[\]{}<>|~*^\/=+#]", " ", text)
    return " ".join(text.split()[:max_words])


_STOP = set("""a an the of in on at to for from by with and or but is are was were be been being which what who
whom whose when where why how that this these those it its as into than then there their they he she his her
him we you your i do does did has have had not no all any each some such can could would should will may might
also especially traditionally often usually known called name named""".split())


def keywords(text: str, max_words: int = 10) -> str:
    words = re.sub(r"[,.]", " ", clean_query(text, 60)).split()
    return " ".join([w for w in words if w.lower() not in _STOP][:max_words])


def search_passages(query: str, k: int = 3, chars: int = 700, fallback: str | None = None) -> list[dict]:
    """Top-k search hits with their intro extracts, one request per attempt.

    Tries the query, then `fallback` (the question alone), then the fallback's keywords:
    Wikipedia search requires most terms to match, so long questions can find nothing.
    """
    attempts = [clean_query(query)] + ([clean_query(fallback), keywords(fallback)] if fallback else [])
    for q in dict.fromkeys(a for a in attempts if a):  # dedupe, keep order
        out = _search(q, k, chars)
        if out:
            return out
    return []


def _search(query: str, k: int, chars: int) -> list[dict]:
    if not query:
        return []
    data = _get({"action": "query", "generator": "search", "gsrsearch": query[:300], "gsrlimit": k,
                 "prop": "extracts", "exintro": 1, "explaintext": 1, "exlimit": k})
    pages = sorted(data.get("query", {}).get("pages", {}).values(), key=lambda p: p.get("index", 0))
    out = []
    for p in pages:
        text = " ".join((p.get("extract") or "").split())
        if text:
            out.append({"title": p.get("title", ""), "text": text[:chars]})
    return out


def format_context(passages: list[dict]) -> str:
    if not passages:
        return "(no relevant passages found)"
    return "\n\n".join(f"[{i + 1}] {p['title']}: {p['text']}" for i, p in enumerate(passages))


def retrieve_all(queries: list[str], k: int = 3, resume_file: str | Path | None = None,
                 max_fail_frac: float = 0.02, fallbacks: list[str] | None = None,
                 workers: int = 3) -> list[list[dict]]:
    done: dict[str, list] = {}
    if resume_file and Path(resume_file).exists():
        with open(resume_file, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                done[r["search"]] = r["passages"]
    todo = [i for i, q in enumerate(queries) if q not in done]
    lock = threading.Lock()
    state = {"fails": 0, "n": 0}
    log = open(resume_file, "a", encoding="utf-8") if resume_file else None

    def work(i):
        q = queries[i]
        try:
            res = search_passages(q, k=k, fallback=fallbacks[i] if fallbacks else None)
        except RuntimeError:
            with lock:
                state["fails"] += 1
                if state["fails"] > max_fail_frac * len(queries):
                    raise
            res = []  # V3 then sees "(no relevant passages found)"
        with lock:
            done[q] = res
            if log:
                log.write(json.dumps({"search": q, "passages": res}, ensure_ascii=False) + "\n")
                log.flush()
            state["n"] += 1
            if state["n"] % 200 == 0:
                print(f"retrieved {state['n']}/{len(todo)} ({state['fails']} failed)", flush=True)

    try:
        with ThreadPoolExecutor(workers) as pool:
            list(pool.map(work, todo))
    finally:
        if log:
            log.close()
    return [done[q] for q in queries]
