"""Google published claim-review lookup with safe errors and a bounded cache."""
from collections import OrderedDict
import copy
import os
from pathlib import Path
from threading import Lock
from time import monotonic

import requests
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")
FACTCHECK_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
_cache = OrderedDict()
_cache_lock = Lock()
_CACHE_SECONDS = 900
_CACHE_LIMIT = 128


class FactCheckUnavailable(ValueError):
    pass


def search_fact_checks(query, max_results=5, timeout=6, refresh=False):
    query = str(query or "").strip()
    if not query:
        raise ValueError("Fact-check query cannot be empty.")
    api_key = os.getenv("GOOGLE_FACTCHECK_API_KEY", "").strip()
    if not api_key:
        raise FactCheckUnavailable("Google Fact Check is not configured.")
    max_results = max(1, min(int(max_results), 10))
    key = (query.casefold(), max_results)
    now = monotonic()
    with _cache_lock:
        cached = _cache.get(key)
        if cached and not refresh and now - cached[0] < _CACHE_SECONDS:
            _cache.move_to_end(key)
            return copy.deepcopy(cached[1])
    try:
        response = requests.get(FACTCHECK_URL,
                                params={"query": query, "languageCode": "en",
                                        "pageSize": max_results, "key": api_key},
                                timeout=(3, timeout))
        if response.status_code in (401, 403):
            raise FactCheckUnavailable("Google Fact Check rejected the key or API permissions.")
        if response.status_code == 429:
            raise FactCheckUnavailable("Google Fact Check's request quota was reached.")
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict) or not isinstance(data.get("claims", []), list):
            raise FactCheckUnavailable("Google Fact Check returned an unexpected response.")
    except requests.Timeout:
        raise FactCheckUnavailable("Google Fact Check timed out.") from None
    except requests.RequestException:
        # Never include the request URL: Google sends its key in query parameters.
        raise FactCheckUnavailable("Google Fact Check could not be reached.") from None
    except ValueError as error:
        if isinstance(error, FactCheckUnavailable):
            raise
        raise FactCheckUnavailable("Google Fact Check returned an unreadable response.") from None
    results = []
    for claim in data.get("claims", []):
        if not isinstance(claim, dict):
            continue
        for review in claim.get("claimReview", []) or []:
            if not isinstance(review, dict):
                continue
            publisher = review.get("publisher") or {}
            results.append({"claim": claim.get("text", ""), "claimant": claim.get("claimant"),
                            "publisher": publisher.get("name", "Unknown publisher"),
                            "publisher_site": publisher.get("site"),
                            "rating": review.get("textualRating", "No rating"),
                            "title": review.get("title", "Fact Check"),
                            "url": review.get("url"), "review_date": review.get("reviewDate")})
    with _cache_lock:
        _cache[key] = (monotonic(), copy.deepcopy(results))
        _cache.move_to_end(key)
        while len(_cache) > _CACHE_LIMIT:
            _cache.popitem(last=False)
    return results
