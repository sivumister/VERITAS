"""On-demand Google claim reviews + Tavily search/extraction.

Web pages are evidence candidates. Search scores/keyword overlap never become
truth labels. Only strictly matched, clearly rated Google claim reviews may
establish a selected-claim review outcome. No extra language model is loaded.
"""
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import copy
from datetime import datetime, timezone
import hashlib
import os
import re
from threading import Lock
from time import monotonic
from urllib.parse import urlsplit, urlunsplit

import requests
from services.evidence_service import (
    clean_words, normalize_claim, safe_public_url, rank_fact_checks,
    determine_evidence_relationship
)
from services.factcheck_service import search_fact_checks, FactCheckUnavailable

CACHE_SECONDS = 900
_CACHE_LIMIT = 64
_cache = OrderedDict()
_state_lock = Lock()
_running_users = set()


class EvidenceBusy(RuntimeError):
    pass


class WebEvidenceUnavailable(ValueError):
    pass


def claim_candidates(text, input_type="article", title=None):
    """Suggest bounded search statements; not a semantic claim-extraction model."""
    text = str(text or "").strip()
    candidates = []
    seen = set()
    def add(value, kind):
        value = re.sub(r"\s+", " ", value).strip()
        key = normalize_claim(value)
        if value and len(value) <= 500 and key not in seen:
            candidates.append({"text": value, "kind": kind})
            seen.add(key)
    if input_type in ("claim", "headline"):
        add(text, "submitted_claim")
    else:
        if title:
            add(title, "headline")
        sentences = [value.strip() for value in re.split(r"(?<=[.!?])\s+|\n+", text)
                     if 4 <= len(value.split()) <= 80 and len(value.strip()) <= 500
                     and not value.strip().endswith("?")]
        if sentences:
            add(sentences[0], "sentence")
        # Prefer concrete statements (numbers/names); do not infer their truth.
        indexed = list(enumerate(sentences))
        indexed.sort(key=lambda pair: (bool(re.search(r"\d", pair[1])),
                                     len(re.findall(r"\b[A-Z][a-z]+\b", pair[1])),
                                     -pair[0]), reverse=True)
        for _, sentence in indexed:
            if len(candidates) >= 3:
                break
            add(sentence, "sentence")
    if not candidates and text:
        add(" ".join(text.split()[:30])[:500], "search_fragment")
    return candidates[:3]


def canonical_url(url):
    if not safe_public_url(url):
        return None
    p = urlsplit(url)
    # Discard query/fragment for conservative duplicate/original-page exclusion.
    return urlunsplit((p.scheme.lower(), p.netloc.lower(), p.path.rstrip("/"), "", ""))


def _post_tavily(endpoint, payload, read_timeout=8):
    api_key = os.getenv("TAVILY_API_KEY", "").strip()
    if not api_key:
        raise WebEvidenceUnavailable("Web evidence search is not configured.")
    try:
        response = requests.post("https://api.tavily.com/" + endpoint,
                                 headers={"Authorization": "Bearer " + api_key,
                                          "Content-Type": "application/json"},
                                 json=payload, timeout=(3, read_timeout))
        if response.status_code in (401, 403):
            raise WebEvidenceUnavailable("Tavily rejected the API key or account permissions.")
        if response.status_code in (429, 432, 433):
            raise WebEvidenceUnavailable("Tavily's request or credit limit was reached.")
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict) or not isinstance(data.get("results"), list):
            raise WebEvidenceUnavailable("Tavily returned an unexpected response.")
        return data
    except requests.Timeout:
        raise WebEvidenceUnavailable("Tavily timed out. Please retry the evidence search.") from None
    except requests.RequestException:
        raise WebEvidenceUnavailable("Tavily could not be reached.") from None
    except ValueError as error:
        if isinstance(error, WebEvidenceUnavailable):
            raise
        raise WebEvidenceUnavailable("Tavily returned an unreadable response.") from None


def _passage(content, claim):
    # Preserve source wording and order; this ranking is only for choosing excerpts.
    content = str(content or "")[:18000]
    sentences = [x.strip() for x in re.split(r"(?<=[.!?])\s+|\n+", content) if x.strip()]
    query_words = clean_words(claim)
    ranks = sorted(range(len(sentences)),
                   key=lambda i: len(query_words & clean_words(sentences[i])), reverse=True)[:3]
    if not ranks:
        return ""
    return " […] ".join(sentences[i] for i in sorted(ranks))[:1400]


def _web_sources(claim, source_url=None):
    payload = {"query": claim, "search_depth": "basic", "max_results": 5,
               "chunks_per_source": 3, "topic": "general",
               "include_answer": False, "include_raw_content": False,
               "include_published_date": True, "auto_parameters": False}
    domains = [x.strip() for x in os.getenv("WEB_EVIDENCE_DOMAINS", "").split(",") if x.strip()]
    if domains:
        payload["include_domains"] = domains[:20]
    result = _post_tavily("search", payload)
    original_url = canonical_url(source_url)
    sources, seen_urls, seen_domains = [], set(), set()
    # Keep provider ranking. Retain at most one URL per publisher domain.
    for item in result["results"]:
        if not isinstance(item, dict):
            continue
        url = safe_public_url(item.get("url"))
        normalized = canonical_url(url)
        domain = (urlsplit(url).hostname or "").lower().removeprefix("www.") if url else None
        if (not url or normalized == original_url or normalized in seen_urls
                or domain in seen_domains):
            continue
        seen_urls.add(normalized); seen_domains.add(domain)
        sources.append({"title": str(item.get("title") or "Web source")[:300],
                        "url": url, "publisher_site": domain,
                        "published_date": str(item.get("published_date") or "")[:80],
                        "passage": _passage(item.get("content"), claim),
                        "passage_type": "search_excerpt", "stance": "NOT_ASSESSED"})
        if len(sources) == 3:
            break
    extraction_warning = None
    if sources:
        try:
            extracted = _post_tavily("extract", {
                "urls": [source["url"] for source in sources], "query": claim,
                "chunks_per_source": 3, "extract_depth": "basic", "format": "text",
                "timeout": 5, "include_images": False}, read_timeout=7)
            by_url = {canonical_url(item.get("url")): item for item in extracted["results"]
                      if isinstance(item, dict) and safe_public_url(item.get("url"))}
            for source in sources:
                full = by_url.get(canonical_url(source["url"]))
                if full and full.get("raw_content"):
                    source["passage"] = _passage(full["raw_content"], claim)
                    source["passage_type"] = "extracted_passage"
            if any(source["passage_type"] != "extracted_passage" for source in sources):
                extraction_warning = "Some pages could not be extracted. Their search excerpts are shown instead."
        except WebEvidenceUnavailable:
            extraction_warning = "Full-page extraction was unavailable. Search excerpts are shown instead."
    return {"sources": sources, "extraction_warning": extraction_warning,
            "search_completed": True}


def collect_external_evidence(claim, ai_prediction, source_url=None, user_id=None,
                              refresh=False, fragment=False):
    claim = str(claim or "").strip()
    if not claim or len(claim) > 500:
        raise ValueError("Choose a search statement of 1–500 characters.")
    # Include configuration in the cache identity without exposing keys.
    credentials = hashlib.sha256((os.getenv("TAVILY_API_KEY", "") + "|" +
                                 os.getenv("GOOGLE_FACTCHECK_API_KEY", "") + "|" +
                                 os.getenv("WEB_EVIDENCE_DOMAINS", "")).encode()).hexdigest()
    key = (claim, str(ai_prediction).upper(), canonical_url(source_url), fragment, credentials)
    with _state_lock:
        cached = _cache.get(key)
        if cached and not refresh and monotonic() - cached[0] < CACHE_SECONDS:
            _cache.move_to_end(key)
            return dict(copy.deepcopy(cached[1]), cached=True)
        if user_id in _running_users or len(_running_users) >= 4:
            raise EvidenceBusy("An evidence search is already running. Please try again shortly.")
        _running_users.add(user_id)
    try:
        google_results, google_error, tavily_error = [], None, None
        web = {"sources": [], "extraction_warning": None, "search_completed": False}
        google_completed = False
        with ThreadPoolExecutor(max_workers=2) as executor:
            google_job = executor.submit(search_fact_checks, claim, max_results=5, timeout=6, refresh=refresh)
            tavily_job = executor.submit(_web_sources, claim, source_url)
            try:
                google_results = rank_fact_checks(claim, google_job.result())
                google_completed = True
            except FactCheckUnavailable as error:
                google_error = str(error)
            except Exception:
                google_error = "Google Fact Check could not complete the search."
            try:
                web = tavily_job.result()
            except WebEvidenceUnavailable as error:
                tavily_error = str(error)
            except Exception:
                tavily_error = "Tavily could not complete the search."
        relationship = determine_evidence_relationship(ai_prediction, google_results)
        if fragment:
            relationship = {"status": "NO_CLEAR_VERDICT", "external_rating": None,
                            "message": "A text fragment was searched. Review the complete claim before applying a verdict."}
        elif relationship["status"] not in ("AGREEMENT", "CONFLICT"):
            if not google_completed and not web["search_completed"]:
                relationship = {"status": "UNAVAILABLE", "external_rating": None,
                                "message": "Neither external lookup could be completed."}
            elif web["sources"] or google_results:
                relationship = {"status": "NO_CLEAR_VERDICT", "external_rating": None,
                                "message": "Related sources were retrieved, but no sufficiently specific automatic claim verdict was established. Review the cited passages."}
            else:
                relationship = {"status": "NO_MATCH", "external_rating": None,
                                "message": "The completed searches found no usable matching evidence. This is not proof that the claim is true or false."}
        relationship["assessed_claim"] = claim
        result = {"selected_claim": claim, "checked_at": datetime.now(timezone.utc).isoformat(),
                  "sources": web["sources"], "factcheck_results": google_results,
                  "google_error": google_error, "tavily_error": tavily_error,
                  "extraction_warning": web["extraction_warning"],
                  "google_completed": google_completed, "tavily_completed": web["search_completed"],
                  "evidence_relationship": relationship, "cached": False,
                  "scope_note": "This search covers the selected statement, not every claim in the article."}
        # Incomplete provider runs remain retryable; do not cache failures.
        if google_completed and web["search_completed"]:
            with _state_lock:
                _cache[key] = (monotonic(), copy.deepcopy(result))
                _cache.move_to_end(key)
                while len(_cache) > _CACHE_LIMIT:
                    _cache.popitem(last=False)
        return result
    finally:
        with _state_lock:
            _running_users.discard(user_id)
