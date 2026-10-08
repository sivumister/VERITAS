"""Conservative matching of published claim reviews.

Topic similarity ranks candidates only. Automatic use of a review's rating
requires the same normalized claim text, including negation and quantities.
This deliberately abstains on paraphrases instead of claiming semantic matching.
"""
import ipaddress
import re
import unicodedata
from urllib.parse import urlsplit

STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "to", "of", "in", "on",
    "for", "and", "or", "that", "this", "it", "with", "as", "by", "from",
    "has", "have", "had", "be", "been"
}
NEGATIONS = {"not", "no", "never", "cannot", "without", "neither", "nor", "can't", "don't", "doesn't", "isn't", "wasn't", "won't", "didn't"}


def normalize_claim(text):
    text = unicodedata.normalize("NFKC", str(text or "")).casefold()
    text = text.replace("’", "'")
    # Keep word order, negations and numeric punctuation significant.
    return " ".join(re.findall(r"[\w]+(?:[.'/-][\w]+)*|[%$€£¥+=<>°-]", text, flags=re.UNICODE))


def clean_words(text):
    return {word for word in normalize_claim(text).split() if word not in STOPWORDS}


def calculate_relevance(query, reviewed_claim):
    query_words, claim_words = clean_words(query), clean_words(reviewed_claim)
    if not query_words or not claim_words:
        return 0.0
    score = len(query_words & claim_words) / len(query_words)
    query_negation = query_words & NEGATIONS
    claim_negation = claim_words & NEGATIONS
    if query_negation != claim_negation:
        score *= 0.5
    return round(score, 2)


def relevance_label(score):
    return "HIGH" if score >= 0.60 else "MEDIUM" if score >= 0.30 else "LOW"


def same_claim(query, reviewed_claim):
    """Strict identity, not proof of truth and not paraphrase/entailment detection."""
    normalized = normalize_claim(query)
    relative_time = re.search(r"\b(today|tomorrow|yesterday|currently|now|latest|recently|this week|this month|this year|next week)\b", normalized)
    return bool(normalized and not relative_time and len(clean_words(query)) >= 2
                and normalized == normalize_claim(reviewed_claim))


def safe_public_url(url):
    if not isinstance(url, str) or len(url) > 3000:
        return None
    try:
        parts = urlsplit(url)
        hostname = (parts.hostname or "").lower()
        if (parts.scheme not in ("http", "https") or not hostname
                or parts.username or parts.password
                or hostname in ("localhost", "localhost.localdomain")
                or hostname.endswith((".local", ".internal"))):
            return None
        try:
            if not ipaddress.ip_address(hostname).is_global:
                return None
        except ValueError:
            if "." not in hostname:
                return None
        return url
    except ValueError:
        return None


def interpret_rating(rating):
    # Exact rating vocabulary avoids accidentally interpreting negated phrases.
    # Unsupported, unproven, mixed and mostly-true/false ratings remain unresolved.
    if not isinstance(rating, str):
        return "UNCLEAR"
    value = normalize_claim(rating)
    true_ratings = {"true", "correct", "accurate", "confirmed", "this is true",
                    "this is correct", "claim is true", "claim is correct"}
    false_ratings = {"false", "incorrect", "not true", "untrue", "fabricated", "fake",
                     "completely false", "pants on fire", "this is false", "claim is false"}
    if value in true_ratings:
        return "TRUE"
    if value in false_ratings:
        return "FALSE"
    return "UNCLEAR"


def rank_fact_checks(query, factcheck_results):
    ranked = []
    for result in factcheck_results:
        score = calculate_relevance(query, result.get("claim", ""))
        exact = same_claim(query, result.get("claim", ""))
        enriched = dict(result, relevance_score=score, relevance=relevance_label(score),
                        claim_match="EXACT" if exact else "RELATED_ONLY",
                        interpreted_rating=interpret_rating(result.get("rating")),
                        assessed_claim=query,
                        matching_note=("Same normalized claim wording." if exact else
                                       "Related wording only; the review is not automatically applied to this claim."))
        enriched["url"] = safe_public_url(result.get("url"))
        ranked.append(enriched)
    ranked.sort(key=lambda item: (item["claim_match"] == "EXACT", item["relevance_score"]), reverse=True)
    return ranked


def determine_evidence_relationship(ai_prediction, factcheck_results):
    if not factcheck_results:
        return {"status": "NO_MATCH", "message": "No matching published fact-check was found.",
                "external_rating": None}
    # Re-check identity here; never trust a supplied relevance/match label alone.
    usable = [review for review in factcheck_results
              if same_claim(review.get("assessed_claim"), review.get("claim"))
              and safe_public_url(review.get("url"))
              and review.get("interpreted_rating") in ("TRUE", "FALSE")]
    verdicts = {review["interpreted_rating"] for review in usable}
    if len(verdicts) != 1:
        return {"status": "NO_CLEAR_VERDICT", "external_rating": None,
                "message": ("Published reviews disagree about the selected claim." if len(verdicts) > 1 else
                            "Related reviews were found, but their wording or rating is not specific enough for an automatic claim verdict.")}
    verdict = next(iter(verdicts))
    agrees = (ai_prediction.upper() == "REAL" and verdict == "TRUE") or (ai_prediction.upper() == "FAKE" and verdict == "FALSE")
    return {"status": "AGREEMENT" if agrees else "CONFLICT", "external_rating": verdict,
            "assessed_claim": usable[0]["assessed_claim"], "publisher": usable[0].get("publisher"),
            "review_url": usable[0]["url"],
            "message": ("A published review of the same selected claim agrees with the model prediction." if agrees else
                        "A published review of the same selected claim disagrees with the model prediction. Review the cited evidence."),
            "scope": "selected_claim", "matching_method": "normalized_text_identity"}
