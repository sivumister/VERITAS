import re


# =========================================================
# BASIC TEXT CLEANING
# =========================================================

STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were",
    "to", "of", "in", "on", "for", "and", "or",
    "that", "this", "it", "with", "as", "by",
    "from", "has", "have", "had", "be", "been"
}


def clean_words(text):

    words = re.findall(
        r"[a-zA-Z0-9]+",
        text.lower()
    )

    return {
        word
        for word in words
        if word not in STOPWORDS
    }


# =========================================================
# RELEVANCE SCORE
# =========================================================

def calculate_relevance(query, reviewed_claim):

    query_words = clean_words(query)

    claim_words = clean_words(reviewed_claim)

    if not query_words or not claim_words:
        return 0.0

    matching_words = (
        query_words
        & claim_words
    )

    # We care about how much of the USER QUERY
    # appears in the reviewed claim.
    score = (
        len(matching_words)
        / len(query_words)
    )

    return round(
        score,
        2
    )


def relevance_label(score):

    if score >= 0.60:
        return "HIGH"

    elif score >= 0.30:
        return "MEDIUM"

    else:
        return "LOW"


# =========================================================
# INTERPRET FACT-CHECK RATING
# =========================================================

def interpret_rating(rating):

    if not rating:
        return "UNCLEAR"

    rating_lower = rating.lower().strip()


    # -----------------------------------------------------
    # AMBIGUOUS / MIXED VERDICTS
    # Check these FIRST.
    # -----------------------------------------------------

    unclear_signals = [

        "unclear",
        "partly",
        "partially",
        "mixed",
        "half true",
        "half-true",
        "needs context",
        "missing context",
        "not enough information",
        "cannot be verified",
        "cannot verify",
        "unproven"
    ]


    for signal in unclear_signals:

        if signal in rating_lower:
            return "UNCLEAR"


    # -----------------------------------------------------
    # FALSE / NEGATIVE VERDICTS
    # -----------------------------------------------------

    false_signals = [

        "mostly false",
        "completely false",
        "false",
        "incorrect",
        "misleading",
        "not true",
        "untrue",
        "fabricated",
        "fake",
        "unsupported",
        "no evidence",
        "without evidence",
        "baseless",
        "inaccurate",
        "did not happen",
        "didn't happen"
    ]


    for signal in false_signals:

        if signal in rating_lower:
            return "FALSE"


    # -----------------------------------------------------
    # TRUE / POSITIVE VERDICTS
    #
    # Be stricter here because words such as "true"
    # can appear inside a longer mixed explanation.
    # -----------------------------------------------------

    exact_true_ratings = {

        "true",
        "correct",
        "accurate",
        "mostly true",
        "confirmed"
    }


    if rating_lower in exact_true_ratings:

        return "TRUE"


    strong_true_signals = [

        "this is correct",
        "this is true",
        "claim is correct",
        "claim is true",
        "we found this to be true",
        "evidence supports the claim"
    ]


    for signal in strong_true_signals:

        if signal in rating_lower:
            return "TRUE"


    # If we cannot confidently understand the rating,
    # do not force it into TRUE or FALSE.
    return "UNCLEAR"

# =========================================================
# RANK FACT-CHECK RESULTS
# =========================================================

def rank_fact_checks(query, factcheck_results):

    ranked = []


    for result in factcheck_results:

        score = calculate_relevance(
            query,
            result.get(
                "claim",
                ""
            )
        )


        enriched_result = (
            result.copy()
        )


        enriched_result[
            "relevance_score"
        ] = score


        enriched_result[
            "relevance"
        ] = relevance_label(
            score
        )


        enriched_result[
            "interpreted_rating"
        ] = interpret_rating(
            result.get(
                "rating"
            )
        )


        ranked.append(
            enriched_result
        )


    ranked.sort(
        key=lambda item:
            item["relevance_score"],
        reverse=True
    )


    return ranked


# =========================================================
# COMPARE AI WITH FACT-CHECK
# =========================================================

def determine_evidence_relationship(
    ai_prediction,
    factcheck_results
):

    if not factcheck_results:

        return {
            "status":
                "NO_MATCH",

            "message":
                "No matching published fact-check was found."
        }


    # Use best-ranked fact-check
    best = factcheck_results[0]


    # Do not make a strong comparison
    # using weakly related evidence.
    if best["relevance"] != "HIGH":

        return {
            "status":
                "NO_CLEAR_VERDICT",

            "message":
                "Related fact-checks were found, but the match was not strong enough for VERITAS to compare them directly with the AI prediction."
        }

    external_rating = best[
        "interpreted_rating"
    ]


    if external_rating == "UNCLEAR":

        return {
            "status":
                "NO_CLEAR_VERDICT",

            "message":
                "A relevant fact-check was found, but its rating could not be interpreted reliably."
        }


    # -----------------------------------------------------
    # AI SAYS FAKE
    # FACT-CHECK SAYS CLAIM FALSE
    # -----------------------------------------------------

    if (
        ai_prediction == "FAKE"
        and external_rating == "FALSE"
    ):

        return {
            "status":
                "AGREEMENT",

            "message":
                "The AI prediction and the most relevant external fact-check point in the same direction."
        }


    # -----------------------------------------------------
    # AI SAYS REAL
    # FACT-CHECK SAYS CLAIM TRUE
    # -----------------------------------------------------

    if (
        ai_prediction == "REAL"
        and external_rating == "TRUE"
    ):

        return {
            "status":
                "AGREEMENT",

            "message":
                "The AI prediction and the most relevant external fact-check point in the same direction."
        }


    # -----------------------------------------------------
    # OTHERWISE THEY DISAGREE
    # -----------------------------------------------------

    return {
        "status":
            "CONFLICT",

        "message":
            "The AI prediction and the most relevant external fact-check do not agree. Review the cited evidence before drawing a conclusion."
    }