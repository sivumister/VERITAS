# =========================================================
# VERITAS OVERALL ASSESSMENT SERVICE
# =========================================================


def build_overall_assessment(
    prediction,
    prediction_strength,
    evidence_relationship,
    factcheck_results=None
):
    """
    Combine the AI assessment and external evidence
    into a user-friendly overall interpretation.

    This function does NOT declare objective truth.

    It explains whether:
    - AI and external evidence agree
    - AI and external evidence conflict
    - external evidence is inconclusive
    - no external evidence was found
    - external verification was unavailable
    """

    prediction = prediction.upper()

    factcheck_results = (
        factcheck_results
        if factcheck_results
        else []
    )

    evidence_relationship = (
        evidence_relationship
        if evidence_relationship
        else {
            "status": "NO_MATCH"
        }
    )

    strength = prediction_strength.get(
        "strength",
        "UNKNOWN"
    )

    evidence_status = evidence_relationship.get(
        "status",
        "NO_MATCH"
    )


    # =====================================================
    # AI LABEL
    # =====================================================

    if prediction == "FAKE":

        ai_label = "Potentially Fake"

    else:

        ai_label = "Potentially Credible"


    # =====================================================
    # BEST FACT-CHECK
    # =====================================================

    best_factcheck = (
        factcheck_results[0]
        if factcheck_results
        else None
    )

    external_rating = None
    publisher = None

    if best_factcheck:

        external_rating = best_factcheck.get(
            "interpreted_rating"
        )

        publisher = best_factcheck.get(
            "publisher"
        )


    # =====================================================
    # AGREEMENT
    # =====================================================

    if evidence_status == "AGREEMENT":

        message = (
            f"The AI model assessed the content as "
            f"{ai_label} with {strength.lower()} prediction "
            f"strength. A highly relevant published "
            f"fact-check points in the same direction."
        )

        if publisher:

            message += (
                f" The most relevant evidence was "
                f"published by {publisher}."
            )

        return {

            "status":
                "ALIGNED",

            "label":
                ai_label,

            "headline":
                "AI and External Evidence Align",

            "message":
                message,

            "guidance":
                (
                    "The agreement increases support for the "
                    "assessment, but it should still not be "
                    "treated as absolute proof."
                ),

            "external_rating":
                external_rating
        }


    # =====================================================
    # CONFLICT
    # =====================================================

    if evidence_status == "CONFLICT":

        message = (
            f"The AI model assessed the content as "
            f"{ai_label}, but a highly relevant external "
            f"fact-check points in the opposite direction."
        )

        if external_rating:

            message += (
                f" The external fact-check was interpreted "
                f"as {external_rating}."
            )

        if publisher:

            message += (
                f" The review was published by "
                f"{publisher}."
            )

        return {

            "status":
                "CONFLICT",

            "label":
                "Conflicting Evidence",

            "headline":
                "AI and External Evidence Disagree",

            "message":
                message,

            "guidance":
                (
                    "VERITAS should not present either result "
                    "as definitive in this situation. Review "
                    "the cited fact-check and original source "
                    "before drawing a conclusion."
                ),

            "external_rating":
                external_rating
        }


    # =====================================================
    # NO CLEAR VERDICT
    # =====================================================

    if evidence_status == "NO_CLEAR_VERDICT":

        return {

            "status":
                "INCONCLUSIVE",

            "label":
                ai_label,

            "headline":
                "External Evidence Is Inconclusive",

            "message":
                (
                    f"The AI model assessed the content as "
                    f"{ai_label} with {strength.lower()} "
                    f"prediction strength. Related external "
                    f"evidence was found, but VERITAS could "
                    f"not interpret it strongly enough to "
                    f"confirm or challenge the AI result."
                ),

            "guidance":
                (
                    "Treat the AI result as an indication "
                    "rather than a verified conclusion."
                ),

            "external_rating":
                external_rating
        }


    # =====================================================
    # EXTERNAL SERVICE UNAVAILABLE
    # =====================================================

    if evidence_status == "UNAVAILABLE":

        return {

            "status":
                "AI_ONLY",

            "label":
                ai_label,

            "headline":
                "AI-Only Assessment",

            "message":
                (
                    f"The AI model assessed the content as "
                    f"{ai_label} with {strength.lower()} "
                    f"prediction strength. External "
                    f"fact-check verification was unavailable."
                ),

            "guidance":
                (
                    "The result currently represents only "
                    "the AI model's assessment."
                ),

            "external_rating":
                None
        }


    # =====================================================
    # NO MATCH
    # =====================================================

    return {

        "status":
            "AI_ONLY",

        "label":
            ai_label,

        "headline":
            "No Matching External Fact-Check",

        "message":
            (
                f"The AI model assessed the content as "
                f"{ai_label} with {strength.lower()} "
                f"prediction strength, but no sufficiently "
                f"relevant published fact-check was found."
            ),

        "guidance":
            (
                "The absence of a matching fact-check does "
                "not mean the content is true or false."
            ),

        "external_rating":
            None
    }