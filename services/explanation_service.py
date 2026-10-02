# =========================================================
# VERITAS EXPLANATION SERVICE
# =========================================================

# =========================================================
# VERITAS EXPLANATION SERVICE
# =========================================================


def build_explanation(
    prediction,
    fake_probability,
    real_probability,
    prediction_strength,
    threshold=None,
    evidence_relationship=None,
    factcheck_results=None
):
    """
    Build a user-friendly explanation of a VERITAS prediction.

    This service does NOT determine objective truth.

    It explains:
    1. Why the AI model produced its classification.
    2. How far the result was from the model's decision boundary.
    3. How strong or borderline the AI prediction was.
    4. Whether external fact-check evidence supports,
       conflicts with, or cannot verify the AI result.
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
            "status": "NO_MATCH",
            "message":
                "No matching published fact-check was found."
        }
    )


    # =====================================================
    # DETERMINE DECISION BOUNDARY
    # =====================================================

    if threshold is None:

        decision_boundary = 50.0

    else:

        decision_boundary = float(
            threshold
        )


    # =====================================================
    # MODEL EXPLANATION
    # =====================================================

    if prediction == "FAKE":

        if threshold is None:

            model_reason = (
                f"The AI model assessed the submitted content "
                f"as potentially fake because its Fake score "
                f"was {fake_probability:.2f}%, which was higher "
                f"than its Real score of "
                f"{real_probability:.2f}%."
            )

        else:

            model_reason = (
                f"The AI model assessed the submitted content "
                f"as potentially fake because its Fake score "
                f"was {fake_probability:.2f}%, which crossed "
                f"the model's decision boundary of "
                f"{decision_boundary:.2f}%."
            )


    elif prediction == "REAL":

        if threshold is None:

            model_reason = (
                f"The AI model assessed the submitted content "
                f"as potentially credible because its Real "
                f"score was {real_probability:.2f}%, which was "
                f"higher than its Fake score of "
                f"{fake_probability:.2f}%."
            )

        else:

            model_reason = (
                f"The AI model assessed the submitted content "
                f"as potentially credible because its Fake "
                f"score was {fake_probability:.2f}%, which "
                f"remained below the model's decision boundary "
                f"of {decision_boundary:.2f}%."
            )

    else:

        model_reason = (
            "The AI model returned an unexpected "
            "classification."
        )


    # =====================================================
    # PREDICTION STRENGTH EXPLANATION
    # =====================================================

    strength = prediction_strength.get(
        "strength",
        "UNKNOWN"
    )

    distance = float(
        prediction_strength.get(
            "distance",
            0
        )
    )


    if strength == "LOW":

        strength_reason = (
            f"The prediction strength is LOW because the "
            f"result is only {distance:.2f} percentage points "
            f"from the model's decision boundary. "
            f"This is a borderline AI result and should be "
            f"interpreted cautiously."
        )


    elif strength == "MODERATE":

        strength_reason = (
            f"The prediction strength is MODERATE because the "
            f"result is {distance:.2f} percentage points from "
            f"the model's decision boundary. "
            f"The model shows a clearer classification signal, "
            f"but the result is still not factual verification."
        )


    elif strength == "HIGH":

        strength_reason = (
            f"The prediction strength is HIGH because the "
            f"result is {distance:.2f} percentage points from "
            f"the model's decision boundary. "
            f"The model shows a strong classification signal, "
            f"although this does not by itself prove whether "
            f"the content is true or false."
        )


    else:

        strength_reason = (
            f"The prediction is {distance:.2f} percentage "
            f"points from the model's decision boundary."
        )


    # =====================================================
    # EXTERNAL EVIDENCE EXPLANATION
    # =====================================================

    evidence_status = evidence_relationship.get(
        "status",
        "NO_MATCH"
    )


    if evidence_status == "AGREEMENT":

        if factcheck_results:

            best_result = factcheck_results[0]

            publisher = best_result.get(
                "publisher",
                "an external fact-checking organisation"
            )

            rating = best_result.get(
                "rating",
                "Unknown"
            )

            relevance = best_result.get(
                "relevance",
                "UNKNOWN"
            )

            evidence_reason = (
                f"A {relevance.lower()}-relevance published "
                f"fact-check was found from {publisher}. "
                f"Its rating was \"{rating}\". "
                f"The external evidence and the AI assessment "
                f"point in the same direction."
            )

        else:

            evidence_reason = (
                "External fact-check evidence points in the "
                "same direction as the AI assessment."
            )


    elif evidence_status == "CONFLICT":

        if factcheck_results:

            best_result = factcheck_results[0]

            publisher = best_result.get(
                "publisher",
                "an external fact-checking organisation"
            )

            rating = best_result.get(
                "rating",
                "Unknown"
            )

            evidence_reason = (
                f"A relevant published fact-check from "
                f"{publisher} was found with the rating "
                f"\"{rating}\". However, the external evidence "
                f"points in a different direction from the AI "
                f"assessment."
            )

        else:

            evidence_reason = (
                "Relevant external fact-check evidence was "
                "found, but it points in a different direction "
                "from the AI assessment."
            )


    elif evidence_status == "NO_CLEAR_VERDICT":

        if factcheck_results:

            best_result = factcheck_results[0]

            publisher = best_result.get(
                "publisher",
                "an external fact-checking organisation"
            )

            relevance = best_result.get(
                "relevance",
                "UNKNOWN"
            )

            evidence_reason = (
                f"Related published evidence was found from "
                f"{publisher}, but the strongest match had "
                f"{relevance.lower()} relevance or could not be "
                f"interpreted strongly enough to confirm or "
                f"challenge the AI assessment."
            )

        else:

            evidence_reason = (
                "Related external fact-check information was "
                "found, but VERITAS could not establish a "
                "sufficiently clear relationship between that "
                "evidence and the AI assessment."
            )


    elif evidence_status == "UNAVAILABLE":

        evidence_reason = (
            "The external fact-check service was unavailable, "
            "so this assessment currently relies on the AI "
            "model only."
        )


    else:

        evidence_reason = (
            "No sufficiently relevant published fact-check "
            "was found for this content. This does not mean "
            "that the content is true or false; it means "
            "VERITAS could not find matching external evidence."
        )


    # =====================================================
    # USER-FRIENDLY RESULT LABEL
    # =====================================================

    if prediction == "FAKE":

        result_label = (
            "Potentially Fake"
        )

    else:

        result_label = (
            "Potentially Credible"
        )


    # =====================================================
    # SUMMARY
    # =====================================================

    summary = (
        f"VERITAS assessed this content as "
        f"{result_label}. "
        f"{model_reason} "
        f"{strength_reason}"
    )


    # =====================================================
    # IMPORTANT INTERPRETATION NOTE
    # =====================================================

    interpretation_note = (
        "The AI classification is a model-based assessment, "
        "not proof that the content is true or false. "
        "Published fact-check evidence, the original source, "
        "and additional context should also be considered."
    )


    # =====================================================
    # RETURN STRUCTURED EXPLANATION
    # =====================================================

    return {

        "result_label":
            result_label,

        "summary":
            summary,

        "model_reason":
            model_reason,

        "strength_reason":
            strength_reason,

        "evidence_reason":
            evidence_reason,

        "interpretation_note":
            interpretation_note
    }