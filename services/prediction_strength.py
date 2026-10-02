# =========================================================
# VERITAS MODEL PREDICTION STRENGTH
# =========================================================


def calculate_prediction_strength(
    prediction,
    fake_probability,
    real_probability,
    threshold=None
):

    prediction = prediction.upper()


    # =====================================================
    # DETERMINE DECISION BOUNDARY
    # =====================================================

  # Models using normal argmax, such as the LIAR claim model,
# use a 50% decision boundary.
    if threshold is None:

        decision_boundary = 50.0


        if prediction == "FAKE":

            distance = (
                fake_probability
                - decision_boundary
            )

        else:

            distance = (
                real_probability
                - decision_boundary
            )


# =====================================================
# THRESHOLD-BASED MODELS
# Used by Headline V2 and Full-Article BERT
# =====================================================

    else:

        decision_boundary = float(
            threshold
        )


# Threshold-based classification uses the distance
# between fake probability and the model's saved boundary.
        distance = abs(
            fake_probability
            - decision_boundary
        )


    distance = abs(
        float(distance)
    )


    # =====================================================
    # STRENGTH LEVEL
    # =====================================================

    if distance < 5:

        strength = "LOW"

    elif distance < 15:

        strength = "MODERATE"

    else:

        strength = "HIGH"


    return {

        "strength":
            strength,

        "distance":
            round(distance, 2),

        "decision_boundary":
            round(decision_boundary, 2)
    }