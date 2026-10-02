from services.assessment_service import (
    build_overall_assessment
)


def show_test(
    name,
    prediction,
    relationship,
    factchecks=None
):

    prediction_strength = {
        "strength": "HIGH",
        "distance": 20.0
    }

    result = build_overall_assessment(
        prediction=prediction,
        prediction_strength=prediction_strength,
        evidence_relationship={
            "status": relationship
        },
        factcheck_results=factchecks
    )

    print()
    print("=" * 70)
    print(name)
    print("=" * 70)

    print("Status:", result["status"])
    print("Label:", result["label"])
    print("Headline:", result["headline"])
    print()
    print(result["message"])
    print()
    print("Guidance:")
    print(result["guidance"])


# =========================================================
# SAMPLE EXTERNAL EVIDENCE
# =========================================================

false_factcheck = [
    {
        "publisher": "Example Fact Checker",
        "relevance": "HIGH",
        "interpreted_rating": "FALSE"
    }
]


true_factcheck = [
    {
        "publisher": "Example Fact Checker",
        "relevance": "HIGH",
        "interpreted_rating": "TRUE"
    }
]


# =========================================================
# TEST 1 — AGREEMENT
# =========================================================

show_test(
    name="TEST 1 - AI FAKE + FACT CHECK FALSE",
    prediction="FAKE",
    relationship="AGREEMENT",
    factchecks=false_factcheck
)


# =========================================================
# TEST 2 — CONFLICT
# =========================================================

show_test(
    name="TEST 2 - AI FAKE + FACT CHECK TRUE",
    prediction="FAKE",
    relationship="CONFLICT",
    factchecks=true_factcheck
)


# =========================================================
# TEST 3 — NO MATCH
# =========================================================

show_test(
    name="TEST 3 - NO FACT CHECK MATCH",
    prediction="REAL",
    relationship="NO_MATCH"
)


# =========================================================
# TEST 4 — UNCLEAR
# =========================================================

show_test(
    name="TEST 4 - UNCLEAR EXTERNAL EVIDENCE",
    prediction="FAKE",
    relationship="NO_CLEAR_VERDICT"
)


# =========================================================
# TEST 5 — API UNAVAILABLE
# =========================================================

show_test(
    name="TEST 5 - EXTERNAL SERVICE UNAVAILABLE",
    prediction="REAL",
    relationship="UNAVAILABLE"
)