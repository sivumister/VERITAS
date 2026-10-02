from services.explanation_service import (
    build_explanation
)


# =========================================================
# TEST 1 — FAKE ARTICLE
# =========================================================

prediction_strength = {
    "strength": "MODERATE",
    "distance": 9.62,
    "decision_boundary": 41.0
}


evidence_relationship = {
    "status": "AGREEMENT",
    "message": "AI and evidence agree."
}


factcheck_results = [
    {
        "publisher": "Example Fact Checker",
        "rating": "False",
        "relevance": "HIGH"
    }
]


explanation = build_explanation(
    prediction="FAKE",
    fake_probability=50.62,
    real_probability=49.38,
    prediction_strength=prediction_strength,
    threshold=41.0,
    evidence_relationship=evidence_relationship,
    factcheck_results=factcheck_results
)


print()
print("=" * 70)
print("VERITAS EXPLANATION TEST")
print("=" * 70)

print()

print(
    "RESULT LABEL:"
)

print(
    explanation["result_label"]
)

print()

print(
    "MODEL REASON:"
)

print(
    explanation["model_reason"]
)

print()

print(
    "STRENGTH:"
)

print(
    explanation["strength_reason"]
)

print()

print(
    "EXTERNAL EVIDENCE:"
)

print(
    explanation["evidence_reason"]
)

print()

print(
    "INTERPRETATION:"
)

print(
    explanation["interpretation_note"]
)

print()