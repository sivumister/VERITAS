from pathlib import Path
import sys


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

sys.path.insert(
    0,
    str(PROJECT_ROOT)
)


from services.factcheck_service import search_fact_checks

from services.evidence_service import (
    rank_fact_checks,
    determine_evidence_relationship
)


query = "5G causes coronavirus"


print()
print("=" * 60)
print("VERITAS EVIDENCE INTERPRETATION TEST")
print("=" * 60)


# ---------------------------------------------------------
# GET FACT CHECKS
# ---------------------------------------------------------

results = search_fact_checks(
    query,
    max_results=5
)


# ---------------------------------------------------------
# RANK THEM
# ---------------------------------------------------------

ranked = rank_fact_checks(
    query,
    results
)


print()
print("RANKED FACT CHECKS")
print()


for number, result in enumerate(
    ranked,
    start=1
):

    print("=" * 60)

    print(
        f"RESULT {number}"
    )

    print(
        "Claim:",
        result["claim"]
    )

    print(
        "Rating:",
        result["rating"]
    )

    print(
        "Interpreted rating:",
        result["interpreted_rating"]
    )

    print(
        "Relevance:",
        result["relevance"]
    )

    print(
        "Relevance score:",
        result["relevance_score"]
    )


# ---------------------------------------------------------
# SIMULATE AN AI RESULT
# ---------------------------------------------------------

ai_prediction = "REAL"


relationship = (
    determine_evidence_relationship(
        ai_prediction,
        ranked
    )
)


print()
print("=" * 60)

print(
    "AI PREDICTION:",
    ai_prediction
)

print(
    "EVIDENCE STATUS:",
    relationship["status"]
)

print(
    "MESSAGE:",
    relationship["message"]
)

print("=" * 60)