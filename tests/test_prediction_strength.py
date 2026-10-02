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


from services.prediction_strength import (
    calculate_prediction_strength
)


print()
print("=" * 60)
print("VERITAS PREDICTION STRENGTH TEST")
print("=" * 60)


# =========================================================
# TEST 1 — YOUR CURRENT LIAR RESULT
# =========================================================

result = calculate_prediction_strength(
    prediction="FAKE",
    fake_probability=52.05,
    real_probability=47.95,
    threshold=None
)


print()
print("TEST 1 — LIAR BERT")
print("Prediction: FAKE")
print("Fake probability: 52.05%")
print("Real probability: 47.95%")
print("Strength:", result["strength"])
print(
    "Distance from decision boundary:",
    result["distance"],
    "percentage points"
)
print(
    "Decision boundary:",
    result["decision_boundary"],
    "%"
)


# =========================================================
# TEST 2 — STRONGER LIAR RESULT
# =========================================================

result = calculate_prediction_strength(
    prediction="REAL",
    fake_probability=25.0,
    real_probability=75.0,
    threshold=None
)


print()
print("=" * 60)
print("TEST 2 — LIAR BERT")
print("Prediction: REAL")
print("Fake probability: 25%")
print("Real probability: 75%")
print("Strength:", result["strength"])


# =========================================================
# TEST 3 — FULL ARTICLE MODEL
# =========================================================

result = calculate_prediction_strength(
    prediction="FAKE",
    fake_probability=42.0,
    real_probability=58.0,
    threshold=41.0
)


print()
print("=" * 60)
print("TEST 3 — FULL ARTICLE BERT")
print("Prediction: FAKE")
print("Fake probability: 42%")
print("Real probability: 58%")
print("Threshold: 41%")
print("Strength:", result["strength"])
print(
    "Distance from threshold:",
    result["distance"],
    "percentage points"
)


print()
print("=" * 60)