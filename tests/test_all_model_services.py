from services.model_service import (
    predict_claim,
    predict_headline,
    predict_article
)


print()
print("=" * 70)
print("VERITAS — ALL MODEL SERVICES TEST")
print("=" * 70)


# =========================================================
# 1. CLAIM MODEL
# =========================================================

claim = (
    "The government increased education funding this year."
)

print()
print("1. CLAIM MODEL")
print("-" * 70)

claim_result = predict_claim(
    claim
)

print("Text:", claim)
print("Model:", claim_result["model_used"])
print("Prediction:", claim_result["prediction"])
print("Fake:", claim_result["fake_probability"], "%")
print("Real:", claim_result["real_probability"], "%")
print("Decision method:", claim_result["decision_method"])
print("Threshold:", claim_result["threshold"])


# =========================================================
# 2. HEADLINE MODEL
# =========================================================

headline = (
    "University opens new computer science laboratory"
)

print()
print("2. HEADLINE MODEL")
print("-" * 70)

headline_result = predict_headline(
    headline
)

print("Text:", headline)
print("Model:", headline_result["model_used"])
print("Prediction:", headline_result["prediction"])
print("Fake:", headline_result["fake_probability"], "%")
print("Real:", headline_result["real_probability"], "%")
print("Decision method:", headline_result["decision_method"])
print("Threshold:", headline_result["threshold"])


# =========================================================
# 3. ARTICLE MODEL
# =========================================================

article = """
A university has opened a new computer science laboratory
for students. The facility contains computers and networking
equipment intended to support practical lessons and research.
University officials said students will begin using the
laboratory during the current academic semester.
"""

print()
print("3. ARTICLE MODEL")
print("-" * 70)

article_result = predict_article(
    article
)

print("Model:", article_result["model_used"])
print("Prediction:", article_result["prediction"])
print("Fake:", article_result["fake_probability"], "%")
print("Real:", article_result["real_probability"], "%")
print("Decision method:", article_result["decision_method"])
print("Threshold:", article_result["threshold"])
print("Chunks:", article_result["chunks_used"])


print()
print("=" * 70)
print("ALL MODEL SERVICES TEST COMPLETE")
print("=" * 70)