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


from services.model_service import analyze_news


# =========================================================
# TEST 1 - CLAIM
# =========================================================

claim = """
The government has announced a new education
programme for university students.
"""


print()
print("=" * 60)
print("TEST 1 - LIAR CLAIM MODEL")
print("=" * 60)


result = analyze_news(
    claim,
    "claim"
)


print(
    "Model:",
    result["model_used"]
)

print(
    "Prediction:",
    result["prediction"]
)

print(
    "Fake probability:",
    f'{result["fake_probability"]}%'
)

print(
    "Real probability:",
    f'{result["real_probability"]}%'
)


# =========================================================
# TEST 2 - HEADLINE
# =========================================================

headline = """
Government announces new education programme
for university students
"""


print()
print("=" * 60)
print("TEST 2 - HEADLINE MODEL")
print("=" * 60)


result = analyze_news(
    headline,
    "headline"
)


print(
    "Model:",
    result["model_used"]
)

print(
    "Prediction:",
    result["prediction"]
)

print(
    "Fake probability:",
    f'{result["fake_probability"]}%'
)

print(
    "Real probability:",
    f'{result["real_probability"]}%'
)


# =========================================================
# TEST 3 - FULL ARTICLE
# =========================================================

article = """
The government announced a new education programme today.
The programme aims to provide additional technology
training for university students and improve access to
digital learning resources.

Officials said the programme would involve universities,
technology organisations and education institutions.
The initiative is expected to begin later this year.
"""


print()
print("=" * 60)
print("TEST 3 - FULL ARTICLE MODEL")
print("=" * 60)


result = analyze_news(
    article,
    "article"
)


print(
    "Model:",
    result["model_used"]
)

print(
    "Prediction:",
    result["prediction"]
)

print(
    "Fake probability:",
    f'{result["fake_probability"]}%'
)

print(
    "Real probability:",
    f'{result["real_probability"]}%'
)

print(
    "Threshold:",
    f'{result["threshold"]}%'
)

print(
    "Chunks:",
    result["chunks_used"]
)


print()
print("=" * 60)
print("ALL VERITAS MODEL TESTS COMPLETE")
print("=" * 60)