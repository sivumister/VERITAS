from pathlib import Path
import sys


# Allow Python to find the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(
    0,
    str(PROJECT_ROOT)
)


from services.model_service import predict_article


article = """
The government announced a new education programme today.
The programme aims to provide technology training to
university students and improve access to digital learning.
"""


result = predict_article(article)


print()
print("=" * 50)
print("VERITAS MODEL SERVICE TEST")
print("=" * 50)

print()

print("Prediction:", result["prediction"])

print(
    "Fake probability:",
    f'{result["fake_probability"]}%'
)

print(
    "Real probability:",
    f'{result["real_probability"]}%'
)

print(
    "Decision threshold:",
    f'{result["threshold"]}%'
)

print(
    "Chunks used:",
    result["chunks_used"]
)

print()

print("=" * 50)