from services.model_service import predict_headline


headline = (
    "Government announces new education programme "
    "for university students"
)


print()
print("=" * 60)
print("VERITAS HEADLINE SERVICE V2 TEST")
print("=" * 60)


result = predict_headline(
    headline
)


print("\nHeadline:")
print(headline)

print(
    "\nModel used:",
    result["model_used"]
)

print(
    "Prediction:",
    result["prediction"]
)

print(
    "Fake probability:",
    result["fake_probability"],
    "%"
)

print(
    "Real probability:",
    result["real_probability"],
    "%"
)

print(
    "Decision method:",
    result["decision_method"]
)

print(
    "Threshold:",
    result["threshold"]
)

print()
print("=" * 60)