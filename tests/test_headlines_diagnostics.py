from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from services.model_service import predict_headline


# =========================================================
# ADD HEADLINES YOU HAVE VERIFIED YOURSELF
# =========================================================

test_headlines = [

    # REAL HEADLINES
    ("REAL", "Rescuers 'pulling people out with their bare hands' as Venezuela earthquakes kill 920"),
    ("REAL", "Smoke seen near Riyadh airport after air raid alerts in Saudi capital"),
    ("REAL", "Millions without power as Cuba hit by latest major blackout"),
        ("REAL", "Anthropic opens a biology laboratory as it expands its AI drug research programme"),

    ("REAL", "Walgreens plans a new technology centre in Chennai expected to create around 250 jobs"),

    ("REAL", "China's technology boom boosts factory activity while consumer spending remains weak"),

    ("REAL", "WHO warns that the global shortage of healthcare workers could worsen as doctors retire"),

    ("REAL", "WHO says progress is being made against Congo's Ebola outbreak but the crisis is far from over"),

    ("REAL", "New MRI contrast agent shows potential for detecting small lung tumours earlier"),

    ("REAL", "New warnings about artificial intelligence risks revive debate over advanced AI safety"),

    ("REAL", "Gene-edited beagles could offer a future option for people with dog allergies"),

    ("REAL", "Artificial intelligence and data centres are driving major increases in energy and water use"),

    ("REAL", "Wildfire smoke contributes to tens of thousands of deaths worldwide each year"),

    # FAKE HEADLINES
   
      ("FAKE", "Scientists confirm drinking four cups of coffee a day makes humans immune to all viruses"),

    ("FAKE", "New study finds smartphones can fully recharge themselves using moonlight"),

    ("FAKE", "Researchers discover a plant that can generate unlimited electricity without sunlight"),

    ("FAKE", "Doctors announce that sleeping only two hours a night improves memory by 300 percent"),

    ("FAKE", "Scientists discover that eating chocolate every day permanently prevents heart disease"),

    ("FAKE", "New technology allows mobile phones to connect to the internet without networks or satellites"),

    ("FAKE", "Researchers prove that listening to music can permanently double a person's intelligence"),

    ("FAKE", "Scientists create a battery that can power a laptop continuously for fifty years"),

    ("FAKE", "New medical treatment allows broken bones to heal completely within twenty four hours"),
]


# =========================================================
# RUN TEST
# =========================================================

correct = 0

real_total = 0
real_predicted_fake = 0

fake_total = 0
fake_predicted_real = 0


print()
print("=" * 80)
print("VERITAS HEADLINE MODEL DIAGNOSTIC TEST")
print("=" * 80)


for expected, headline in test_headlines:

    result = predict_headline(headline)

    predicted = result["prediction"]

    fake_probability = result["fake_probability"]
    real_probability = result["real_probability"]


    if predicted == expected:
        correct += 1


    if expected == "REAL":

        real_total += 1

        if predicted == "FAKE":
            real_predicted_fake += 1


    if expected == "FAKE":

        fake_total += 1

        if predicted == "REAL":
            fake_predicted_real += 1


    print()
    print("-" * 80)
    print("Headline:", headline)
    print("Expected:", expected)
    print("Predicted:", predicted)
    print("Fake probability:", fake_probability)
    print("Real probability:", real_probability)


# =========================================================
# SUMMARY
# =========================================================

total = len(test_headlines)

accuracy = (
    correct / total * 100
    if total
    else 0
)


false_positive_rate = (
    real_predicted_fake / real_total * 100
    if real_total
    else 0
)


false_negative_rate = (
    fake_predicted_real / fake_total * 100
    if fake_total
    else 0
)


print()
print("=" * 80)
print("SUMMARY")
print("=" * 80)

print(
    f"Correct: {correct}/{total}"
)

print(
    f"Accuracy: {accuracy:.2f}%"
)

print()

print(
    f"REAL headlines tested: {real_total}"
)

print(
    f"REAL incorrectly classified FAKE: "
    f"{real_predicted_fake}"
)

print(
    f"False-positive rate: "
    f"{false_positive_rate:.2f}%"
)

print()

print(
    f"FAKE headlines tested: {fake_total}"
)

print(
    f"FAKE incorrectly classified REAL: "
    f"{fake_predicted_real}"
)

print(
    f"False-negative rate: "
    f"{false_negative_rate:.2f}%"
)

print("=" * 80)