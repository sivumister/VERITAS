import csv
import random
from pathlib import Path

from services.model_service import predict_headline


# =========================================================
# SETTINGS
# =========================================================

SAMPLES_PER_FILE = 10
RANDOM_SEED = 42


# =========================================================
# FIND DATASET DIRECTORY
# =========================================================

# Current file:
# WebPage/tests/test_headline_dataset_diagnostics.py

WEBPAGE_ROOT = Path(__file__).resolve().parents[1]

REPOSITORY_ROOT = WEBPAGE_ROOT.parents[2]

DATASET_DIR = (
    REPOSITORY_ROOT
    / "FakeNewsNet"
    / "dataset"
)


# =========================================================
# DATASET FILES
# =========================================================

DATASETS = [

    {
        "name": "PolitiFact Fake",
        "file": "politifact_fake.csv",
        "expected": "FAKE"
    },

    {
        "name": "PolitiFact Real",
        "file": "politifact_real.csv",
        "expected": "REAL"
    },

    {
        "name": "GossipCop Fake",
        "file": "gossipcop_fake.csv",
        "expected": "FAKE"
    },

    {
        "name": "GossipCop Real",
        "file": "gossipcop_real.csv",
        "expected": "REAL"
    }
]


# =========================================================
# LOAD RANDOM HEADLINES
# =========================================================

def load_sample_headlines(
    file_path,
    sample_size
):
    """
    Load headlines from FakeNewsNet and return a
    deterministic random sample.
    """

    headlines = []

    # Some FakeNewsNet rows contain large fields.
    csv.field_size_limit(
        10_000_000
    )

    with open(
        file_path,
        "r",
        encoding="utf-8",
        errors="ignore"
    ) as file:

        reader = csv.DictReader(
            file
        )

        for row in reader:

            title = row.get(
                "title",
                ""
            ).strip()

            if title:

                headlines.append(
                    title
                )


    random_generator = random.Random(
        RANDOM_SEED
    )


    if len(headlines) <= sample_size:

        return headlines


    return random_generator.sample(
        headlines,
        sample_size
    )


# =========================================================
# START DIAGNOSTIC
# =========================================================

print()

print(
    "=" * 80
)

print(
    "VERITAS FAKENEWSNET HEADLINE DIAGNOSTIC"
)

print(
    "=" * 80
)


print()

print(
    "Dataset directory:"
)

print(
    DATASET_DIR
)


# =========================================================
# OVERALL COUNTERS
# =========================================================

total_tests = 0

total_correct = 0

known_fake_total = 0
known_fake_correct = 0

known_real_total = 0
known_real_correct = 0


# =========================================================
# TEST EACH DATASET
# =========================================================

for dataset in DATASETS:

    print()

    print(
        "=" * 80
    )

    print(
        dataset["name"].upper()
    )

    print(
        "=" * 80
    )


    file_path = (
        DATASET_DIR
        / dataset["file"]
    )


    if not file_path.exists():

        print(
            "ERROR: Dataset file not found:"
        )

        print(
            file_path
        )

        continue


    headlines = load_sample_headlines(
        file_path,
        SAMPLES_PER_FILE
    )


    dataset_correct = 0


    for number, headline in enumerate(
        headlines,
        start=1
    ):

        result = predict_headline(
            headline
        )


        prediction = result[
            "prediction"
        ]

        fake_probability = result[
            "fake_probability"
        ]

        real_probability = result[
            "real_probability"
        ]


        correct = (
            prediction
            == dataset["expected"]
        )


        if correct:

            dataset_correct += 1

            result_text = "CORRECT"

        else:

            result_text = "INCORRECT"


        total_tests += 1


        if correct:

            total_correct += 1


        if dataset["expected"] == "FAKE":

            known_fake_total += 1

            if correct:

                known_fake_correct += 1


        else:

            known_real_total += 1

            if correct:

                known_real_correct += 1


        print()

        print(
            f"{number}. {headline}"
        )

        print(
            f"Expected: "
            f"{dataset['expected']}"
        )

        print(
            f"Predicted: "
            f"{prediction}"
        )

        print(
            f"Fake probability: "
            f"{fake_probability}%"
        )

        print(
            f"Real probability: "
            f"{real_probability}%"
        )

        print(
            f"Result: "
            f"{result_text}"
        )


    # =====================================================
    # DATASET SUMMARY
    # =====================================================

    print()

    print(
        "-" * 80
    )

    print(
        f"{dataset['name']} summary"
    )

    print(
        f"Correct: "
        f"{dataset_correct}/{len(headlines)}"
    )

    if headlines:

        percentage = (
            dataset_correct
            / len(headlines)
            * 100
        )

        print(
            f"Diagnostic accuracy: "
            f"{percentage:.2f}%"
        )


# =========================================================
# FINAL SUMMARY
# =========================================================

print()

print(
    "=" * 80
)

print(
    "FINAL DIAGNOSTIC SUMMARY"
)

print(
    "=" * 80
)


print()

print(
    "Known FAKE:"
)

print(
    f"Correct: "
    f"{known_fake_correct}/"
    f"{known_fake_total}"
)


if known_fake_total:

    fake_accuracy = (
        known_fake_correct
        / known_fake_total
        * 100
    )

    print(
        f"Correctly identified: "
        f"{fake_accuracy:.2f}%"
    )


print()

print(
    "Known REAL:"
)

print(
    f"Correct: "
    f"{known_real_correct}/"
    f"{known_real_total}"
)


if known_real_total:

    real_accuracy = (
        known_real_correct
        / known_real_total
        * 100
    )

    print(
        f"Correctly identified: "
        f"{real_accuracy:.2f}%"
    )


print()

print(
    "Overall:"
)

print(
    f"Correct: "
    f"{total_correct}/"
    f"{total_tests}"
)


if total_tests:

    overall_accuracy = (
        total_correct
        / total_tests
        * 100
    )

    print(
        f"Diagnostic accuracy: "
        f"{overall_accuracy:.2f}%"
    )


print()

print(
    "=" * 80
)

print(
    "DIAGNOSTIC COMPLETE"
)

print(
    "=" * 80
)

print()