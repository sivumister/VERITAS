import csv
import gc
import json
import random
from pathlib import Path

import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)


# =========================================================
# PATHS
# =========================================================

WEBPAGE_ROOT = Path(__file__).resolve().parents[1]

REPOSITORY_ROOT = WEBPAGE_ROOT.parents[2]

V1_PATH = (
    WEBPAGE_ROOT
    / "models"
    / "fakenewsnet_bert_model"
)

V2_PATH = (
    WEBPAGE_ROOT
    / "models"
    / "fakenewsnet_headline_bert_v2"
)

DATASET_DIR = (
    REPOSITORY_ROOT
    / "FakeNewsNet"
    / "dataset"
)


# FakeNewsNet contains very large tweet_ids fields.
csv.field_size_limit(
    50_000_000
)


# =========================================================
# SETTINGS
# =========================================================

RANDOM_SEED = 42

SAMPLES_PER_DATASET = 10


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
# SYNTHETIC HEADLINES
# =========================================================

SYNTHETIC_HEADLINES = [

    # Ordinary / credible-style

    (
        "Local school opens new science laboratory for students",
        "ORDINARY"
    ),

    (
        "Heavy rainfall causes flooding in several communities",
        "ORDINARY"
    ),

    (
        "University introduces new computer science programme",
        "ORDINARY"
    ),

    (
        "Health officials launch vaccination awareness campaign",
        "ORDINARY"
    ),

    (
        "Farmers prepare for start of rainy season",
        "ORDINARY"
    ),


    # Suspicious / implausible-style

    (
        "Scientists confirm drinking bleach cures every known disease",
        "SUSPICIOUS"
    ),

    (
        "Secret government device can read every citizen's thoughts",
        "SUSPICIOUS"
    ),

    (
        "Doctors reveal one fruit that makes humans live for 300 years",
        "SUSPICIOUS"
    ),

    (
        "Scientists discover that humans no longer need sleep",
        "SUSPICIOUS"
    ),

    (
        "New phone application can predict exact date of your death",
        "SUSPICIOUS"
    )
]


# =========================================================
# LOAD KNOWN-LABEL DATASET SAMPLE
# =========================================================

def load_sample(
    file_path,
    expected_label,
    group_name
):

    rows = []

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

                rows.append(
                    {
                        "headline": title,
                        "expected": expected_label,
                        "group": group_name
                    }
                )


    generator = random.Random(
        RANDOM_SEED
    )


    if len(rows) <= SAMPLES_PER_DATASET:

        return rows


    return generator.sample(
        rows,
        SAMPLES_PER_DATASET
    )


# =========================================================
# LOAD MODEL CONFIG
# =========================================================

def load_config(
    model_path
):

    config_path = (
        model_path
        / "veritas_config.json"
    )

    with open(
        config_path,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(
            file
        )


# =========================================================
# RUN ONE MODEL
# =========================================================

def run_model(
    model_name,
    model_path,
    test_cases
):

    print()
    print("=" * 80)
    print(
        f"LOADING {model_name}"
    )
    print("=" * 80)


    config = load_config(
        model_path
    )


    print(
        "Model path:",
        model_path
    )

    print(
        "Decision method:",
        config.get(
            "decision_method"
        )
    )

    print(
        "FAKE threshold:",
        config.get(
            "fake_threshold"
        )
    )


    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            model_path,
            local_files_only=True
        )
    )


    model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            model_path,
            local_files_only=True
        )
    )


    model.eval()


    fake_label = config[
        "fake_label"
    ]

    real_label = config[
        "real_label"
    ]

    max_length = config.get(
        "max_length",
        128
    )

    decision_method = config.get(
        "decision_method",
        "argmax"
    )

    fake_threshold = config.get(
        "fake_threshold",
        0.50
    )


    results = []


    for case in test_cases:

        headline = case[
            "headline"
        ]


        inputs = tokenizer(
            headline,
            return_tensors="pt",
            truncation=True,
            max_length=max_length
        )


        with torch.no_grad():

            outputs = model(
                **inputs
            )


        probabilities = torch.softmax(
            outputs.logits,
            dim=1
        )[0]


        fake_probability = float(
            probabilities[
                fake_label
            ].item()
        )


        real_probability = float(
            probabilities[
                real_label
            ].item()
        )


        # =============================================
        # DECISION RULE
        # =============================================

        if (
            decision_method
            == "fake_probability_threshold"
        ):

            prediction = (
                "FAKE"
                if fake_probability
                >= fake_threshold
                else "REAL"
            )

        else:

            predicted_class = int(
                torch.argmax(
                    probabilities
                ).item()
            )

            prediction = (
                "FAKE"
                if predicted_class
                == fake_label
                else "REAL"
            )


        results.append({

            "headline":
                headline,

            "group":
                case["group"],

            "expected":
                case.get(
                    "expected"
                ),

            "prediction":
                prediction,

            "fake_probability":
                fake_probability
                * 100,

            "real_probability":
                real_probability
                * 100
        })


    # =====================================================
    # UNLOAD MODEL
    # =====================================================

    del model
    del tokenizer

    gc.collect()


    if torch.cuda.is_available():

        torch.cuda.empty_cache()


    print(
        f"{model_name} unloaded."
    )


    return results


# =========================================================
# BUILD TEST CASES
# =========================================================

test_cases = []


for dataset in DATASETS:

    file_path = (
        DATASET_DIR
        / dataset["file"]
    )


    samples = load_sample(
        file_path,
        dataset["expected"],
        dataset["name"]
    )


    test_cases.extend(
        samples
    )


# Add synthetic examples

for headline, group in SYNTHETIC_HEADLINES:

    test_cases.append({

        "headline":
            headline,

        "expected":
            None,

        "group":
            group
    })


print()
print("=" * 80)
print("VERITAS HEADLINE MODEL COMPARISON")
print("=" * 80)

print(
    "Total cases:",
    len(test_cases)
)


# =========================================================
# RUN V1
# =========================================================

v1_results = run_model(
    "HEADLINE BERT V1",
    V1_PATH,
    test_cases
)


# =========================================================
# RUN V2
# =========================================================

v2_results = run_model(
    "HEADLINE BERT V2",
    V2_PATH,
    test_cases
)


# =========================================================
# SIDE-BY-SIDE RESULTS
# =========================================================

print()
print("=" * 100)
print("SIDE-BY-SIDE RESULTS")
print("=" * 100)


for number, (
    case,
    v1,
    v2
) in enumerate(
    zip(
        test_cases,
        v1_results,
        v2_results
    ),
    start=1
):

    print()

    print(
        f"{number}. "
        f"[{case['group']}]"
    )

    print(
        case["headline"]
    )


    if case.get(
        "expected"
    ):

        print(
            "Expected:",
            case["expected"]
        )


    print(
        f"V1: "
        f"{v1['prediction']} | "
        f"Fake {v1['fake_probability']:.2f}% | "
        f"Real {v1['real_probability']:.2f}%"
    )


    print(
        f"V2: "
        f"{v2['prediction']} | "
        f"Fake {v2['fake_probability']:.2f}% | "
        f"Real {v2['real_probability']:.2f}%"
    )


    if (
        v1["prediction"]
        != v2["prediction"]
    ):

        print(
            "*** MODELS DISAGREE ***"
        )


# =========================================================
# KNOWN-LABEL SUMMARY
# =========================================================

print()
print("=" * 80)
print("KNOWN-LABEL DATASET SUMMARY")
print("=" * 80)


for model_name, results in [

    ("V1", v1_results),

    ("V2", v2_results)

]:

    known = [
        result
        for result in results
        if result["expected"]
        is not None
    ]


    fake_cases = [
        result
        for result in known
        if result["expected"]
        == "FAKE"
    ]


    real_cases = [
        result
        for result in known
        if result["expected"]
        == "REAL"
    ]


    fake_correct = sum(
        result["prediction"]
        == "FAKE"
        for result in fake_cases
    )


    real_correct = sum(
        result["prediction"]
        == "REAL"
        for result in real_cases
    )


    total_correct = sum(
        result["prediction"]
        == result["expected"]
        for result in known
    )


    print()

    print(
        f"{model_name}:"
    )

    print(
        f"Known FAKE correctly detected: "
        f"{fake_correct}/{len(fake_cases)}"
    )

    print(
        f"Known REAL correctly detected: "
        f"{real_correct}/{len(real_cases)}"
    )

    print(
        f"Overall correct: "
        f"{total_correct}/{len(known)}"
    )


# =========================================================
# SYNTHETIC SUMMARY
# =========================================================

print()
print("=" * 80)
print("SYNTHETIC HEADLINE SUMMARY")
print("=" * 80)


for model_name, results in [

    ("V1", v1_results),

    ("V2", v2_results)

]:

    ordinary = [
        result
        for result in results
        if result["group"]
        == "ORDINARY"
    ]


    suspicious = [
        result
        for result in results
        if result["group"]
        == "SUSPICIOUS"
    ]


    ordinary_real = sum(
        result["prediction"]
        == "REAL"
        for result in ordinary
    )


    suspicious_fake = sum(
        result["prediction"]
        == "FAKE"
        for result in suspicious
    )


    print()

    print(
        f"{model_name}:"
    )

    print(
        "Ordinary-style predicted REAL:",
        f"{ordinary_real}/{len(ordinary)}"
    )

    print(
        "Suspicious-style predicted FAKE:",
        f"{suspicious_fake}/{len(suspicious)}"
    )


print()
print("=" * 80)
print("COMPARISON COMPLETE")
print("=" * 80)