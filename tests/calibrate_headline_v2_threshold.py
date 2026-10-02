import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix
)

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)


# =========================================================
# PATHS
# =========================================================

WEBPAGE_ROOT = Path(__file__).resolve().parents[1]

REPOSITORY_ROOT = WEBPAGE_ROOT.parents[2]

MODEL_PATH = (
    WEBPAGE_ROOT
    / "models"
    / "fakenewsnet_headline_bert_v2"
)

DATASET_DIR = (
    REPOSITORY_ROOT
    / "FakeNewsNet"
    / "dataset"
)


csv.field_size_limit(
    50_000_000
)


# =========================================================
# LOAD DATA
# =========================================================

print("=" * 70)
print("LOADING FAKENEWSNET")
print("=" * 70)


politifact_fake = pd.read_csv(
    DATASET_DIR / "politifact_fake.csv"
)

politifact_real = pd.read_csv(
    DATASET_DIR / "politifact_real.csv"
)

gossipcop_fake = pd.read_csv(
    DATASET_DIR / "gossipcop_fake.csv"
)

gossipcop_real = pd.read_csv(
    DATASET_DIR / "gossipcop_real.csv"
)


# =========================================================
# LABEL DATA
# 0 = FAKE
# 1 = REAL
# =========================================================

politifact_fake["label"] = 0
politifact_real["label"] = 1
gossipcop_fake["label"] = 0
gossipcop_real["label"] = 1


df = pd.concat(
    [
        politifact_fake,
        politifact_real,
        gossipcop_fake,
        gossipcop_real
    ],
    ignore_index=True
)


df = df[
    ["title", "label"]
].copy()


# =========================================================
# CLEAN DATA
# =========================================================

df = df.dropna(
    subset=["title"]
)

df["title"] = (
    df["title"]
    .astype(str)
    .str.strip()
)

df = df[
    df["title"] != ""
]


# ---------------------------------------------------------
# Remove titles that have conflicting labels
# ---------------------------------------------------------

label_counts = (
    df.groupby("title")["label"]
    .nunique()
)

conflicting_titles = label_counts[
    label_counts > 1
].index


df_clean = df[
    ~df["title"].isin(
        conflicting_titles
    )
].copy()


# ---------------------------------------------------------
# Remove duplicate titles
# ---------------------------------------------------------

df_clean = df_clean.drop_duplicates(
    subset=["title"],
    keep="first"
)

df_clean = df_clean.reset_index(
    drop=True
)


print(
    "Clean dataset size:",
    len(df_clean)
)

print()

print(
    df_clean["label"]
    .value_counts()
    .sort_index()
)


# =========================================================
# RECREATE ORIGINAL 70 / 15 / 15 SPLIT
# =========================================================

X = df_clean["title"]
y = df_clean["label"]


X_train, X_temp, y_train, y_temp = (
    train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y
    )
)


X_val, X_test, y_val, y_test = (
    train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=42,
        stratify=y_temp
    )
)


print()
print("=" * 70)
print("DATA SPLIT")
print("=" * 70)

print(
    "Training:",
    len(X_train)
)

print(
    "Validation:",
    len(X_val)
)

print(
    "Testing:",
    len(X_test)
)


# =========================================================
# LOAD V2 CONFIG
# =========================================================

with open(
    MODEL_PATH / "veritas_config.json",
    "r",
    encoding="utf-8"
) as file:

    config = json.load(file)


fake_label = config["fake_label"]
real_label = config["real_label"]

max_length = config.get(
    "max_length",
    128
)


print()
print("=" * 70)
print("LOADING LOCAL V2")
print("=" * 70)

print(
    "Model:",
    config["model_type"]
)


# =========================================================
# LOAD TOKENIZER + MODEL
# =========================================================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True
)

model = (
    AutoModelForSequenceClassification
    .from_pretrained(
        MODEL_PATH,
        local_files_only=True
    )
)

model.eval()


# =========================================================
# GENERATE VALIDATION PROBABILITIES
# =========================================================

validation_titles = (
    X_val.tolist()
)

validation_labels = (
    y_val.to_numpy()
)


fake_probabilities = []


BATCH_SIZE = 32


print()
print(
    "Running V2 on validation set..."
)


for start in range(
    0,
    len(validation_titles),
    BATCH_SIZE
):

    batch_titles = validation_titles[
        start:start + BATCH_SIZE
    ]


    inputs = tokenizer(
        batch_titles,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=max_length
    )


    with torch.no_grad():

        outputs = model(
            **inputs
        )


    probabilities = torch.softmax(
        outputs.logits,
        dim=1
    )


    batch_fake_probs = (
        probabilities[
            :,
            fake_label
        ]
        .cpu()
        .numpy()
    )


    fake_probabilities.extend(
        batch_fake_probs
    )


fake_probabilities = np.array(
    fake_probabilities
)


# =========================================================
# SWEEP THRESHOLDS
# =========================================================

results = []


for threshold in np.arange(
    0.40,
    0.91,
    0.01
):

    predictions = np.where(
        fake_probabilities >= threshold,
        fake_label,
        real_label
    )


    accuracy = accuracy_score(
        validation_labels,
        predictions
    )


    macro_precision, macro_recall, macro_f1, _ = (
        precision_recall_fscore_support(
            validation_labels,
            predictions,
            average="macro",
            zero_division=0
        )
    )


    fake_precision, fake_recall, fake_f1, _ = (
        precision_recall_fscore_support(
            validation_labels,
            predictions,
            labels=[fake_label],
            average=None,
            zero_division=0
        )
    )


    real_precision, real_recall, real_f1, _ = (
        precision_recall_fscore_support(
            validation_labels,
            predictions,
            labels=[real_label],
            average=None,
            zero_division=0
        )
    )


    results.append({

        "threshold":
            float(threshold),

        "accuracy":
            accuracy,

        "macro_f1":
            macro_f1,

        "fake_precision":
            fake_precision[0],

        "fake_recall":
            fake_recall[0],

        "fake_f1":
            fake_f1[0],

        "real_precision":
            real_precision[0],

        "real_recall":
            real_recall[0],

        "real_f1":
            real_f1[0]
    })


# =========================================================
# ORIGINAL V2 THRESHOLD
# =========================================================

original = min(
    results,
    key=lambda result:
        abs(
            result["threshold"]
            - 0.51
        )
)


print()
print("=" * 70)
print("ORIGINAL V2 THRESHOLD")
print("=" * 70)

print(
    f"Threshold: "
    f"{original['threshold']:.2f}"
)

print(
    f"Accuracy: "
    f"{original['accuracy']:.4f}"
)

print(
    f"Macro F1: "
    f"{original['macro_f1']:.4f}"
)

print(
    f"FAKE Precision: "
    f"{original['fake_precision']:.4f}"
)

print(
    f"FAKE Recall: "
    f"{original['fake_recall']:.4f}"
)

print(
    f"REAL Recall: "
    f"{original['real_recall']:.4f}"
)


# =========================================================
# CONSERVATIVE CANDIDATES
# REQUIRE REAL RECALL >= 95%
# =========================================================

conservative = [

    result
    for result in results

    if result["real_recall"] >= 0.95
]


if not conservative:

    print()
    print(
        "No threshold achieved "
        "REAL recall >= 95%."
    )

else:

    best_conservative = max(
        conservative,
        key=lambda result:
            result["macro_f1"]
    )


    print()
    print("=" * 70)
    print(
        "BEST CONSERVATIVE THRESHOLD "
        "(REAL RECALL >= 95%)"
    )
    print("=" * 70)


    print(
        f"Threshold: "
        f"{best_conservative['threshold']:.2f}"
    )

    print(
        f"Accuracy: "
        f"{best_conservative['accuracy']:.4f}"
    )

    print(
        f"Macro F1: "
        f"{best_conservative['macro_f1']:.4f}"
    )

    print(
        f"FAKE Precision: "
        f"{best_conservative['fake_precision']:.4f}"
    )

    print(
        f"FAKE Recall: "
        f"{best_conservative['fake_recall']:.4f}"
    )

    print(
        f"FAKE F1: "
        f"{best_conservative['fake_f1']:.4f}"
    )

    print(
        f"REAL Precision: "
        f"{best_conservative['real_precision']:.4f}"
    )

    print(
        f"REAL Recall: "
        f"{best_conservative['real_recall']:.4f}"
    )

    print(
        f"REAL F1: "
        f"{best_conservative['real_f1']:.4f}"
    )


# =========================================================
# SHOW TOP CONSERVATIVE OPTIONS
# =========================================================

print()
print("=" * 70)
print("TOP CONSERVATIVE OPTIONS")
print("=" * 70)


top_candidates = sorted(
    conservative,
    key=lambda result:
        result["macro_f1"],
    reverse=True
)[:10]


for result in top_candidates:

    print(
        f"Threshold {result['threshold']:.2f}"
        f" | Macro F1 {result['macro_f1']:.4f}"
        f" | Fake Recall {result['fake_recall']:.4f}"
        f" | Fake Precision {result['fake_precision']:.4f}"
        f" | Real Recall {result['real_recall']:.4f}"
    )


print()
print("=" * 70)
print("CALIBRATION COMPLETE")
print("=" * 70)