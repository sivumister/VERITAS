from pathlib import Path
import sys

import pandas as pd
import numpy as np
import torch

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from transformers import (
    BertTokenizer,
    BertForSequenceClassification
)


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = Path(
    r"C:\Users\HP Home\Documents\projects\Veritas-FakeNewsDetector\FakeNewsNet\dataset"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "fakenewsnet_bert_model"
)


print("=" * 70)
print("VERITAS ORIGINAL FAKENEWSNET TEST SET")
print("=" * 70)


# =========================================================
# LOAD DATASETS
# =========================================================

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
# LABELS
# =========================================================

politifact_fake["label"] = 0
politifact_real["label"] = 1

gossipcop_fake["label"] = 0
gossipcop_real["label"] = 1


# =========================================================
# COMBINE
# =========================================================

df = pd.concat(
    [
        politifact_fake,
        politifact_real,
        gossipcop_fake,
        gossipcop_real
    ],
    ignore_index=True
)


print("Original rows:", len(df))


# =========================================================
# SAME CLEANING USED DURING TRAINING
# =========================================================

conflicting_titles = (
    df.groupby("title")["label"]
    .nunique()
)

conflicting_titles = conflicting_titles[
    conflicting_titles > 1
]

df_clean = df[
    ~df["title"].isin(
        conflicting_titles.index
    )
].copy()


df_clean = df_clean.drop_duplicates(
    subset=["title"],
    keep="first"
).reset_index(drop=True)


print("Clean rows:", len(df_clean))

print("\nLabel distribution:")
print(df_clean["label"].value_counts())


# =========================================================
# SAME VARIABLES
# =========================================================

X = df_clean["title"]
y = df_clean["label"]


# =========================================================
# SAME 70 / 15 / 15 SPLIT
# SAME RANDOM STATE
# =========================================================

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)


X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)


print()
print("Training:", len(X_train))
print("Validation:", len(X_val))
print("Testing:", len(X_test))


print("\nTest labels:")
print(y_test.value_counts())


# =========================================================
# LOAD SAVED MODEL
# =========================================================

print("\nLoading saved headline BERT...")


tokenizer = BertTokenizer.from_pretrained(
    MODEL_DIR,
    local_files_only=True
)


model = BertForSequenceClassification.from_pretrained(
    MODEL_DIR,
    local_files_only=True
)


model.eval()


print("Model loaded.")


# =========================================================
# RUN TEST IN BATCHES
# =========================================================

test_texts = X_test.tolist()

predictions = []

batch_size = 16


print("\nRunning predictions...")


for start in range(
    0,
    len(test_texts),
    batch_size
):

    batch_texts = test_texts[
        start:start + batch_size
    ]


    encoded = tokenizer(
        batch_texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )


    with torch.no_grad():

        outputs = model(
            **encoded
        )


    batch_predictions = torch.argmax(
        outputs.logits,
        dim=1
    )


    predictions.extend(
        batch_predictions.cpu().tolist()
    )


# =========================================================
# METRICS
# =========================================================

accuracy = accuracy_score(
    y_test,
    predictions
)


matrix = confusion_matrix(
    y_test,
    predictions
)


print()
print("=" * 70)
print("RESULTS")
print("=" * 70)


print(
    f"Accuracy: {accuracy * 100:.2f}%"
)


print("\nCLASSIFICATION REPORT")

print(
    classification_report(
        y_test,
        predictions,
        target_names=[
            "FAKE",
            "REAL"
        ],
        digits=4
    )
)


print("CONFUSION MATRIX")

print(matrix)


# =========================================================
# EXTRA ERROR RATES
# =========================================================

tn, fp, fn, tp = matrix.ravel()


fake_missed_as_real = fp

real_misclassified_fake = fn


fake_total = tn + fp
real_total = fn + tp


false_negative_rate = (
    fake_missed_as_real
    / fake_total
    * 100
)


false_positive_rate = (
    real_misclassified_fake
    / real_total
    * 100
)


print()

print(
    "FAKE incorrectly predicted REAL:",
    fake_missed_as_real
)

print(
    f"Fake-news miss rate: "
    f"{false_negative_rate:.2f}%"
)


print()

print(
    "REAL incorrectly predicted FAKE:",
    real_misclassified_fake
)

print(
    f"Real-news false-positive rate: "
    f"{false_positive_rate:.2f}%"
)


print("=" * 70)