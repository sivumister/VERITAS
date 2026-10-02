from pathlib import Path
import json

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "LIAR_MODEL"
)

CONFIG_PATH = MODEL_PATH / "veritas_config.json"


# ---------------------------------------------------------
# LOAD VERITAS CONFIG
# ---------------------------------------------------------

with open(
    CONFIG_PATH,
    "r",
    encoding="utf-8"
) as file:

    config = json.load(file)


FAKE_LABEL = config["fake_label"]
REAL_LABEL = config["real_label"]
MAX_LENGTH = config["max_length"]


# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------

print("Loading VERITAS LIAR model...")


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True
)


model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH,
    local_files_only=True
)


model.eval()


print("LIAR model loaded successfully.")
print("Number of labels:", model.config.num_labels)


# ---------------------------------------------------------
# PREDICTION FUNCTION
# ---------------------------------------------------------

def predict_claim(text):

    text = text.strip()

    if not text:
        raise ValueError("Claim cannot be empty.")

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH
    )

    with torch.no_grad():

        outputs = model(**inputs)

        probabilities = torch.softmax(
            outputs.logits,
            dim=1
        )[0]

    fake_probability = (
        probabilities[FAKE_LABEL].item()
    )

    real_probability = (
        probabilities[REAL_LABEL].item()
    )

    predicted_class = torch.argmax(
        probabilities
    ).item()

    if predicted_class == FAKE_LABEL:

        prediction = "FAKE"

    else:

        prediction = "REAL"

    return {
        "prediction": prediction,

        "fake_probability":
            round(fake_probability * 100, 2),

        "real_probability":
            round(real_probability * 100, 2)
    }


# ---------------------------------------------------------
# TEST CLAIM
# ---------------------------------------------------------

claim = """
The government has announced a new education programme
for university students.
"""


result = predict_claim(claim)


# ---------------------------------------------------------
# DISPLAY RESULT
# ---------------------------------------------------------

print()
print("=" * 50)
print("VERITAS LIAR MODEL TEST")
print("=" * 50)

print()

print("Claim:")
print(claim.strip())

print()

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

print()

print("=" * 50)