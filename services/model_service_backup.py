from pathlib import Path
import json

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ---------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "fakenewsnet_fullarticle_bert_model"
)

CONFIG_PATH = MODEL_PATH / "veritas_config.json"


# ---------------------------------------------------------
# LOAD MODEL CONFIGURATION
# ---------------------------------------------------------

with open(CONFIG_PATH, "r", encoding="utf-8") as file:
    veritas_config = json.load(file)


FAKE_LABEL = veritas_config["fake_label"]
REAL_LABEL = veritas_config["real_label"]
FAKE_THRESHOLD = veritas_config["fake_threshold"]
CHUNK_SIZE = veritas_config["chunk_size"]
MAX_CHUNKS = veritas_config["max_chunks"]


# ---------------------------------------------------------
# LOAD TOKENIZER AND MODEL
# ---------------------------------------------------------

print("Loading VERITAS Full-Article model...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True
)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH,
    local_files_only=True
)

model.eval()

print("VERITAS Full-Article model loaded.")


# ---------------------------------------------------------
# ARTICLE PREDICTION
# ---------------------------------------------------------

def predict_article(text):

    # Clean basic whitespace
    text = text.strip()

    if not text:
        raise ValueError("Article text cannot be empty.")

    # Tokenise entire article without special tokens
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_attention_mask=False,
        return_token_type_ids=False
    )

    token_ids = encoded["input_ids"]

    # -----------------------------------------------------
    # SPLIT INTO ARTICLE CHUNKS
    # -----------------------------------------------------

    chunks = []

    for i in range(0, len(token_ids), CHUNK_SIZE):

        chunk = token_ids[i:i + CHUNK_SIZE]

        chunks.append(chunk)

        if len(chunks) >= MAX_CHUNKS:
            break

    # -----------------------------------------------------
    # ANALYSE CHUNKS
    # -----------------------------------------------------

    fake_probabilities = []

    for chunk in chunks:

        input_ids = [
            tokenizer.cls_token_id,
            *chunk,
            tokenizer.sep_token_id
        ]

        input_ids = torch.tensor(
            [input_ids],
            dtype=torch.long
        )

        attention_mask = torch.ones_like(input_ids)

        token_type_ids = torch.zeros_like(input_ids)

        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids
        }

        with torch.no_grad():

            outputs = model(**inputs)

            probabilities = torch.softmax(
                outputs.logits,
                dim=1
            )

        fake_probability = (
            probabilities[0][FAKE_LABEL].item()
        )

        fake_probabilities.append(
            fake_probability
        )

    # -----------------------------------------------------
    # COMBINE CHUNK PREDICTIONS
    # -----------------------------------------------------

    average_fake_probability = (
        sum(fake_probabilities)
        / len(fake_probabilities)
    )

    real_probability = 1 - average_fake_probability

    # -----------------------------------------------------
    # FINAL CLASSIFICATION
    # -----------------------------------------------------

    if average_fake_probability >= FAKE_THRESHOLD:

        prediction = "FAKE"

    else:

        prediction = "REAL"

    # -----------------------------------------------------
    # RETURN RESULT
    # -----------------------------------------------------

    return {
        "prediction": prediction,

        "fake_probability":
            round(average_fake_probability * 100, 2),

        "real_probability":
            round(real_probability * 100, 2),

        "threshold":
            round(FAKE_THRESHOLD * 100, 2),

        "chunks_used":
            len(chunks)
    }
    