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
    / "fakenewsnet_fullarticle_bert_model"
)

VERITAS_CONFIG_PATH = MODEL_PATH / "veritas_config.json"


# ---------------------------------------------------------
# LOAD VERITAS CONFIGURATION
# ---------------------------------------------------------

with open(VERITAS_CONFIG_PATH, "r", encoding="utf-8") as file:
    veritas_config = json.load(file)


FAKE_LABEL = veritas_config["fake_label"]
REAL_LABEL = veritas_config["real_label"]
FAKE_THRESHOLD = veritas_config["fake_threshold"]
CHUNK_SIZE = veritas_config["chunk_size"]
MAX_CHUNKS = veritas_config["max_chunks"]


# ---------------------------------------------------------
# LOAD TOKENIZER AND MODEL
# ---------------------------------------------------------

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True
)

print("Loading model...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH,
    local_files_only=True
)

model.eval()

print("Model loaded successfully.")

print("NUmber of model labels:", model.config.num_labels)

# ---------------------------------------------------------
# PREDICTION FUNCTION
# ---------------------------------------------------------

def predict_article(text):

    # -----------------------------------------------------
    # TOKENISE ARTICLE WITHOUT SPECIAL TOKENS
    # -----------------------------------------------------

    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_attention_mask=False,
        return_token_type_ids=False
    )

    token_ids = encoded["input_ids"]

    # -----------------------------------------------------
    # SPLIT ARTICLE INTO CHUNKS
    # -----------------------------------------------------

    chunks = []

    for i in range(0, len(token_ids), CHUNK_SIZE):

        chunk = token_ids[i:i + CHUNK_SIZE]

        chunks.append(chunk)

        if len(chunks) >= MAX_CHUNKS:
            break

    # Prevent problems if empty text is supplied
    if not chunks:
        raise ValueError("Article text cannot be empty.")

    # -----------------------------------------------------
    # ANALYSE EACH CHUNK
    # -----------------------------------------------------

    fake_probabilities = []

    for chunk in chunks:

        # Add BERT special tokens:
        # [CLS] article tokens [SEP]

        input_ids = [
            tokenizer.cls_token_id,
            *chunk,
            tokenizer.sep_token_id
        ]

        # Convert to PyTorch tensor
        input_ids = torch.tensor(
            [input_ids],
            dtype=torch.long
        )

        # Every token should be attended to
        attention_mask = torch.ones_like(input_ids)

        # BERT normally uses token type 0 for
        # single-sentence classification
        token_type_ids = torch.zeros_like(input_ids)

        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids
        }

        # -------------------------------------------------
        # MODEL PREDICTION
        # -------------------------------------------------

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
    # COMBINE CHUNK RESULTS
    # -----------------------------------------------------

    average_fake_probability = (
        sum(fake_probabilities)
        / len(fake_probabilities)
    )

    # -----------------------------------------------------
    # APPLY SAVED VERITAS THRESHOLD
    # -----------------------------------------------------

    if average_fake_probability >= FAKE_THRESHOLD:

        prediction = "FAKE"
        confidence = average_fake_probability

    else:

        prediction = "REAL"
        confidence = 1 - average_fake_probability

    # -----------------------------------------------------
    # RETURN RESULT
    # -----------------------------------------------------

    return {
        "prediction": prediction,
        "confidence": confidence,
        "fake_probability": average_fake_probability,
        "real_probability": 1 - average_fake_probability,
        "chunks_used": len(chunks)
    }

# ---------------------------------------------------------
# TEST ARTICLE
# ---------------------------------------------------------

article = """
The government announced a new education programme today.
The programme is intended to provide additional technology
training for university students and improve access to
digital learning resources.
"""


result = predict_article(article)


# ---------------------------------------------------------
# DISPLAY RESULT
# ---------------------------------------------------------

print()
print("=" * 50)
print("VERITAS FULL-ARTICLE MODEL TEST")
print("=" * 50)

print()
print("Article:")
print(article.strip())

print()
print("Prediction:", result["prediction"])

print(
    "Confidence:",
    f'{result["confidence"] * 100:.2f}%'
)

print(
    "Fake probability:",
    f'{result["fake_probability"] * 100:.2f}%'
)

print(
    "Chunks analysed:",
    result["chunks_used"]
)

print()
print("=" * 50)