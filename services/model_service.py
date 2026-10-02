from pathlib import Path
import json
import gc

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)


# =========================================================
# PROJECT PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATHS = {

    "claim":
        BASE_DIR
        / "models"
        / "LIAR_MODEL",

    "headline":
        BASE_DIR
        / "models"
        / "fakenewsnet_headline_bert_v2",

    "article":
        BASE_DIR
        / "models"
        / "fakenewsnet_fullarticle_bert_model"
}


# =========================================================
# ACTIVE MODEL
# =========================================================

active_model_type = None

tokenizer = None
model = None
model_config = None


# =========================================================
# UNLOAD CURRENT MODEL
# =========================================================

def unload_model():

    global active_model_type
    global tokenizer
    global model
    global model_config

    if model is not None:

        print(
            f"Unloading {active_model_type} model..."
        )

    model = None
    tokenizer = None
    model_config = None
    active_model_type = None

    gc.collect()


# =========================================================
# LOAD REQUESTED MODEL
# =========================================================

def load_model(model_type):

    global active_model_type
    global tokenizer
    global model
    global model_config

    if model_type not in MODEL_PATHS:

        raise ValueError(
            f"Unknown model type: {model_type}"
        )

    # If correct model is already loaded,
    # do not reload it.
    if (
        active_model_type == model_type
        and model is not None
        and tokenizer is not None
    ):

        return tokenizer, model, model_config

    # Remove previous model
    unload_model()

    model_path = MODEL_PATHS[model_type]

    config_path = (
        model_path
        / "veritas_config.json"
    )

    print()
    print(
        f"Loading VERITAS {model_type} model..."
    )

    # Load our custom VERITAS config
    with open(
        config_path,
        "r",
        encoding="utf-8"
    ) as file:

        model_config = json.load(file)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True
    )

    # Load trained BERT model
    model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            model_path,
            local_files_only=True
        )
    )

    model.eval()

    active_model_type = model_type

    print(
        f"VERITAS {model_type} model loaded."
    )

    return tokenizer, model, model_config


# =========================================================
# SHORT TEXT PREDICTION
# Used by LIAR and Headline models
# =========================================================

def predict_short_text(text, model_type):

    text = text.strip()

    if not text:

        raise ValueError(
            "Text cannot be empty."
        )


    current_tokenizer, current_model, config = (
        load_model(model_type)
    )


    fake_label = int(
        config["fake_label"]
    )

    real_label = int(
        config["real_label"]
    )

    max_length = int(
        config["max_length"]
    )


    # -----------------------------------------------------
    # DECISION METHOD
    # -----------------------------------------------------

    decision_method = config.get(
        "decision_method",
        "argmax"
    )


    # -----------------------------------------------------
    # TOKENISE INPUT
    # -----------------------------------------------------

    inputs = current_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=max_length
    )


    # -----------------------------------------------------
    # MODEL PREDICTION
    # -----------------------------------------------------

    with torch.no_grad():

        outputs = current_model(
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


    # -----------------------------------------------------
    # APPLY MODEL-SPECIFIC DECISION RULE
    # -----------------------------------------------------

    fake_threshold = None


    if (
        decision_method
        == "fake_probability_threshold"
    ):

        fake_threshold = float(
            config["fake_threshold"]
        )

        if (
            fake_probability
            >= fake_threshold
        ):

            prediction = "FAKE"

        else:

            prediction = "REAL"


    elif (
        decision_method
        == "argmax"
    ):

        predicted_class = int(
            torch.argmax(
                probabilities
            ).item()
        )


        if (
            predicted_class
            == fake_label
        ):

            prediction = "FAKE"

        elif (
            predicted_class
            == real_label
        ):

            prediction = "REAL"

        else:

            raise ValueError(
                "Model returned an unknown class."
            )


    else:

        raise ValueError(
            f"Unknown decision method: "
            f"{decision_method}"
        )


    # -----------------------------------------------------
    # RETURN RESULT
    # -----------------------------------------------------

    return {

        "input_type":
            model_type,

        "model_used":
            config["model_type"],

        "prediction":
            prediction,

        "fake_probability":
            round(
                fake_probability * 100,
                2
            ),

        "real_probability":
            round(
                real_probability * 100,
                2
            ),

        "decision_method":
            decision_method,

        "threshold":
            (
                round(
                    fake_threshold * 100,
                    2
                )
                if fake_threshold is not None
                else None
            ),

        "chunks_used":
            None
    }

# =========================================================
# LIAR CLAIM PREDICTION
# =========================================================

def predict_claim(text):

    return predict_short_text(
        text,
        "claim"
    )


# =========================================================
# HEADLINE PREDICTION
# =========================================================

def predict_headline(text):

    return predict_short_text(
        text,
        "headline"
    )


# =========================================================
# FULL ARTICLE PREDICTION
# =========================================================

def predict_article(text):

    text = text.strip()

    if not text:

        raise ValueError(
            "Article text cannot be empty."
        )

    current_tokenizer, current_model, config = (
        load_model("article")
    )

    fake_label = int(
        config["fake_label"]
    )

    fake_threshold = float(
        config["fake_threshold"]
    )

    chunk_size = int(
        config["chunk_size"]
    )

    max_chunks = int(
        config["max_chunks"]
    )


    # -----------------------------------------------------
    # TOKENISE WHOLE ARTICLE
    # -----------------------------------------------------

    encoded = current_tokenizer(
        text,
        add_special_tokens=False,
        return_attention_mask=False,
        return_token_type_ids=False
    )

    token_ids = encoded[
        "input_ids"
    ]


    # -----------------------------------------------------
    # SPLIT ARTICLE INTO CHUNKS
    # -----------------------------------------------------

    chunks = []

    for i in range(
        0,
        len(token_ids),
        chunk_size
    ):

        chunk = token_ids[
            i:i + chunk_size
        ]

        chunks.append(
            chunk
        )

        if len(chunks) >= max_chunks:

            break


    if not chunks:

        raise ValueError(
            "Article produced no tokens."
        )


    # -----------------------------------------------------
    # ANALYSE EACH CHUNK
    # -----------------------------------------------------

    fake_probabilities = []


    for chunk in chunks:

        input_ids = [

            current_tokenizer.cls_token_id,

            *chunk,

            current_tokenizer.sep_token_id
        ]

        input_ids = torch.tensor(
            [input_ids],
            dtype=torch.long
        )

        attention_mask = (
            torch.ones_like(
                input_ids
            )
        )

        token_type_ids = (
            torch.zeros_like(
                input_ids
            )
        )

        inputs = {

            "input_ids":
                input_ids,

            "attention_mask":
                attention_mask,

            "token_type_ids":
                token_type_ids
        }


        with torch.no_grad():

            outputs = current_model(
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

        fake_probabilities.append(
            fake_probability
        )


    # -----------------------------------------------------
    # MEAN PROBABILITY
    # -----------------------------------------------------

    average_fake_probability = (

        sum(fake_probabilities)

        /

        len(fake_probabilities)
    )


    real_probability = (
        1
        - average_fake_probability
    )


    # -----------------------------------------------------
    # APPLY SAVED THRESHOLD
    # -----------------------------------------------------

    if (
        average_fake_probability
        >= fake_threshold
    ):

        prediction = "FAKE"

    else:

        prediction = "REAL"


    return {

        "input_type":
            "article",

        "model_used":
            config["model_type"],

        "prediction":
            prediction,

        "fake_probability":
            round(
                average_fake_probability
                * 100,
                2
            ),

        "real_probability":
            round(
                real_probability
                * 100,
                2
            ),

        "decision_method":
            config.get(
                "aggregation",
                "mean_probability"
            ),

        "threshold":
            round(
                fake_threshold * 100,
                2
            ),

        "chunks_used":
            len(chunks)
    }


# =========================================================
# GENERAL VERITAS ANALYSIS FUNCTION
# =========================================================

def analyze_news(text, input_type):

    input_type = (
        input_type
        .strip()
        .lower()
    )

    if input_type == "claim":

        return predict_claim(
            text
        )

    elif input_type == "headline":

        return predict_headline(
            text
        )

    elif input_type == "article":

        return predict_article(
            text
        )

    else:

        raise ValueError(
            "Input type must be claim, "
            "headline, or article."
        )