import re

from services.model_service import predict_article


# =========================================================
# SENTENCE SPLITTING
# =========================================================

def split_into_sentences(text):
    """
    Split article text into meaningful sentences.

    This is intentionally lightweight so that VERITAS
    does not require another NLP library just for basic
    sentence segmentation.
    """

    if not text:
        return []

    # Normalise whitespace
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    # Split after sentence-ending punctuation
    sentences = re.split(
        r"(?<=[.!?])\s+",
        text
    )

    cleaned_sentences = []

    for sentence in sentences:

        sentence = sentence.strip()

        # Ignore extremely short fragments
        if len(sentence.split()) < 5:
            continue

        cleaned_sentences.append(
            sentence
        )

    return cleaned_sentences


# =========================================================
# DETERMINE SENTENCE SIGNAL
# =========================================================

def determine_sentence_signal(
    fake_probability,
    threshold
):
    """
    Convert the sentence fake probability into a
    user-friendly signal relative to the article
    model's decision threshold.
    """

    distance = (
        fake_probability
        - threshold
    )

    # ---------------------------------------------
    # FAKE-LEANING SIGNALS
    # ---------------------------------------------

    if distance >= 15:

        return {
            "direction": "FAKE",
            "strength": "STRONG"
        }

    elif distance >= 5:

        return {
            "direction": "FAKE",
            "strength": "MODERATE"
        }

    elif distance >= 0:

        return {
            "direction": "FAKE",
            "strength": "WEAK"
        }


    # ---------------------------------------------
    # CREDIBLE-LEANING SIGNALS
    # ---------------------------------------------

    elif distance <= -15:

        return {
            "direction": "REAL",
            "strength": "STRONG"
        }

    elif distance <= -5:

        return {
            "direction": "REAL",
            "strength": "MODERATE"
        }

    else:

        return {
            "direction": "REAL",
            "strength": "WEAK"
        }


# =========================================================
# ANALYSE ARTICLE SENTENCES
# =========================================================

def analyze_article_sentences(
    text,
    max_sentences=25
):
    """
    Analyse individual article sentences with the
    existing VERITAS full-article model.

    The results are intended to provide explanatory
    sentence-level signals. They are not independent
    factual proof that a sentence is true or false.
    """

    sentences = split_into_sentences(
        text
    )

    # Prevent extremely long articles from causing
    # excessive inference time.
    sentences = sentences[
        :max_sentences
    ]

    results = []


    for index, sentence in enumerate(
        sentences,
        start=1
    ):

        result = predict_article(
            sentence
        )

        fake_probability = result[
            "fake_probability"
        ]

        real_probability = result[
            "real_probability"
        ]

        threshold = result[
            "threshold"
        ]

        signal = determine_sentence_signal(
            fake_probability,
            threshold
        )


        results.append({

            "sentence_number":
                index,

            "text":
                sentence,

            "fake_probability":
                fake_probability,

            "real_probability":
                real_probability,

            "threshold":
                threshold,

            "direction":
                signal["direction"],

            "strength":
                signal["strength"]
        })


    return results