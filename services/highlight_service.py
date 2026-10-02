import re

import torch

from services.model_service import load_model


# =========================================================
# SPLIT ARTICLE INTO SENTENCES
# =========================================================

def split_into_sentences(text):

    text = text.strip()

    if not text:
        return []

    sentences = re.split(
        r'(?<=[.!?])\s+',
        text
    )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


# =========================================================
# SCORE ONE COMPLETE ARTICLE
# Uses the same basic chunking strategy as article model
# =========================================================

def score_complete_article(
    text,
    tokenizer,
    model,
    config
):

    fake_label = int(
        config["fake_label"]
    )

    chunk_size = int(
        config["chunk_size"]
    )

    max_chunks = int(
        config["max_chunks"]
    )


    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_attention_mask=False,
        return_token_type_ids=False
    )


    token_ids = encoded[
        "input_ids"
    ]


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
        return 0.0


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


        fake_probabilities.append(
            fake_probability
        )


    return (
        sum(fake_probabilities)
        / len(fake_probabilities)
    )


# =========================================================
# CALCULATE SENTENCE INFLUENCE
# =========================================================

def score_article_passages(text):

    sentences = split_into_sentences(
        text
    )


    if not sentences:

        return {
            "segments": [],
            "total_segments": 0,
            "baseline_fake_score": None,
            "article_threshold": None
        }


    tokenizer, model, config = (
        load_model("article")
    )


    article_threshold = float(
        config["fake_threshold"]
    )


    # =====================================================
    # ORIGINAL ARTICLE SCORE
    # =====================================================

    baseline_score = score_complete_article(
        text,
        tokenizer,
        model,
        config
    )


    segments = []


    # =====================================================
    # REMOVE EACH SENTENCE ONCE
    # =====================================================

    for index, sentence in enumerate(
        sentences
    ):

        remaining_sentences = [

            current_sentence

            for current_index, current_sentence
            in enumerate(sentences)

            if current_index != index
        ]


        reduced_article = " ".join(
            remaining_sentences
        )


        reduced_score = score_complete_article(
            reduced_article,
            tokenizer,
            model,
            config
        )


        influence = (
            baseline_score
            - reduced_score
        )


        influence_percent = (
            influence * 100
        )


        segments.append({

            "text":
                sentence,

            "baseline_fake_score":
                round(
                    baseline_score * 100,
                    2
                ),

            "score_without_sentence":
                round(
                    reduced_score * 100,
                    2
                ),

            "influence":
                round(
                    influence_percent,
                    2
                )
        })


    # =====================================================
    # RANK BY POSITIVE CONTRIBUTION
    # =====================================================

    positive_segments = [

        segment

        for segment in segments

        if segment["influence"] > 0
    ]


    ranked_positive = sorted(
        positive_segments,
        key=lambda segment:
            segment["influence"],
        reverse=True
    )


    return {

        "segments":
            segments,

        "ranked_positive":
            ranked_positive,

        "total_segments":
            len(segments),

        "baseline_fake_score":
            round(
                baseline_score * 100,
                2
            ),

        "article_threshold":
            round(
                article_threshold * 100,
                2
            )
    }