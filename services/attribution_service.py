import re

import torch

from services.model_service import load_model


# =========================================================
# INTEGRATED GRADIENTS FOR ARTICLE MODEL
# =========================================================
def attribute_article_text(
    text,
    steps=24
):

    text = text.strip()

    if not text:

        return {
            "fake_score": None,
            "words": [],
            "tokens": [],
            "positive_words": [],
            "negative_words": [],
            "truncated": False,
            "chunks_used": 0,
            "total_tokens": 0,
            "analysed_tokens": 0,
            "article_threshold": None
        }


    # =====================================================
    # LOAD ARTICLE MODEL
    # =====================================================

    tokenizer, model, config = (
        load_model("article")
    )

    model.eval()


    fake_label = int(
        config["fake_label"]
    )

    article_threshold = float(
        config["fake_threshold"]
    )

    chunk_size = int(
        config["chunk_size"]
    )

    max_chunks = int(
        config["max_chunks"]
    )


    device = next(
        model.parameters()
    ).device


    # =====================================================
    # TOKENISE THE WHOLE ARTICLE
    # =====================================================
    #
    # This now mirrors model_service.py:
    #
    # article
    #   -> tokens
    #   -> chunks
    #   -> analyse each chunk
    #   -> mean FAKE probability
    #
    # Offsets are kept so contributions can later be
    # mapped back onto the original article text.
    # =====================================================

    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_attention_mask=False,
        return_token_type_ids=False,
        return_offsets_mapping=True
    )


    all_token_ids = (
        encoded["input_ids"]
    )

    all_offsets = (
        encoded["offset_mapping"]
    )


    total_tokens = len(
        all_token_ids
    )


    # =====================================================
    # SPLIT INTO THE SAME ARTICLE CHUNKS
    # =====================================================

    chunks = []


    for start in range(
        0,
        total_tokens,
        chunk_size
    ):

        chunk_ids = all_token_ids[
            start:start + chunk_size
        ]

        chunk_offsets = all_offsets[
            start:start + chunk_size
        ]


        if not chunk_ids:
            continue


        chunks.append({
            "ids": chunk_ids,
            "offsets": chunk_offsets
        })


        if len(chunks) >= max_chunks:
            break


    if not chunks:

        return {
            "fake_score": None,
            "words": [],
            "tokens": [],
            "positive_words": [],
            "negative_words": [],
            "truncated": False,
            "chunks_used": 0,
            "total_tokens": total_tokens,
            "analysed_tokens": 0,
            "article_threshold":
                round(
                    article_threshold * 100,
                    2
                )
        }


    analysed_tokens = sum(
        len(chunk["ids"])
        for chunk in chunks
    )


    truncated = (
        analysed_tokens
        < total_tokens
    )


    chunks_used = len(
        chunks
    )


    embedding_layer = (
        model.get_input_embeddings()
    )


    pad_token_id = (
        tokenizer.pad_token_id
    )


    chunk_fake_scores = []

    chunk_token_results = []


    # =====================================================
    # PROCESS EACH MODEL CHUNK
    # =====================================================

    for chunk_number, chunk in enumerate(
        chunks,
        start=1
    ):

        chunk_ids = chunk["ids"]

        chunk_offsets = (
            chunk["offsets"]
        )


        # -------------------------------------------------
        # ADD BERT SPECIAL TOKENS
        # -------------------------------------------------

        sequence_ids = [

            tokenizer.cls_token_id,

            *chunk_ids,

            tokenizer.sep_token_id
        ]


        input_ids = torch.tensor(
            [sequence_ids],
            dtype=torch.long,
            device=device
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


        # =================================================
        # NORMAL MODEL SCORE FOR THIS CHUNK
        # =================================================

        with torch.no_grad():

            outputs = model(

                input_ids=input_ids,

                attention_mask=
                    attention_mask,

                token_type_ids=
                    token_type_ids
            )


            probabilities = torch.softmax(
                outputs.logits,
                dim=1
            )


            fake_probability = float(
                probabilities[
                    0,
                    fake_label
                ].item()
            )


        chunk_fake_scores.append(
            fake_probability
        )


        # =================================================
        # CREATE IG BASELINE
        # =================================================

        baseline_ids = torch.full_like(
            input_ids,
            pad_token_id
        )


        # Keep CLS and SEP
        baseline_ids[:, 0] = (
            input_ids[:, 0]
        )

        baseline_ids[:, -1] = (
            input_ids[:, -1]
        )


        input_embeddings = (
            embedding_layer(
                input_ids
            )
            .detach()
        )


        baseline_embeddings = (
            embedding_layer(
                baseline_ids
            )
            .detach()
        )


        embedding_difference = (
            input_embeddings
            - baseline_embeddings
        )


        total_gradients = (
            torch.zeros_like(
                input_embeddings
            )
        )


        # =================================================
        # INTEGRATED GRADIENTS
        # =================================================

        for step in range(
            steps
        ):

            alpha = (
                step + 0.5
            ) / steps


            interpolated_embeddings = (

                baseline_embeddings

                + alpha
                * embedding_difference
            )


            interpolated_embeddings = (

                interpolated_embeddings
                .detach()
                .requires_grad_(True)
            )


            outputs = model(

                inputs_embeds=
                    interpolated_embeddings,

                attention_mask=
                    attention_mask,

                token_type_ids=
                    token_type_ids
            )


            # Explain the FAKE class output.
            target_logit = (
                outputs.logits[
                    0,
                    fake_label
                ]
            )


            gradients = torch.autograd.grad(

                target_logit,

                interpolated_embeddings

            )[0]


            total_gradients += (
                gradients.detach()
            )


        # =================================================
        # TOKEN ATTRIBUTIONS FOR THIS CHUNK
        # =================================================

        average_gradients = (
            total_gradients
            / steps
        )


        integrated_gradients = (

            embedding_difference
            * average_gradients
        )


        token_attributions = (

            integrated_gradients
            .sum(dim=-1)
            .squeeze(0)
            .detach()
            .cpu()
            .tolist()
        )


        # Remove CLS and SEP attribution values.
        content_attributions = (
            token_attributions[1:-1]
        )


        tokens = (
            tokenizer.convert_ids_to_tokens(
                chunk_ids
            )
        )


        # =================================================
        # NORMALISE WITHIN EACH CHUNK
        # =================================================
        #
        # The article model gives each chunk equal weight
        # when calculating its mean probability.
        #
        # Therefore each chunk receives an equal share of
        # the attribution percentage.
        # =================================================

        attribution_total = sum(

            abs(value)

            for value
            in content_attributions
        )


        if attribution_total == 0:

            attribution_total = 1.0


        chunk_share = (
            100.0
            / chunks_used
        )


        for (
            token,
            attribution,
            offset
        ) in zip(

            tokens,
            content_attributions,
            chunk_offsets
        ):

            start, end = offset


            if start == end:
                continue


            relative_attribution = (

                attribution
                / attribution_total

            ) * chunk_share


            chunk_token_results.append({

                "token":
                    token,

                "text":
                    text[start:end],

                "start":
                    start,

                "end":
                    end,

                "chunk":
                    chunk_number,

                "raw_attribution":
                    attribution,

                "relative_attribution":
                    round(
                        relative_attribution,
                        3
                    )
            })


    # =====================================================
    # ARTICLE FAKE SCORE
    # =====================================================
    #
    # Same aggregation method as model_service.py.
    # =====================================================

    average_fake_score = (

        sum(
            chunk_fake_scores
        )

        / len(
            chunk_fake_scores
        )
    )


    # =====================================================
    # MERGE BERT WORDPIECES
    # =====================================================

    words = []


    for item in chunk_token_results:

        token = item["token"]

        contribution = (
            item[
                "relative_attribution"
            ]
        )


        if (
            token.startswith("##")
            and words
            and item["start"]
                == words[-1]["end"]
        ):

            words[-1]["word"] += (
                item["text"]
            )

            words[-1][
                "contribution"
            ] += contribution

            words[-1]["end"] = (
                item["end"]
            )


        else:

            words.append({

                "word":
                    item["text"],

                "start":
                    item["start"],

                "end":
                    item["end"],

                "chunk":
                    item["chunk"],

                "contribution":
                    contribution
            })


    for word in words:

        word["contribution"] = round(
            word["contribution"],
            3
        )


    # =====================================================
    # RANK MEANINGFUL WORDS
    # =====================================================

    meaningful_words = [

        word

        for word in words

        if re.search(
            r"[A-Za-z0-9]",
            word["word"]
        )
    ]


    positive_words = sorted(

        [
            word

            for word
            in meaningful_words

            if (
                word["contribution"]
                > 0
            )
        ],

        key=lambda item:
            item["contribution"],

        reverse=True
    )


    negative_words = sorted(

        [
            word

            for word
            in meaningful_words

            if (
                word["contribution"]
                < 0
            )
        ],

        key=lambda item:
            item["contribution"]
    )


    # =====================================================
    # LAST CHARACTER ACTUALLY ANALYSED
    # =====================================================

    analysed_char_end = 0


    if chunk_token_results:

        analysed_char_end = max(

            item["end"]

            for item
            in chunk_token_results
        )


    return {

        "fake_score":
            round(
                average_fake_score
                * 100,
                2
            ),

        "article_threshold":
            round(
                article_threshold
                * 100,
                2
            ),

        "steps":
            steps,

        "chunks_used":
            chunks_used,

        "total_tokens":
            total_tokens,

        "analysed_tokens":
            analysed_tokens,

        "truncated":
            truncated,

        "analysed_char_end":
            analysed_char_end,

        "chunk_fake_scores": [

            round(
                score * 100,
                2
            )

            for score
            in chunk_fake_scores
        ],

        "tokens":
            chunk_token_results,

        "words":
            words,

        "positive_words":
            positive_words,

        "negative_words":
            negative_words
    }


def build_influence_highlights(
    text,
    attribution_result,
    minimum_net=1.5,
    maximum_highlights=5
):

    text = text.strip()

    if not text:

        return {
            "segments": [],
            "highlight_count": 0
        }


    # =====================================================
    # SPLIT TEXT INTO SENTENCE SPANS
    # =====================================================

    sentence_pattern = re.compile(
        r'[^.!?]+[.!?]?'
    )


    sentence_spans = []


    for match in sentence_pattern.finditer(
        text
    ):

        raw_sentence = (
            match.group()
        )


        sentence = (
            raw_sentence.strip()
        )


        if not sentence:
            continue


        leading_spaces = (
            len(raw_sentence)
            - len(
                raw_sentence.lstrip()
            )
        )


        start = (
            match.start()
            + leading_spaces
        )


        end = (
            start
            + len(sentence)
        )


        sentence_spans.append({

            "text":
                sentence,

            "start":
                start,

            "end":
                end
        })


    # =====================================================
    # CALCULATE SENTENCE ATTRIBUTION
    # =====================================================

    sentence_results = []


    words = attribution_result.get(
        "words",
        []
    )


    for sentence in sentence_spans:

            sentence_words = [

                word

                for word in words

                if (
                    word["start"]
                    >= sentence["start"]
                    and
                    word["start"]
                    < sentence["end"]
                )
            ]


            positive = sum(

                word["contribution"]

                for word in sentence_words

                if word["contribution"] > 0
            )


            negative = sum(

                word["contribution"]

                for word in sentence_words

                if word["contribution"] < 0
            )


            net = (
                positive
                + negative
            )


            analysed = bool(
                sentence_words
            )


            sentence_results.append({

                "text":
                    sentence["text"],

                "start":
                    sentence["start"],

                "end":
                    sentence["end"],

                "positive_contribution":
                    round(
                        positive,
                        3
                    ),

                "negative_contribution":
                    round(
                        negative,
                        3
                    ),

                "net_contribution":
                    round(
                        net,
                        3
                    ),

                "analysed":
                    analysed,

                "highlighted":
                    False,

                "signal":
                    "NONE"
            })


    # =====================================================
    # FIND ELIGIBLE POSITIVE PASSAGES
    # =====================================================

    eligible = [

        sentence

        for sentence
        in sentence_results

        if (
            sentence["analysed"]
            and
            sentence[
                "net_contribution"
            ]
            >= minimum_net
        )
    ]


    eligible = sorted(

        eligible,

        key=lambda item:
            item[
                "net_contribution"
            ],

        reverse=True
    )


    selected = (
        eligible[
            :maximum_highlights
        ]
    )


    selected_starts = {

        item["start"]

        for item in selected
    }


    # =====================================================
    # ASSIGN DISPLAY STRENGTH
    # =====================================================

    for sentence in sentence_results:

        if (
            sentence["start"]
            in selected_starts
        ):

            sentence[
                "highlighted"
            ] = True


            contribution = (
                sentence[
                    "net_contribution"
                ]
            )


            if contribution >= 8:

                sentence[
                    "signal"
                ] = "HIGH"

            elif contribution >= 3:

                sentence[
                    "signal"
                ] = "MODERATE"

            else:

                sentence[
                    "signal"
                ] = "LOW"


    return {

        "segments":
            sentence_results,

        "highlight_count":
            len(selected),

        "minimum_net":
            minimum_net,

        "maximum_highlights":
            maximum_highlights
    }