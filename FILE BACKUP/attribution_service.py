import hashlib
import math
import re
from collections import OrderedDict
from threading import Lock

import torch

from services.model_service import load_model


# Eight midpoint samples on at most two chunks keep CPU attribution bounded.
# This is an approximate, local FAKE-logit explanation, not a factual verdict.
DEFAULT_STEPS = 8
DEFAULT_MAX_CHUNKS = 2
_CACHE_LIMIT = 32
_highlight_cache = OrderedDict()
_cache_lock = Lock()
_generation_lock = Lock()


class HighlightsBusy(RuntimeError):
    pass


def attribute_article_text(text, steps=DEFAULT_STEPS,
                           max_explanation_chunks=DEFAULT_MAX_CHUNKS,
                           prediction_context=None):
    text = text.strip()
    if not 1 <= steps <= 64 or not 1 <= max_explanation_chunks <= 10:
        raise ValueError("Invalid attribution settings.")
    if not text:
        return {"fake_score": None, "words": [], "tokens": [],
                "positive_words": [], "negative_words": [], "truncated": False,
                "chunks_used": 0, "prediction_chunks": 0, "total_tokens": 0,
                "analysed_tokens": 0, "article_threshold": None,
                "partial_coverage": False, "selected_chunks": [], "steps": steps}

    tokenizer, model, config = load_model("article")
    model.eval()
    fake_label = int(config["fake_label"])
    chunk_size = int(config["chunk_size"])
    max_chunks = int(config["max_chunks"])
    device = next(model.parameters()).device
    encoded = tokenizer(text, add_special_tokens=False,
                        return_attention_mask=False, return_token_type_ids=False,
                        return_offsets_mapping=True)
    all_ids = encoded["input_ids"]
    offsets = encoded["offset_mapping"]
    chunks = [{"ids": all_ids[start:start + chunk_size],
               "offsets": offsets[start:start + chunk_size], "number": number}
              for number, start in enumerate(
                  range(0, min(len(all_ids), chunk_size * max_chunks), chunk_size), 1)]
    if not chunks:
        raise ValueError("Article produced no tokens.")

    # Reuse the original inference scores only when tokenization/config match.
    context = prediction_context or {}
    scores = context.get("chunk_fake_probabilities", [])
    reusable = (
        context.get("text_sha256") == hashlib.sha256(text.encode("utf-8")).hexdigest()
        and context.get("model_used") == config["model_type"]
        and context.get("fake_label") == fake_label
        and context.get("chunk_size") == chunk_size
        and context.get("max_chunks") == max_chunks
        and context.get("total_tokens") == len(all_ids)
        and len(scores) == len(chunks)
        and all(isinstance(score, (int, float)) and math.isfinite(score)
                and 0 <= score <= 1 for score in scores)
    )
    if not reusable:
        # Older history entries have no stored chunk scores.
        scores = []
        for chunk in chunks:
            ids = torch.tensor([[tokenizer.cls_token_id, *chunk["ids"],
                                 tokenizer.sep_token_id]], dtype=torch.long, device=device)
            with torch.no_grad():
                logits = model(input_ids=ids, attention_mask=torch.ones_like(ids),
                               token_type_ids=torch.zeros_like(ids)).logits
                scores.append(float(torch.softmax(logits, dim=1)[0, fake_label].item()))

    # Rank by FAKE probability, then restore article order for word/span mapping.
    selected = sorted(sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)
                      [:max_explanation_chunks])
    embedding_layer = model.get_input_embeddings()
    token_results = []
    for index in selected:
        chunk = chunks[index]
        ids = torch.tensor([[tokenizer.cls_token_id, *chunk["ids"],
                             tokenizer.sep_token_id]], dtype=torch.long, device=device)
        baseline_ids = torch.full_like(ids, tokenizer.pad_token_id)
        baseline_ids[:, 0], baseline_ids[:, -1] = ids[:, 0], ids[:, -1]
        with torch.no_grad():
            embeddings = embedding_layer(ids).detach()
            baseline = embedding_layer(baseline_ids).detach()
        difference = embeddings - baseline
        total_gradients = torch.zeros_like(embeddings)
        with torch.enable_grad():
            for step in range(steps):
                interpolated = (baseline + ((step + 0.5) / steps) * difference
                                ).detach().requires_grad_(True)
                logits = model(inputs_embeds=interpolated,
                               attention_mask=torch.ones_like(ids),
                               token_type_ids=torch.zeros_like(ids)).logits
                gradients = torch.autograd.grad(logits[0, fake_label], interpolated)[0]
                total_gradients += gradients.detach()
        values = (difference * (total_gradients / steps)).sum(dim=-1)
        values = values.squeeze(0).detach().cpu().tolist()[1:-1]
        absolute_total = sum(abs(value) for value in values) or 1.0
        # Preserve original per-chunk scaling even when only a subset is explained.
        chunk_share = 100.0 / len(chunks)
        for token, value, (start, end) in zip(
                tokenizer.convert_ids_to_tokens(chunk["ids"]), values, chunk["offsets"]):
            if start != end:
                token_results.append({"token": token, "text": text[start:end],
                                      "start": start, "end": end, "chunk": chunk["number"],
                                      "raw_attribution": value,
                                      "relative_attribution": round(value / absolute_total * chunk_share, 3)})

    words = []
    for item in token_results:
        if (item["token"].startswith("##") and words
                and item["chunk"] == words[-1]["chunk"]
                and item["start"] == words[-1]["end"]):
            words[-1]["word"] += item["text"]
            words[-1]["end"] = item["end"]
            words[-1]["contribution"] += item["relative_attribution"]
        else:
            words.append({"word": item["text"], "start": item["start"], "end": item["end"],
                          "chunk": item["chunk"], "contribution": item["relative_attribution"]})
    for word in words:
        word["contribution"] = round(word["contribution"], 3)
    meaningful = [word for word in words if re.search(r"[A-Za-z0-9]", word["word"])]
    prediction_tokens = sum(len(chunk["ids"]) for chunk in chunks)
    return {"fake_score": round(sum(scores) / len(scores) * 100, 2),
            "article_threshold": round(float(config["fake_threshold"]) * 100, 2),
            "steps": steps, "chunks_used": len(selected), "prediction_chunks": len(chunks),
            "selected_chunks": [chunks[i]["number"] for i in selected],
            "total_tokens": len(all_ids),
            "analysed_tokens": sum(len(chunks[i]["ids"]) for i in selected),
            "truncated": prediction_tokens < len(all_ids),
            "partial_coverage": len(selected) < len(chunks) or prediction_tokens < len(all_ids),
            "analysed_char_end": max((item["end"] for item in token_results), default=0),
            "chunk_fake_scores": [round(score * 100, 2) for score in scores],
            "scores_reused": bool(reusable), "tokens": token_results, "words": words,
            "positive_words": sorted([w for w in meaningful if w["contribution"] > 0],
                                     key=lambda w: w["contribution"], reverse=True),
            "negative_words": sorted([w for w in meaningful if w["contribution"] < 0],
                                     key=lambda w: w["contribution"])}


def generate_article_highlights(text, prediction_context=None):
    """Cache compact display results. Never store an unbounded article dictionary."""
    context = prediction_context or {}
    key = (hashlib.sha256(text.strip().encode("utf-8")).hexdigest(),
           context.get("model_used"), DEFAULT_STEPS, DEFAULT_MAX_CHUNKS)
    with _cache_lock:
        if key in _highlight_cache:
            _highlight_cache.move_to_end(key)
            return _highlight_cache[key]
    # Avoid multiple CPU-heavy explanations competing during the demonstration.
    if not _generation_lock.acquire(blocking=False):
        raise HighlightsBusy("Another article is being explained. Please try again shortly.")
    try:
        with _cache_lock:
            if key in _highlight_cache:
                return _highlight_cache[key]
        info = attribute_article_text(text, prediction_context=prediction_context)
        highlights = build_influence_highlights(text, info, minimum_net=1.5, maximum_highlights=5)
        summary = {k: v for k, v in info.items()
                   if k not in {"words", "tokens", "positive_words", "negative_words"}}
        result = {"influence_highlights": highlights, "attribution_info": summary,
                  "message": None, "skipped": False}
        with _cache_lock:
            _highlight_cache[key] = result
            while len(_highlight_cache) > _CACHE_LIMIT:
                _highlight_cache.popitem(last=False)
        return result
    finally:
        _generation_lock.release()


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