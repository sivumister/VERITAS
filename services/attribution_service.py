"""Approximate local token attribution for VERITAS BERT classifiers.

This explains the *model's output*, not whether news is true. Integrated
Gradients is approximated using midpoint samples. For full articles, a small
number of informative chunks is explained to keep CPU use reasonable.
"""

import hashlib
import math
import re
from collections import OrderedDict
from threading import Lock

import torch

from services.model_service import load_model

DEFAULT_STEPS = 8
DEFAULT_MAX_CHUNKS = 2
ATTRIBUTION_VERSION = "predicted-class-v3-word-fallback"
_CACHE_LIMIT = 32
_highlight_cache = OrderedDict()
_cache_lock = Lock()
_generation_lock = Lock()


class HighlightsBusy(RuntimeError):
    pass


def _chunk_probabilities(model, ids, fake_label, device):
    """Return the original model's FAKE probability for a chunk."""
    ids = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        scores = model(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            token_type_ids=torch.zeros_like(ids),
        ).logits
        return float(torch.softmax(scores, dim=-1)[0, fake_label].item())


def attribute_text(text, input_type="article", prediction="FAKE",
                   steps=DEFAULT_STEPS, max_explanation_chunks=DEFAULT_MAX_CHUNKS,
                   prediction_context=None):
    """Explain the predicted-class logit for claim, headline, or article text.

    Prediction and label names are not inferred from the model's raw argmax;
    predictions can use the stored, model-specific FAKE probability threshold.
    """
    text = text.strip()
    model_type = "article" if input_type == "url" else input_type
    target_class = prediction.upper()
    if model_type not in {"article", "claim", "headline"}:
        raise ValueError("Unknown input type for explanation.")
    if target_class not in {"FAKE", "REAL"}:
        raise ValueError("Prediction must be FAKE or REAL.")
    if not (1 <= steps <= 64 and 1 <= max_explanation_chunks <= 10):
        raise ValueError("Invalid attribution settings.")
    if not text:
        raise ValueError("Text cannot be empty.")

    tokenizer, model, config = load_model(model_type)
    model.eval()
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("Fast tokenization is required for accurate text offsets.")
    fake_label = int(config["fake_label"])
    real_label = int(config["real_label"])
    target_label = fake_label if target_class == "FAKE" else real_label
    device = next(model.parameters()).device

    if model_type == "article":
        encoded = tokenizer(text, add_special_tokens=False,
                            return_attention_mask=False,
                            return_token_type_ids=False,
                            return_offsets_mapping=True)
        all_ids = encoded["input_ids"]
        offsets = encoded["offset_mapping"]
        chunk_size = int(config["chunk_size"])
        max_chunks = int(config["max_chunks"])
        chunks = [
            {"ids": all_ids[start:start + chunk_size],
             "offsets": offsets[start:start + chunk_size], "number": i + 1}
            for i, start in enumerate(range(0, min(len(all_ids), chunk_size * max_chunks), chunk_size))
        ]
    else:
        # The short-text prediction uses truncation at its saved max_length.
        encoded = tokenizer(text, truncation=True, padding=True,
                            max_length=int(config["max_length"]),
                            return_offsets_mapping=True)
        input_ids = encoded["input_ids"]
        input_offsets = encoded["offset_mapping"]
        if not input_ids or len(input_ids) < 3:
            raise ValueError("No usable tokens in the submitted text.")
        # Strip special tokens; we reconstruct [CLS]/[SEP] exactly as inference.
        special_mask = tokenizer.get_special_tokens_mask(input_ids, already_has_special_tokens=True)
        included = [(token, offset) for token, offset, special in
                    zip(input_ids, input_offsets, special_mask) if not special]
        all_ids = [token for token, _ in included]
        offsets = [offset for _, offset in included]
        chunks = [{"ids": all_ids, "offsets": offsets, "number": 1}]
    if not chunks or not chunks[0]["ids"]:
        raise ValueError("The submitted text produced no usable tokens.")

    # Reuse stored article probabilities only when the saved inference context matches.
    context = prediction_context or {}
    stored_scores = context.get("chunk_fake_probabilities", [])
    reusable = (model_type == "article"
                and context.get("text_sha256") == hashlib.sha256(text.encode("utf-8")).hexdigest()
                and context.get("model_used") == config["model_type"]
                and context.get("fake_label") == fake_label
                and context.get("chunk_size") == chunk_size
                and context.get("max_chunks") == max_chunks
                and context.get("total_tokens") == len(all_ids)
                and len(stored_scores) == len(chunks)
                and all(isinstance(s, (int, float)) and math.isfinite(s)
                        and 0 <= s <= 1 for s in stored_scores))
    scores = stored_scores if reusable else [
        _chunk_probabilities(model, [tokenizer.cls_token_id, *chunk["ids"], tokenizer.sep_token_id],
                             fake_label, device)
        for chunk in chunks
    ]

    if model_type == "article":
        # Select the chunks strongest for the *displayed* class, not always FAKE.
        direction = -1 if target_class == "FAKE" else 1
        selected = sorted(sorted(range(len(chunks)), key=lambda i: direction * scores[i])
                          [:max_explanation_chunks])
    else:
        selected = [0]

    embedding_layer = model.get_input_embeddings()
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        raise ValueError("Tokenizer needs a pad token for attribution baseline.")
    token_results = []
    for idx in selected:
        chunk = chunks[idx]
        ids = [tokenizer.cls_token_id, *chunk["ids"], tokenizer.sep_token_id]
        # Avoid mutating model state; keep the explicit baseline in a helper.
        contributions = _attributions_for_ids(model, embedding_layer, ids,
                                              target_label, pad_id, device, steps)
        total_abs = sum(abs(x) for x in contributions) or 1.0
        chunk_share = 100.0 / len(chunks)
        for token, value, (start, end) in zip(
                tokenizer.convert_ids_to_tokens(chunk["ids"]), contributions, chunk["offsets"]):
            if start != end:
                token_results.append({
                    "token": token, "text": text[start:end], "start": start, "end": end,
                    "chunk": chunk["number"], "raw_attribution": value,
                    "relative_attribution": round(value / total_abs * chunk_share, 3),
                })

    words = []
    for item in token_results:
        if (item["token"].startswith("##") and words
                and words[-1]["chunk"] == item["chunk"]
                and words[-1]["end"] == item["start"]):
            words[-1]["word"] += item["text"]
            words[-1]["end"] = item["end"]
            words[-1]["contribution"] += item["relative_attribution"]
        else:
            words.append({"word": item["text"], "start": item["start"],
                          "end": item["end"], "chunk": item["chunk"],
                          "contribution": item["relative_attribution"]})
    for word in words:
        word["contribution"] = round(word["contribution"], 3)
    meaningful = [w for w in words if re.search(r"[A-Za-z0-9]", w["word"])]
    prediction_tokens = sum(len(c["ids"]) for c in chunks)
    return {
        "version": ATTRIBUTION_VERSION, "target_class": target_class,
        "input_type": model_type, "fake_score": round(sum(scores) / len(scores) * 100, 2),
        "article_threshold": round(float(config["fake_threshold"]) * 100, 2)
            if model_type == "article" else None,
        "steps": steps, "chunks_used": len(selected), "prediction_chunks": len(chunks),
        "selected_chunks": [chunks[i]["number"] for i in selected],
        "total_tokens": len(all_ids), "analysed_tokens": sum(len(chunks[i]["ids"]) for i in selected),
        "truncated": prediction_tokens < len(all_ids),
        "partial_coverage": len(selected) < len(chunks) or prediction_tokens < len(all_ids)
            or (model_type != "article" and len(text) > max((e for _, e in offsets), default=0)),
        "analysed_char_end": max((r["end"] for r in token_results), default=0),
        "chunk_fake_scores": [round(s * 100, 2) for s in scores],
        "scores_reused": bool(reusable), "tokens": token_results, "words": words,
        "positive_words": sorted((w for w in meaningful if w["contribution"] > 0),
                                 key=lambda w: w["contribution"], reverse=True),
        "negative_words": sorted((w for w in meaningful if w["contribution"] < 0),
                                 key=lambda w: w["contribution"]),
    }


def _attributions_for_ids(model, embedding_layer, ids, target_label, pad_id, device, steps):
    tensor_ids = torch.tensor([ids], dtype=torch.long, device=device)
    baseline_ids = torch.full_like(tensor_ids, pad_id)
    baseline_ids[:, 0] = tensor_ids[:, 0]
    baseline_ids[:, -1] = tensor_ids[:, -1]
    with torch.no_grad():
        inputs = embedding_layer(tensor_ids).detach()
        baseline = embedding_layer(baseline_ids).detach()
    difference = inputs - baseline
    total_gradients = torch.zeros_like(inputs)
    with torch.enable_grad():
        for step in range(steps):
            interpolated = (baseline + (step + 0.5) / steps * difference).detach().requires_grad_(True)
            logits = model(inputs_embeds=interpolated,
                           attention_mask=torch.ones_like(tensor_ids),
                           token_type_ids=torch.zeros_like(tensor_ids)).logits
            grad = torch.autograd.grad(logits[0, target_label], interpolated)[0]
            total_gradients += grad.detach()
    return ((difference * (total_gradients / steps)).sum(-1)
            .squeeze(0).detach().cpu().tolist()[1:-1])


def attribute_article_text(text, steps=DEFAULT_STEPS,
                           max_explanation_chunks=DEFAULT_MAX_CHUNKS,
                           prediction_context=None, prediction="FAKE"):
    """Compatibility with earlier article-attribution callers."""
    return attribute_text(text, input_type="article", prediction=prediction, steps=steps,
                          max_explanation_chunks=max_explanation_chunks,
                          prediction_context=prediction_context)


def generate_text_highlights(text, input_type="article", prediction="FAKE",
                             prediction_context=None):
    """Generate compact, lazily computed highlights for the displayed class."""
    model_type = "article" if input_type == "url" else input_type
    context = prediction_context or {}
    key = (hashlib.sha256(text.strip().encode("utf-8")).hexdigest(), model_type,
           prediction.upper(), context.get("model_used"), DEFAULT_STEPS,
           DEFAULT_MAX_CHUNKS, ATTRIBUTION_VERSION)
    with _cache_lock:
        if key in _highlight_cache:
            _highlight_cache.move_to_end(key)
            return _highlight_cache[key]
    if not _generation_lock.acquire(blocking=False):
        raise HighlightsBusy("Another explanation is running. Please try again shortly.")
    try:
        with _cache_lock:
            if key in _highlight_cache:
                _highlight_cache.move_to_end(key)
                return _highlight_cache[key]
        info = attribute_text(text, input_type=model_type, prediction=prediction,
                              prediction_context=prediction_context)
        highlights = build_influence_highlights(text, info, minimum_net=0.5,
                                                maximum_highlights=5)
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


def generate_article_highlights(text, prediction_context=None, prediction="FAKE"):
    """Compatibility with existing callers."""
    return generate_text_highlights(text, "article", prediction, prediction_context)


def build_influence_highlights(text, attribution_result,
                               minimum_net=1.5, maximum_highlights=5):
    """Convert token attribution into human-readable sentence spans."""
    text = text.strip()
    if not text:
        return {"segments": [], "highlight_count": 0}
    sentences = []
    # Preserve spacing and order in the original submitted text.
    for match in re.finditer(r"[^.!?\n]+[.!?]?|\n+", text):
        raw = match.group(0)
        if not raw.strip():
            continue
        start = match.start() + len(raw) - len(raw.lstrip())
        end = match.end() - (len(raw) - len(raw.rstrip()))
        sentences.append({"text": text[start:end], "start": start, "end": end})
    words = attribution_result.get("words", [])
    for sentence in sentences:
        values = [word["contribution"] for word in words
                  if sentence["start"] <= word["start"] < sentence["end"]]
        net = round(sum(values), 3)
        sentence.update(
            positive_contribution=round(sum(v for v in values if v > 0), 3),
            negative_contribution=round(sum(v for v in values if v < 0), 3),
            net_contribution=net, analysed=bool(values), highlighted=False, signal="NONE",
        )
    eligible = sorted((s for s in sentences if s["analysed"] and
                       s["net_contribution"] >= minimum_net),
                      key=lambda s: s["net_contribution"], reverse=True)
    selected_starts = {s["start"] for s in eligible[:maximum_highlights]}

    # If every complete sentence falls below the minimum, show genuinely
    # positive *word-level* attributions rather than inventing a strong passage.
    # These are deliberately labelled LOW: they are weaker local signals and
    # should not be presented as decisive evidence for the prediction.
    if not selected_starts:
        positive_words = sorted(
            (w for w in words
             if w.get("contribution", 0) >= 0.005
             and 0 <= w.get("start", -1) < w.get("end", -1) <= len(text)
             and re.search(r"[A-Za-z0-9]", text[w["start"]:w["end"]])),
            key=lambda w: w["contribution"], reverse=True
        )
        chosen = []
        for word in positive_words:
            # Exclude overlapping word-pieces and keep source offsets exact.
            if all(word["end"] <= other["start"] or word["start"] >= other["end"]
                   for other in chosen):
                chosen.append(word)
            if len(chosen) >= maximum_highlights:
                break
        if chosen:
            segments = []
            cursor = 0
            for word in sorted(chosen, key=lambda w: w["start"]):
                start, end = word["start"], word["end"]
                if start > cursor:
                    segments.append({"text": text[cursor:start], "highlighted": False,
                                     "signal": "NONE", "net_contribution": 0})
                segments.append({"text": text[start:end], "highlighted": True,
                                 "signal": "LOW", "net_contribution": round(word["contribution"], 3)})
                cursor = end
            if cursor < len(text):
                segments.append({"text": text[cursor:], "highlighted": False,
                                 "signal": "NONE", "net_contribution": 0})
            return {"segments": segments, "highlight_count": len(chosen),
                    "minimum_net": minimum_net, "maximum_highlights": maximum_highlights,
                    "display_mode": "weak_word_fallback",
                    "display_note": "No complete passage met the standard threshold. The highlighted words have smaller positive contributions to the predicted-class output and should be interpreted cautiously."}

    for sentence in sentences:
        if sentence["start"] in selected_starts:
            sentence["highlighted"] = True
            v = sentence["net_contribution"]
            sentence["signal"] = "HIGH" if v >= 8 else "MODERATE" if v >= 3 else "LOW"
    return {"segments": sentences, "highlight_count": len(selected_starts),
            "minimum_net": minimum_net, "maximum_highlights": maximum_highlights,
            "display_mode": "sentence", "display_note": None}
