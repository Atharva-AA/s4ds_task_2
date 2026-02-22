"""Lightweight grounding & hallucination metrics (token-overlap based)."""

import re


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer, lowercased."""
    return re.findall(r"\b\w+\b", text.lower())


# Common English stop words to exclude from grounding computation
_STOP_WORDS = frozenset(
    "a an the is are was were be been being have has had do does did will "
    "would shall should may might can could of in to for on with at by from "
    "as into through during before after above below between out off over "
    "under again further then once here there when where why how all each "
    "every both few more most other some such no nor not only own same so "
    "than too very and but or if while that this these those it its i he "
    "she they them their we you your his her about also".split()
)


def compute_grounding_score(answer: str, context: str) -> float:
    """Token-overlap ratio: fraction of non-stopword answer tokens in context.
    Returns float in [0, 1]. Higher = better grounded."""
    answer_tokens = [t for t in _tokenize(answer) if t not in _STOP_WORDS]
    if not answer_tokens:
        return 1.0  # empty answer is trivially grounded
    context_tokens = set(_tokenize(context))
    overlap = sum(1 for t in answer_tokens if t in context_tokens)
    return overlap / len(answer_tokens)


def detect_hallucination(answer: str, context: str) -> float:
    """Estimate hallucination risk.  Returns float in [0, 1].
    Higher = more likely hallucinated."""
    grounding = compute_grounding_score(answer, context)

    # Penalty for hedging / fabrication markers
    HALLUCINATION_MARKERS = [
        "i think",
        "probably",
        "it is possible that",
        "generally speaking",
        "in most cases",
        "typically",
        "as far as i know",
        "i believe",
        "it seems like",
        "i would guess",
    ]
    lower_answer = answer.lower()
    marker_hits = sum(1 for m in HALLUCINATION_MARKERS if m in lower_answer)
    marker_penalty = min(marker_hits * 0.05, 0.2)

    # Check for "no info" disclaimer — if the model correctly declines, score is low
    NO_INFO_PHRASES = [
        "i don't have enough information",
        "i cannot find the answer",
        "not mentioned in the context",
    ]
    if any(p in lower_answer for p in NO_INFO_PHRASES):
        return 0.0

    score = (1.0 - grounding) * 0.8 + marker_penalty
    return min(max(score, 0.0), 1.0)
