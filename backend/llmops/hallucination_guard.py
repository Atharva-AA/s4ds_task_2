"""Hallucination guard — blocks or flags answers above a score threshold."""

from .metrics import detect_hallucination

THRESHOLD = 0.4

BLOCKED_MSG = (
    "I'm not confident enough in my answer based on the available documents. "
    "Please try rephrasing your question or uploading more relevant documents."
)


def guard(
    answer: str,
    context: str,
    threshold: float = THRESHOLD,
) -> dict:
    """Check the answer for hallucination.

    Returns:
        {
            "answer": str,        # possibly replaced with BLOCKED_MSG
            "blocked": bool,
            "hallucination_score": float,
            "original_answer": str | None,  # set only when blocked
        }
    """
    score = detect_hallucination(answer, context)
    if score > threshold:
        return {
            "answer": BLOCKED_MSG,
            "blocked": True,
            "hallucination_score": round(score, 4),
            "original_answer": answer,
        }
    return {
        "answer": answer,
        "blocked": False,
        "hallucination_score": round(score, 4),
        "original_answer": None,
    }
