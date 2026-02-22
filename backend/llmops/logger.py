"""Query logging — prints every RAG interaction to the console."""

import uuid
from datetime import datetime, timezone


def log_query(
    question: str,
    answer: str,
    context: str,
    latency_ms: float,
    grounding_score: float,
    hallucination_score: float,
    prompt_version: str = "default",
    blocked: bool = False,
) -> str:
    """Print a single RAG interaction to stdout. Returns the query_id."""
    query_id = str(uuid.uuid4())
    print(f"\n{'='*60}")
    print(f"[QUERY LOG] {datetime.now(timezone.utc).isoformat()}")
    print(f"  ID:                 {query_id}")
    print(f"  Question:           {question}")
    print(f"  Answer:             {answer[:200]}{'...' if len(answer) > 200 else ''}")
    print(f"  Latency:            {latency_ms:.2f} ms")
    print(f"  Grounding Score:    {grounding_score:.4f}")
    print(f"  Hallucination Score:{hallucination_score:.4f}")
    print(f"  Prompt Version:     {prompt_version}")
    print(f"  Blocked:            {blocked}")
    print(f"{'='*60}\n")
    return query_id


def get_recent_logs(n: int = 50) -> list[dict]:
    """Not applicable — logs are only printed to console."""
    return []


def get_log_stats() -> dict:
    """Not applicable — logs are only printed to console."""
    return {"total_queries": 0, "note": "Logs are printed to console only."}
