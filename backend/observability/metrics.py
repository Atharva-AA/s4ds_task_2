"""
Analytics functions that query the Chroma telemetry collections.

All functions are safe — they return a default value if telemetry is
unavailable or if Chroma raises an exception.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger("uvicorn.error")


def get_hallucination_rate(last_n: int = 100) -> dict:
    """
    Return the fraction of recent queries flagged as hallucinations.

    Returns:
        {
            "hallucination_rate": float,   # 0.0 – 1.0
            "total_evaluated": int,
            "hallucinated": int,
        }
    """
    try:
        from .logger import get_telemetry
        tel = get_telemetry()
        if tel is None:
            return _unavailable("hallucination_rate")

        evals = tel.get_evals(last_n=last_n)
        if not evals:
            return {"hallucination_rate": 0.0, "total_evaluated": 0, "hallucinated": 0}

        hallucinated = sum(1 for e in evals if e.get("is_hallucination", False))
        return {
            "hallucination_rate": round(hallucinated / len(evals), 4),
            "total_evaluated": len(evals),
            "hallucinated": hallucinated,
        }
    except Exception as exc:
        logger.debug(f"[observability] get_hallucination_rate error: {exc}")
        return _unavailable("hallucination_rate")


def get_avg_latency(last_n: int = 100) -> dict:
    """
    Return average response latency over the last N queries.

    Returns:
        {
            "avg_latency_ms": float,
            "min_latency_ms": float,
            "max_latency_ms": float,
            "total_responses": int,
        }
    """
    try:
        from .logger import get_telemetry
        tel = get_telemetry()
        if tel is None:
            return _unavailable("avg_latency_ms")

        responses = tel.get_responses(last_n=last_n)
        if not responses:
            return {
                "avg_latency_ms": 0.0,
                "min_latency_ms": 0.0,
                "max_latency_ms": 0.0,
                "total_responses": 0,
            }

        latencies = [r["latency_ms"] for r in responses if "latency_ms" in r]
        if not latencies:
            return {
                "avg_latency_ms": 0.0,
                "min_latency_ms": 0.0,
                "max_latency_ms": 0.0,
                "total_responses": 0,
            }

        return {
            "avg_latency_ms": round(sum(latencies) / len(latencies), 2),
            "min_latency_ms": round(min(latencies), 2),
            "max_latency_ms": round(max(latencies), 2),
            "total_responses": len(latencies),
        }
    except Exception as exc:
        logger.debug(f"[observability] get_avg_latency error: {exc}")
        return _unavailable("avg_latency_ms")


def get_retrieval_failure_rate(last_n: int = 100, distance_threshold: float = 0.8) -> dict:
    """
    Return the fraction of retrievals where avg_distance exceeded the threshold
    (indicating poor retrieval quality / likely retrieval failure).

    Returns:
        {
            "retrieval_failure_rate": float,
            "total_retrievals": int,
            "failed_retrievals": int,
            "distance_threshold": float,
        }
    """
    try:
        from .logger import get_telemetry
        tel = get_telemetry()
        if tel is None:
            return _unavailable("retrieval_failure_rate")

        logs = tel.get_retrieval_logs(last_n=last_n)
        if not logs:
            return {
                "retrieval_failure_rate": 0.0,
                "total_retrievals": 0,
                "failed_retrievals": 0,
                "distance_threshold": distance_threshold,
            }

        failed = sum(
            1 for r in logs
            if r.get("avg_distance", 0.0) > distance_threshold
        )
        return {
            "retrieval_failure_rate": round(failed / len(logs), 4),
            "total_retrievals": len(logs),
            "failed_retrievals": failed,
            "distance_threshold": distance_threshold,
        }
    except Exception as exc:
        logger.debug(f"[observability] get_retrieval_failure_rate error: {exc}")
        return _unavailable("retrieval_failure_rate")


def _unavailable(key: str) -> dict:
    return {key: None, "error": "telemetry_unavailable"}
