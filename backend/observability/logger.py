"""
ChromaTelemetry — singleton that owns four Chroma telemetry collections:

    queries        — user questions
    retrieval_logs — similarity search stats
    responses      — LLM outputs + latency
    evals          — groundedness / hallucination scores

All public methods are safe: they catch every exception internally and
never propagate errors to callers.

The embedding model is injected at initialisation time so it is NEVER
reloaded here — the same instance used by the RAG pipeline is reused.
"""

import logging
import os
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger("uvicorn.error")

_telemetry_instance: Optional["ChromaTelemetry"] = None
_telemetry_disabled: bool = False

CHROMA_TELEMETRY_DIR = os.path.join(os.path.dirname(__file__), "..", "chroma_telemetry")


def get_telemetry() -> Optional["ChromaTelemetry"]:
    """Return the singleton ChromaTelemetry, or None if disabled."""
    if _telemetry_disabled:
        return None
    return _telemetry_instance


def init_telemetry(embedding_function) -> None:
    """
    Initialise the ChromaTelemetry singleton.

    Must be called once at app startup, AFTER the embedding model is loaded.
    Passing the already-loaded embedding_function avoids reloading the model.
    """
    global _telemetry_instance, _telemetry_disabled
    try:
        _telemetry_instance = ChromaTelemetry(embedding_function=embedding_function)
        logger.info("[observability] ChromaTelemetry initialised")
    except Exception as exc:
        _telemetry_disabled = True
        logger.warning(f"[observability] ChromaTelemetry init failed, telemetry disabled: {exc}")


class ChromaTelemetry:
    """Manages four Chroma collections for LLM observability."""

    def __init__(self, embedding_function) -> None:
        import chromadb
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

        os.makedirs(CHROMA_TELEMETRY_DIR, exist_ok=True)
        self._client = chromadb.PersistentClient(path=CHROMA_TELEMETRY_DIR)
        self._ef = embedding_function

        self._queries = self._client.get_or_create_collection(
            name="queries",
            metadata={"hnsw:space": "cosine"},
        )
        self._retrieval_logs = self._client.get_or_create_collection(
            name="retrieval_logs",
        )
        self._responses = self._client.get_or_create_collection(
            name="responses",
        )
        self._evals = self._client.get_or_create_collection(
            name="evals",
        )

    # ── public logging methods ────────────────────────────────────────

    def log_query(self, qid: str, text: str, timestamp: str) -> None:
        """Store user question text."""
        try:
            embedding = self._embed(text)
            self._queries.add(
                ids=[qid],
                embeddings=[embedding],
                documents=[text],
                metadatas=[{"timestamp": timestamp}],
            )
        except Exception as exc:
            logger.debug(f"[observability] log_query error: {exc}")

    def log_retrieval(self, qid: str, avg_distance: float, k: int) -> None:
        """Store retrieval stats for a query."""
        try:
            self._retrieval_logs.add(
                ids=[qid],
                documents=[qid],
                metadatas=[{
                    "avg_distance": round(avg_distance, 6),
                    "k": k,
                    "timestamp": _now(),
                }],
            )
        except Exception as exc:
            logger.debug(f"[observability] log_retrieval error: {exc}")

    def log_response(self, qid: str, answer: str, latency_ms: float, tokens: int) -> None:
        """Store LLM answer and latency."""
        try:
            self._responses.add(
                ids=[qid],
                documents=[answer[:2000]],
                metadatas=[{
                    "latency_ms": round(latency_ms, 2),
                    "tokens": tokens,
                    "timestamp": _now(),
                }],
            )
        except Exception as exc:
            logger.debug(f"[observability] log_response error: {exc}")

    def log_eval(
        self,
        qid: str,
        groundedness: float,
        hallucination_score: float,
        blocked: bool,
    ) -> None:
        """Store groundedness / hallucination eval for a query."""
        try:
            self._evals.add(
                ids=[qid],
                documents=[qid],
                metadatas=[{
                    "groundedness": round(groundedness, 6),
                    "hallucination_score": round(hallucination_score, 6),
                    "is_hallucination": is_hallucination(groundedness),
                    "blocked": blocked,
                    "timestamp": _now(),
                }],
            )
        except Exception as exc:
            logger.debug(f"[observability] log_eval error: {exc}")

    # ── internal helpers ─────────────────────────────────────────────

    def _embed(self, text: str) -> list[float]:
        """Embed text using the injected embedding function."""
        result = self._ef.embed_query(text)
        return result

    # ── raw collection accessors (used by metrics.py) ────────────────

    def get_evals(self, last_n: int = 100) -> list[dict]:
        try:
            result = self._evals.get(
                limit=last_n,
                include=["metadatas"],
            )
            return result.get("metadatas") or []
        except Exception:
            return []

    def get_responses(self, last_n: int = 100) -> list[dict]:
        try:
            result = self._responses.get(
                limit=last_n,
                include=["metadatas"],
            )
            return result.get("metadatas") or []
        except Exception:
            return []

    def get_retrieval_logs(self, last_n: int = 100) -> list[dict]:
        try:
            result = self._retrieval_logs.get(
                limit=last_n,
                include=["metadatas"],
            )
            return result.get("metadatas") or []
        except Exception:
            return []


# ── module-level helpers ──────────────────────────────────────────────

def is_hallucination(groundedness_score: float) -> bool:
    """Return True if groundedness score is below the hallucination threshold."""
    return groundedness_score < 0.45


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
