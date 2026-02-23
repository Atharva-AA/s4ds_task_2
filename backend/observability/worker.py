"""
Async fire-and-forget background worker for telemetry events.

Pattern:
    request thread → enqueue_event(event) → asyncio.Queue → worker writes to Chroma

The main request NEVER awaits Chroma writes.
Queue overflow silently drops events — never blocks callers.
"""

import asyncio
import logging
from typing import Any

logger = logging.getLogger("uvicorn.error")

_MAX_QUEUE_SIZE = 500
_queue: asyncio.Queue[dict[str, Any]] | None = None
_worker_task: asyncio.Task | None = None
_telemetry_disabled = False


def enqueue_event(event: dict[str, Any]) -> None:
    """Push a telemetry event onto the queue. Silently drops if full or disabled."""
    global _telemetry_disabled
    if _telemetry_disabled or _queue is None:
        return
    try:
        _queue.put_nowait(event)
    except asyncio.QueueFull:
        pass
    except Exception:
        pass


async def _worker_loop() -> None:
    """Background coroutine that drains the queue and writes to Chroma."""
    global _telemetry_disabled

    from .logger import get_telemetry

    while True:
        try:
            event: dict[str, Any] = await _queue.get()
            try:
                tel = get_telemetry()
                if tel is None:
                    _queue.task_done()
                    continue

                kind = event.get("kind")

                if kind == "query":
                    tel.log_query(
                        qid=event["qid"],
                        text=event["text"],
                        timestamp=event["timestamp"],
                    )
                elif kind == "retrieval":
                    tel.log_retrieval(
                        qid=event["qid"],
                        avg_distance=event["avg_distance"],
                        k=event["k"],
                    )
                elif kind == "response":
                    tel.log_response(
                        qid=event["qid"],
                        answer=event["answer"],
                        latency_ms=event["latency_ms"],
                        tokens=event.get("tokens", 0),
                    )
                elif kind == "eval":
                    tel.log_eval(
                        qid=event["qid"],
                        groundedness=event["groundedness"],
                        hallucination_score=event["hallucination_score"],
                        blocked=event["blocked"],
                    )

            except Exception as exc:
                logger.debug(f"[observability] worker write error: {exc}")
            finally:
                _queue.task_done()

        except asyncio.CancelledError:
            break
        except Exception as exc:
            logger.debug(f"[observability] worker loop error: {exc}")


async def start_worker() -> None:
    """Create the queue and start the background worker. Call from app lifespan startup."""
    global _queue, _worker_task, _telemetry_disabled
    try:
        _queue = asyncio.Queue(maxsize=_MAX_QUEUE_SIZE)
        _worker_task = asyncio.create_task(_worker_loop())
        logger.info("[observability] background worker started")
    except Exception as exc:
        _telemetry_disabled = True
        logger.warning(f"[observability] failed to start worker, telemetry disabled: {exc}")


async def stop_worker() -> None:
    """Cancel the background worker. Call from app lifespan shutdown."""
    global _worker_task
    if _worker_task and not _worker_task.done():
        try:
            _worker_task.cancel()
            await asyncio.gather(_worker_task, return_exceptions=True)
            logger.info("[observability] background worker stopped")
        except Exception:
            pass
