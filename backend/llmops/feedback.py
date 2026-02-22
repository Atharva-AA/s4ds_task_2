"""
Feedback Store – persists user feedback (thumbs-up / thumbs-down) as JSONL.
"""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
FEEDBACK_FILE = LOG_DIR / "feedback.jsonl"


class FeedbackStore:
    """Append-only JSONL store for user feedback."""

    def __init__(self, path: Optional[Path] = None):
        self.path = path or FEEDBACK_FILE
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def add(
        self,
        *,
        query_id: str,
        rating: Literal["up", "down"],
        comment: str = "",
    ) -> dict:
        record = {
            "id": uuid.uuid4().hex[:12],
            "query_id": query_id,
            "rating": rating,
            "comment": comment,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record

    def read_all(self) -> list[dict]:
        if not self.path.exists():
            return []
        records = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    def summary(self) -> dict:
        """Return counts of up / down ratings."""
        records = self.read_all()
        up = sum(1 for r in records if r["rating"] == "up")
        down = sum(1 for r in records if r["rating"] == "down")
        return {"total": len(records), "up": up, "down": down}
