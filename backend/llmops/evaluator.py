"""
Evaluator – automated evaluation of the RAG chain against a predefined
question set (evaluation/eval_set.json).

Each test case has:
  - question: str
  - must_include: list[str]   (keywords that MUST appear in the answer)

The evaluator runs every question through the chain and checks inclusion.
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

EVAL_DIR = Path(__file__).resolve().parent.parent / "evaluation"
EVAL_SET = EVAL_DIR / "eval_set.json"
REPORTS_DIR = EVAL_DIR / "reports"


class Evaluator:
    """Run eval set through RAG chain and produce scored reports."""

    def __init__(self, eval_set_path: Path | None = None):
        self.eval_set_path = eval_set_path or EVAL_SET
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    def load_eval_set(self) -> list[dict]:
        if not self.eval_set_path.exists():
            return []
        with open(self.eval_set_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def run(self, chain_invoke: Callable[[str], str]) -> dict:
        """
        Run all eval cases through chain_invoke(question) -> answer.

        Returns:
            {
                "timestamp": ...,
                "total": int,
                "passed": int,
                "failed": int,
                "score": float,     # passed / total
                "details": [...]
            }
        """
        cases = self.load_eval_set()
        if not cases:
            return {"error": "No evaluation cases found."}

        details = []
        passed = 0

        for case in cases:
            question = case["question"]
            must_include = [kw.lower() for kw in case.get("must_include", [])]

            t0 = time.time()
            try:
                answer = chain_invoke(question)
            except Exception as exc:
                answer = f"[ERROR] {exc}"
            latency_ms = (time.time() - t0) * 1000

            answer_lower = answer.lower()
            hits = [kw for kw in must_include if kw in answer_lower]
            misses = [kw for kw in must_include if kw not in answer_lower]
            ok = len(misses) == 0

            if ok:
                passed += 1

            details.append({
                "question": question,
                "answer": answer,
                "must_include": must_include,
                "hits": hits,
                "misses": misses,
                "passed": ok,
                "latency_ms": round(latency_ms, 2),
            })

        total = len(cases)
        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total": total,
            "passed": passed,
            "failed": total - passed,
            "score": round(passed / total, 4) if total else 0.0,
            "details": details,
        }

        # persist report
        fname = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(REPORTS_DIR / fname, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        return report
