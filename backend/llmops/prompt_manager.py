"""
Prompt Manager – version-controlled prompt templates.

Active prompt is selected via the ACTIVE_PROMPT env var (default: reasoning_v1).
Templates live in  backend/prompts/<name>.txt  and contain a {context} / {question} pair.
"""

import os
from pathlib import Path
from langchain_core.prompts import ChatPromptTemplate

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


class PromptManager:
    """Load, list, and switch prompt versions."""

    def __init__(self, prompts_dir: Path | None = None):
        self.prompts_dir = prompts_dir or PROMPTS_DIR
        self.prompts_dir.mkdir(parents=True, exist_ok=True)

    # ── helpers ──────────────────────────────────────────────────
    @property
    def active_name(self) -> str:
        return os.getenv("ACTIVE_PROMPT", "reasoning_v1")

    def list_versions(self) -> list[str]:
        """Return sorted list of available prompt names (without .txt)."""
        return sorted(
            p.stem for p in self.prompts_dir.glob("*.txt")
        )

    def _read_template(self, name: str) -> str:
        path = self.prompts_dir / f"{name}.txt"
        if not path.exists():
            raise FileNotFoundError(f"Prompt '{name}' not found at {path}")
        return path.read_text(encoding="utf-8")

    # ── public API ───────────────────────────────────────────────
    def get_prompt(self, name: str | None = None) -> ChatPromptTemplate:
        """
        Return a ChatPromptTemplate for the given (or active) prompt version.
        The .txt file must contain ``{context}`` and ``{question}`` placeholders.
        """
        name = name or self.active_name
        raw = self._read_template(name)
        return ChatPromptTemplate.from_template(raw)

    def get_active_prompt(self) -> ChatPromptTemplate:
        return self.get_prompt(self.active_name)

    def get_raw(self, name: str | None = None) -> str:
        name = name or self.active_name
        return self._read_template(name)
