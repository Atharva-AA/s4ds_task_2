"""
Reindex Pipeline – hash-based change detection for PDF documents.

Only rebuilds the Chroma vector store when the set of PDFs (or their
contents) has actually changed, saving time & compute on startup or
scheduled runs.
"""

import glob
import hashlib
import json
import os
from pathlib import Path
from typing import Callable, Optional

HASH_FILE = Path(__file__).resolve().parent.parent / "chroma_store" / "doc_hashes.json"


class ReindexPipeline:
    """Detect document changes and conditionally reindex."""

    def __init__(
        self,
        data_dir: str,
        chroma_dir: str,
        build_fn: Callable,
        hash_file: Path | None = None,
    ):
        """
        Args:
            data_dir:   folder containing PDF files
            chroma_dir: Chroma persistence directory
            build_fn:   callable(docs) -> vector_store  (the existing build_vector_store)
            hash_file:  optional path to store document hashes
        """
        self.data_dir = data_dir
        self.chroma_dir = chroma_dir
        self.build_fn = build_fn
        self.hash_file = hash_file or HASH_FILE

    # ── hashing ──────────────────────────────────────────────────
    def _compute_hashes(self) -> dict[str, str]:
        """Return {filename: sha256} for all PDFs in data_dir."""
        hashes = {}
        for pdf_path in sorted(glob.glob(os.path.join(self.data_dir, "*.pdf"))):
            sha = hashlib.sha256()
            with open(pdf_path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    sha.update(chunk)
            hashes[os.path.basename(pdf_path)] = sha.hexdigest()
        return hashes

    def _load_stored_hashes(self) -> dict[str, str]:
        if not self.hash_file.exists():
            return {}
        with open(self.hash_file, "r") as f:
            return json.load(f)

    def _save_hashes(self, hashes: dict[str, str]):
        self.hash_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.hash_file, "w") as f:
            json.dump(hashes, f, indent=2)

    # ── public API ───────────────────────────────────────────────
    def needs_reindex(self) -> bool:
        """Return True if documents have changed since last index."""
        current = self._compute_hashes()
        stored = self._load_stored_hashes()
        return current != stored

    def reindex_if_needed(self, docs) -> tuple[bool, Optional[object]]:
        """
        Reindex only when document hashes differ.

        Args:
            docs: list of loaded langchain Documents (from load_pdfs)

        Returns:
            (reindexed: bool, vector_store_or_None)
        """
        current_hashes = self._compute_hashes()
        stored_hashes = self._load_stored_hashes()

        if current_hashes == stored_hashes:
            print("[reindex] Documents unchanged – skipping rebuild.")
            return False, None

        print("[reindex] Documents changed – rebuilding vector store ...")
        vs = self.build_fn(docs)
        self._save_hashes(current_hashes)
        print("[reindex] Done. Hashes saved.")
        return True, vs

    def force_reindex(self, docs) -> object:
        """Always rebuild, regardless of hashes."""
        vs = self.build_fn(docs)
        self._save_hashes(self._compute_hashes())
        return vs
