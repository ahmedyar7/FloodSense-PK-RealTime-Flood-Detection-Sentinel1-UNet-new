"""
Structural, token-aware chunking for the RAG ingestion pipeline.

Implements a hierarchical :class:`RecursiveCharacterTextSplitter` (in the spirit
of LangChain's splitter, but dependency-free) that prefers to break text on the
largest natural boundary that still fits the target size — paragraph, then line,
then sentence, then word — so that tables, JSON schemas and individual sections
are kept intact wherever possible rather than being cut down the middle.

Sizes are measured in *tokens* using a lightweight heuristic (~4 characters per
token) which avoids pulling in a heavyweight tokeniser while keeping the
configured ``chunk_size`` / ``chunk_overlap`` in the 500–1000 / 100 token range
requested by the knowledge-pipeline spec.
"""

from __future__ import annotations

import re
from typing import Callable

# Default separators, tried in order from coarsest (paragraph) to finest
# (character). Tabular rows in this corpus are newline-delimited, so splitting
# on line breaks before sentences keeps table rows together.
DEFAULT_SEPARATORS: list[str] = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""]

# Target chunk sizing, expressed in (estimated) tokens.
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 100

# Average characters per token for English prose. Good enough for budgeting
# without instantiating a real tokeniser.
_CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    """Cheaply estimate the token count of ``text`` (~4 chars per token)."""
    if not text:
        return 0
    return max(1, len(text) // _CHARS_PER_TOKEN)


class RecursiveCharacterTextSplitter:
    """Recursively split text on a priority list of separators."""

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        separators: list[str] | None = None,
        length_function: Callable[[str], int] = estimate_tokens,
    ) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or DEFAULT_SEPARATORS
        self.length_function = length_function

    def split_text(self, text: str) -> list[str]:
        """Split ``text`` into chunks no larger than ``chunk_size`` tokens."""
        return self._split(text, self.separators)

    def _split(self, text: str, separators: list[str]) -> list[str]:
        final: list[str] = []

        # Pick the first separator that actually occurs in this text; anything
        # finer is reserved for splitting oversized pieces further down.
        separator = separators[-1]
        remaining: list[str] = []
        for i, sep in enumerate(separators):
            if sep == "":
                separator = sep
                remaining = []
                break
            if re.search(re.escape(sep), text):
                separator = sep
                remaining = separators[i + 1 :]
                break

        splits = list(text) if separator == "" else text.split(separator)

        good: list[str] = []
        for piece in splits:
            if self.length_function(piece) < self.chunk_size:
                good.append(piece)
                continue
            # Flush what fits, then recurse into the oversized piece.
            if good:
                final.extend(self._merge(good, separator))
                good = []
            if remaining:
                final.extend(self._split(piece, remaining))
            else:
                final.append(piece)
        if good:
            final.extend(self._merge(good, separator))
        return final

    def _merge(self, splits: list[str], separator: str) -> list[str]:
        """Greedily merge small pieces up to ``chunk_size`` with overlap."""
        sep_len = self.length_function(separator)
        docs: list[str] = []
        current: list[str] = []
        total = 0
        for piece in splits:
            piece_len = self.length_function(piece)
            joined_len = sep_len if current else 0
            if total + piece_len + joined_len > self.chunk_size and current:
                doc = separator.join(current).strip()
                if doc:
                    docs.append(doc)
                # Drop from the front until the running window is back under the
                # overlap budget (and small enough to admit the next piece).
                while current and (
                    total > self.chunk_overlap
                    or total + piece_len + joined_len > self.chunk_size
                ):
                    total -= self.length_function(current[0]) + (
                        sep_len if len(current) > 1 else 0
                    )
                    current = current[1:]
                    joined_len = sep_len if current else 0
            current.append(piece)
            total += piece_len + (sep_len if len(current) > 1 else 0)

        doc = separator.join(current).strip()
        if doc:
            docs.append(doc)
        return docs


def chunk_document(
    document: dict, splitter: RecursiveCharacterTextSplitter | None = None
) -> list[dict]:
    """
    Split one ingestion document into structurally chunked records.

    Each output record carries clean source-attribution metadata:
    ``text``, ``source``, ``title``, ``section``, ``page_number`` and a
    per-document ``chunk_index``.
    """
    splitter = splitter or RecursiveCharacterTextSplitter()
    chunks = splitter.split_text(document.get("content", ""))
    records: list[dict] = []
    for index, chunk in enumerate(c for c in chunks if c.strip()):
        records.append(
            {
                "text": chunk,
                "source": document["source"],
                "title": document.get("title", ""),
                "section": document.get("title", ""),
                "page_number": document.get("page_number"),
                "chunk_index": index,
            }
        )
    return records


def chunk_documents(
    documents: list[dict], splitter: RecursiveCharacterTextSplitter | None = None
) -> list[dict]:
    """Structurally chunk a list of ingestion documents, preserving metadata."""
    splitter = splitter or RecursiveCharacterTextSplitter()
    records: list[dict] = []
    for document in documents:
        records.extend(chunk_document(document, splitter))
    return records
