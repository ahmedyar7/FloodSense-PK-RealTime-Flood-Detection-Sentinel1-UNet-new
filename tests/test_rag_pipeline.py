"""
Comprehensive RAG ingestion pipeline tests for FloodSense-PK.

Exercises the production ingestion of the structured architecture PDF
(``rag/data/Structured-Knowledge-Instruction-Pipeline.pdf``) end to end:

  * Test 1 — PDF Loader Validation: the PDF exists, opens cleanly and extracts
    readable text without corruption/encoding errors.
  * Test 2 — Structural Chunking Check: chunking yields a list of non-empty,
    non-garbled chunks within sensible bounds.
  * Test 3 — System-Specific Knowledge Verification: domain terminology from
    the spec survives into the generated chunks.
  * Test 4 — End-to-End Qdrant Integration (mocked): ingestion upserts the
    correctly dimensioned vectors and formatted metadata payloads.
"""

from __future__ import annotations

import re
from unittest.mock import MagicMock

import pytest

from rag.chunking import RecursiveCharacterTextSplitter, estimate_tokens
from rag.embeddings import VECTOR_SIZE
from rag.ingest import (
    COLLECTION_NAME,
    ingest_documents,
    load_default_documents,
    structural_chunk_documents,
)
from rag.pdf_loader import (
    DEFAULT_PDF,
    extract_pdf_pages,
    extract_pdf_text,
    load_pdf_documents,
)

# Domain-specific terminology that must survive extraction -> sectioning ->
# chunking. "FloodSense-PK" and "Trimmu Headworks" originate in the PDF spec;
# "Charsadda" appears in the wider production corpus the pipeline ingests.
EXPECTED_KEYWORDS = ["FloodSense-PK", "Trimmu Headworks", "Charsadda"]


@pytest.fixture(scope="module")
def pdf_text() -> str:
    return extract_pdf_text(DEFAULT_PDF)


@pytest.fixture(scope="module")
def corpus() -> list[dict]:
    """The full production corpus (structured PDF + mock disaster docs)."""
    return load_default_documents()


@pytest.fixture(scope="module")
def chunks(corpus: list[dict]) -> list[dict]:
    return structural_chunk_documents(corpus)


# --------------------------------------------------------------------------- #
# Test 1 — PDF Loader Validation
# --------------------------------------------------------------------------- #
def test_pdf_loader_validation(pdf_text: str) -> None:
    # The source document must be present on disk.
    assert DEFAULT_PDF.exists(), f"Expected PDF at {DEFAULT_PDF}"

    # Extraction must not raise and must yield substantial, readable text.
    assert isinstance(pdf_text, str)
    assert len(pdf_text) > 1000, "Extracted PDF text is suspiciously short"

    words = pdf_text.split()
    assert len(words) > 200, "Too few words extracted from the PDF"

    # A clean (non-garbled) extraction is mostly alphabetic tokens.
    alpha_words = [w for w in words if re.search(r"[A-Za-z]", w)]
    ratio = len(alpha_words) / len(words)
    assert ratio > 0.6, f"Text looks garbled: only {ratio:.0%} of tokens have letters"


def test_pdf_pages_extracted_with_page_numbers() -> None:
    pages = extract_pdf_pages(DEFAULT_PDF)
    assert len(pages) > 1, "Multi-page PDF should yield multiple pages"
    assert [p["page_number"] for p in pages] == list(range(1, len(pages) + 1))
    assert any(p["text"].strip() for p in pages), "At least one page must have text"


# --------------------------------------------------------------------------- #
# Test 2 — Structural Chunking Check
# --------------------------------------------------------------------------- #
def test_structural_chunking(chunks: list[dict]) -> None:
    # The chunking operation returns a list of chunk records.
    assert isinstance(chunks, list)
    assert len(chunks) > 5, f"Expected more than 5 chunks, got {len(chunks)}"

    for chunk in chunks:
        text = chunk["text"]
        # No null / zero-length / whitespace-only chunks.
        assert isinstance(text, str)
        assert text.strip(), "Encountered an empty or whitespace-only chunk"

        # No garbled chunks: readable average word length.
        words = text.split()
        assert words, "Chunk has no words"
        avg_len = sum(len(w) for w in words) / len(words)
        assert 2 <= avg_len <= 20, f"Garbled chunk (avg word len {avg_len:.1f})"

        # Chunks respect the configured token budget (with slack for the
        # estimate). Default splitter targets 500 tokens.
        assert estimate_tokens(text) <= 600, "Chunk exceeds the token budget"

    # Each chunk must carry clean source-attribution metadata.
    for chunk in chunks:
        assert set(chunk) >= {"text", "source", "section", "page_number", "chunk_index"}


def test_pdf_documents_carry_page_metadata() -> None:
    pdf_docs = load_pdf_documents(DEFAULT_PDF)
    assert len(pdf_docs) > 5
    for doc in pdf_docs:
        assert doc["source"] == DEFAULT_PDF.name
        assert isinstance(doc["page_number"], int) and doc["page_number"] >= 1


def test_recursive_splitter_keeps_small_sections_intact() -> None:
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    short = "Trimmu Headworks discharge data for the Chenab-Jhelum system."
    # A short section comfortably under the budget stays as a single chunk.
    assert splitter.split_text(short) == [short]


# --------------------------------------------------------------------------- #
# Test 3 — System-Specific Knowledge Verification
# --------------------------------------------------------------------------- #
def test_system_specific_knowledge(chunks: list[dict]) -> None:
    haystack = " ".join(chunk["text"] for chunk in chunks)
    for keyword in EXPECTED_KEYWORDS:
        assert keyword in haystack, (
            f"Domain keyword '{keyword}' missing from the generated chunks"
        )


# --------------------------------------------------------------------------- #
# Test 4 — End-to-End Qdrant Integration (mocked)
# --------------------------------------------------------------------------- #
def test_qdrant_integration_upsert_payloads() -> None:
    # Mock the Qdrant client so we can inspect the upload payload directly.
    client = MagicMock()
    # No existing collections -> ensure_collection should create ours.
    client.get_collections.return_value.collections = []

    # Deterministic, correctly dimensioned embeddings.
    embedder = MagicMock()
    embedder.embed.side_effect = lambda texts: [[0.1] * VECTOR_SIZE for _ in texts]

    documents = load_pdf_documents(DEFAULT_PDF)
    chunk_count = ingest_documents(client, embedder, documents=documents)

    assert chunk_count > 0, "Ingestion should report a positive chunk count"

    # The collection must have been created with the right vector size.
    client.create_collection.assert_called_once()
    _, create_kwargs = client.create_collection.call_args
    assert create_kwargs["collection_name"] == COLLECTION_NAME
    assert create_kwargs["vectors_config"].size == VECTOR_SIZE

    # The upsert method must have been called exactly once with our points.
    client.upsert.assert_called_once()
    _, upsert_kwargs = client.upsert.call_args
    assert upsert_kwargs["collection_name"] == COLLECTION_NAME

    points = upsert_kwargs["points"]
    assert len(points) == chunk_count

    for point in points:
        # Correct dimensional payload.
        assert len(point.vector) == VECTOR_SIZE
        # Formatted metadata payload with clean source attribution.
        payload = point.payload
        assert set(payload) >= {
            "source",
            "section",
            "page_number",
            "chunk_index",
            "text",
        }
        assert payload["source"] == DEFAULT_PDF.name
        assert payload["text"].strip()
        assert isinstance(payload["page_number"], int)

    # Sanity: domain terminology from the PDF is present in the upserted text.
    upserted_text = " ".join(point.payload["text"] for point in points)
    assert "FloodSense-PK" in upserted_text
    assert "Trimmu Headworks" in upserted_text
