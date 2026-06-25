"""
Tests for the real FloodSense-PK PDF ingestion pipeline.

Covers:
  * The PDF loads successfully without raising.
  * Structure-aware chunking yields a reasonable number of non-garbled chunks.
  * Domain keywords from the PDF survive into the ingested Qdrant chunks.
"""

import re

import pytest
from unittest.mock import MagicMock
from qdrant_client import QdrantClient

from rag.pdf_loader import (
    DEFAULT_PDF,
    extract_pdf_text,
    load_pdf_documents,
    split_into_sections,
)
from rag.ingest import (
    chunk_text,
    ingest_documents,
    load_default_documents,
    COLLECTION_NAME,
)


# Keywords that must survive extraction -> sectioning -> chunking -> ingestion.
EXPECTED_KEYWORDS = ["Trimmu Headworks", "FloodSense-PK"]


@pytest.fixture(scope="module")
def pdf_text() -> str:
    return extract_pdf_text(DEFAULT_PDF)


@pytest.fixture(scope="module")
def pdf_documents() -> list[dict]:
    return load_pdf_documents(DEFAULT_PDF)


def test_pdf_file_exists():
    assert DEFAULT_PDF.exists(), f"Expected PDF at {DEFAULT_PDF}"


def test_pdf_loads_without_error(pdf_text):
    # Loading + extraction must not raise and must yield substantial text.
    assert isinstance(pdf_text, str)
    assert len(pdf_text) > 1000, "Extracted PDF text is suspiciously short"


def test_extracted_text_is_not_garbled(pdf_text):
    # A healthy extraction is mostly printable ASCII-ish words, not noise.
    words = pdf_text.split()
    assert len(words) > 200, "Too few words extracted from the PDF"

    alpha_words = [w for w in words if re.search(r"[A-Za-z]", w)]
    ratio = len(alpha_words) / len(words)
    assert ratio > 0.6, f"Text looks garbled: only {ratio:.0%} of tokens have letters"

    # Whitespace normalisation should have collapsed pypdf's per-token newlines.
    assert "\n" not in pdf_text
    assert "  " not in pdf_text, "Double spaces should be collapsed"


def test_sections_follow_document_hierarchy(pdf_text):
    sections = split_into_sections(pdf_text)
    # The source document has seven top-level numbered sections.
    assert len(sections) == 7, f"Expected 7 sections, got {len(sections)}"
    numbers = [s["number"] for s in sections]
    assert numbers == list(range(1, 8)), f"Sections out of order: {numbers}"
    for section in sections:
        assert section["content"].strip(), "Section content should be non-empty"
        assert section["title"].strip(), "Section title should be non-empty"


def test_pdf_documents_have_ingestion_schema(pdf_documents):
    assert len(pdf_documents) == 7
    for doc in pdf_documents:
        assert set(doc) >= {"source", "title", "content"}
        assert doc["source"] == DEFAULT_PDF.name


def test_chunking_produces_reasonable_chunks(pdf_documents):
    all_chunks = []
    for doc in pdf_documents:
        all_chunks.extend(chunk_text(doc["content"]))

    # Reasonable lower/upper bounds: more than a handful, not absurdly many.
    assert 5 <= len(all_chunks) <= 500, f"Unexpected chunk count: {len(all_chunks)}"

    # Chunks should contain real, readable words rather than noise.
    for chunk in all_chunks:
        assert chunk.strip(), "No empty chunks expected"
        words = chunk.split()
        assert len(words) >= 1
        avg_len = sum(len(w) for w in words) / len(words)
        assert 2 <= avg_len <= 20, f"Garbled chunk (avg word len {avg_len:.1f})"


def test_keywords_present_in_pdf_chunks(pdf_documents):
    chunks = []
    for doc in pdf_documents:
        chunks.extend(chunk_text(doc["content"]))
    haystack = " ".join(chunks)
    for keyword in EXPECTED_KEYWORDS:
        assert keyword in haystack, f"Keyword '{keyword}' missing from PDF chunks"


def test_keywords_present_in_ingested_qdrant_chunks():
    client = QdrantClient(":memory:")

    embedder = MagicMock()
    embedder.embed.side_effect = lambda texts: [[0.1] * 384 for _ in texts]

    documents = load_pdf_documents(DEFAULT_PDF)
    chunk_count = ingest_documents(client, embedder, documents=documents)
    assert chunk_count > 0

    points, _ = client.scroll(
        collection_name=COLLECTION_NAME, limit=10_000, with_payload=True
    )
    ingested_text = " ".join(p.payload["text"] for p in points)
    for keyword in EXPECTED_KEYWORDS:
        assert keyword in ingested_text, (
            f"Keyword '{keyword}' not found in ingested Qdrant chunks"
        )


def test_default_corpus_includes_pdf_and_mock():
    documents = load_default_documents()
    sources = {doc["source"] for doc in documents}
    assert DEFAULT_PDF.name in sources, "Default corpus should include the PDF"
    assert "NDMA Flood Report 2010" in sources, "Default corpus should keep mock docs"
