"""
Test 1 — Ingestion: Assert that a sample text document is successfully
chunked and stored in the Qdrant vector database collection.
"""

import pytest
from unittest.mock import MagicMock
from qdrant_client import QdrantClient

from rag.ingest import ingest_documents, chunk_text, COLLECTION_NAME


def test_chunk_text_produces_multiple_chunks():
    long_text = ("flood water rose rapidly across the plains. " * 10).strip()
    chunks = chunk_text(long_text, chunk_size=10, overlap=2)
    assert len(chunks) > 1, "Long text should produce more than one chunk"


def test_chunk_text_single_chunk_for_short_text():
    short_text = "The Kabul River overflowed its banks."
    chunks = chunk_text(short_text, chunk_size=150, overlap=20)
    assert len(chunks) == 1


def test_document_ingestion_stores_chunks_in_qdrant():
    client = QdrantClient(":memory:")

    embedder = MagicMock()
    embedder.embed.side_effect = lambda texts: [[0.1] * 384 for _ in texts]

    sample_doc = [
        {
            "source": "NDMA Test Report",
            "title": "Test Flood Document",
            "content": (
                "The 2010 Pakistan floods caused severe inundation in Charsadda district "
                "due to Kabul River overflow. Displacement of residents was widespread. "
            )
            * 10,
        }
    ]

    chunk_count = ingest_documents(client, embedder, documents=sample_doc)

    assert chunk_count > 0, "Ingestion should return a positive chunk count"

    stored = client.count(collection_name=COLLECTION_NAME)
    assert stored.count == chunk_count, (
        f"Qdrant collection should hold {chunk_count} points, got {stored.count}"
    )


def test_ingestion_payload_contains_source_and_text():
    client = QdrantClient(":memory:")

    embedder = MagicMock()
    embedder.embed.side_effect = lambda texts: [[0.0] * 384 for _ in texts]

    sample_doc = [
        {
            "source": "FFD Bulletin",
            "title": "River Stage Report",
            "content": "Indus River discharge at Sukkur Barrage reached extreme levels. " * 5,
        }
    ]

    ingest_documents(client, embedder, documents=sample_doc)

    points, _ = client.scroll(collection_name=COLLECTION_NAME, limit=10, with_payload=True)
    assert len(points) > 0
    for point in points:
        assert "source" in point.payload
        assert "text" in point.payload
        assert point.payload["source"] == "FFD Bulletin"
