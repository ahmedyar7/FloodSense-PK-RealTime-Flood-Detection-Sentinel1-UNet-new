"""
Production ingestion pipeline for the FloodSense-PK RAG knowledge base.

Loads the structured architecture PDF (plus the legacy mock disaster corpus),
splits it with a structure-aware recursive chunker that keeps tables and
sections intact, embeds each chunk, and upserts the vectors — together with
clean source-attribution metadata (``source``, ``page_number``, ``section``) —
into a Qdrant collection.
"""

import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from .chunking import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    RecursiveCharacterTextSplitter,
    chunk_documents,
)
from .embeddings import VECTOR_SIZE
from .mock_documents import MOCK_DOCUMENTS
from .pdf_loader import DEFAULT_PDF, load_pdf_documents

COLLECTION_NAME = "flood_knowledge"


def load_default_documents() -> list[dict]:
    """
    Build the default knowledge corpus: the structured FloodSense-PK PDF plus
    the legacy mock disaster documents. Falls back to mock-only if the PDF is
    unavailable so ingestion never hard-fails on a missing file.
    """
    documents = list(MOCK_DOCUMENTS)
    if DEFAULT_PDF.exists():
        documents = load_pdf_documents(DEFAULT_PDF) + documents
    return documents


def chunk_text(text: str, chunk_size: int = 150, overlap: int = 20) -> list[str]:
    """
    Split text into overlapping word-count chunks.

    Retained as a lightweight helper for callers that only need plain string
    chunks. The production ingestion path uses
    :class:`~rag.chunking.RecursiveCharacterTextSplitter` instead — see
    :func:`structural_chunk_documents`.
    """
    words = text.split()
    chunks = []
    step = chunk_size - overlap
    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks


def default_splitter() -> RecursiveCharacterTextSplitter:
    """Construct the structural splitter used by the production pipeline."""
    return RecursiveCharacterTextSplitter(
        chunk_size=DEFAULT_CHUNK_SIZE,
        chunk_overlap=DEFAULT_CHUNK_OVERLAP,
    )


def structural_chunk_documents(
    documents: list[dict],
    splitter: RecursiveCharacterTextSplitter | None = None,
) -> list[dict]:
    """
    Structurally chunk documents into embedding-ready records.

    Each record carries ``text`` plus source-attribution metadata
    (``source``, ``title``, ``section``, ``page_number``, ``chunk_index``).
    """
    splitter = splitter or default_splitter()
    return chunk_documents(documents, splitter)


def ensure_collection(
    client: QdrantClient, collection_name: str = COLLECTION_NAME
) -> None:
    existing = {c.name for c in client.get_collections().collections}
    if collection_name not in existing:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )


def ingest_documents(
    client: QdrantClient,
    embedder,
    documents: list[dict] | None = None,
    collection_name: str = COLLECTION_NAME,
    splitter: RecursiveCharacterTextSplitter | None = None,
) -> int:
    """
    Chunk documents structurally, embed each chunk, and upsert into Qdrant.

    Returns the number of chunks (points) ingested.
    """
    if documents is None:
        documents = load_default_documents()

    ensure_collection(client, collection_name)

    records = structural_chunk_documents(documents, splitter)
    if not records:
        return 0

    texts = [record["text"] for record in records]
    embeddings = embedder.embed(texts)

    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload={
                "source": record["source"],
                "title": record["title"],
                "section": record["section"],
                "page_number": record["page_number"],
                "chunk_index": record["chunk_index"],
                "text": record["text"],
            },
        )
        for record, vector in zip(records, embeddings)
    ]

    client.upsert(collection_name=collection_name, points=points)
    return len(points)
