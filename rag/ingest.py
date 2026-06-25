import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

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
    """Split text into overlapping word-count chunks."""
    words = text.split()
    chunks = []
    step = chunk_size - overlap
    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks


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
) -> int:
    """Chunk documents, embed each chunk, and upsert into Qdrant. Returns chunk count."""
    if documents is None:
        documents = load_default_documents()

    ensure_collection(client, collection_name)

    points = []
    for doc in documents:
        chunks = chunk_text(doc["content"])
        texts = chunks
        embeddings = embedder.embed(texts)
        for idx, (chunk, vector) in enumerate(zip(chunks, embeddings)):
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload={
                        "source": doc["source"],
                        "title": doc.get("title", ""),
                        "chunk_index": idx,
                        "text": chunk,
                    },
                )
            )

    if points:
        client.upsert(collection_name=collection_name, points=points)

    return len(points)
