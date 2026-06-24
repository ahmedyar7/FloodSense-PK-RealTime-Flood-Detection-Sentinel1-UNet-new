from typing import Callable
from qdrant_client import QdrantClient

from .ingest import COLLECTION_NAME


def retrieve(
    query: str,
    client: QdrantClient,
    embedder,
    collection_name: str = COLLECTION_NAME,
    top_k: int = 3,
) -> list[dict]:
    """Embed query and return top-k matching document payloads from Qdrant."""
    query_vector = embedder.embed([query])[0]
    results = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=top_k,
    )
    return [hit.payload for hit in results.points]


def build_context(docs: list[dict]) -> str:
    """Format retrieved payloads as numbered context for an LLM prompt."""
    parts = []
    for i, doc in enumerate(docs, 1):
        parts.append(f"[{i}] Source: {doc['source']}\n{doc['text']}")
    return "\n\n".join(parts)


def rag_query(
    query: str,
    client: QdrantClient,
    embedder,
    llm_fn: Callable[[str], str],
    collection_name: str = COLLECTION_NAME,
    top_k: int = 3,
) -> tuple[str, list[dict]]:
    """
    Full RAG pipeline: retrieve relevant chunks then call llm_fn with context.

    Returns (llm_response, retrieved_docs).
    """
    docs = retrieve(query, client, embedder, collection_name, top_k)
    context = build_context(docs)

    prompt = (
        "You are a disaster intelligence assistant for Pakistan. "
        "Answer the question strictly using the provided context. "
        "If the context does not contain enough information, say so.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )

    response = llm_fn(prompt)
    return response, docs
