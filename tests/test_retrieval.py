"""
Test 2 — Retrieval: Query "What happened in Charsadda during the 2010 floods?"
Mocks the vector DB to return Kabul River overflow data and asserts that the
generated response mentions the expected facts.
"""

import pytest
from unittest.mock import MagicMock

from rag.retriever import rag_query, retrieve, build_context


CHARSADDA_PAYLOAD = {
    "source": "NDMA Flood Report 2010",
    "title": "National Disaster Management Authority — Super Floods 2010",
    "text": (
        "Charsadda district in KPK experienced severe inundation due to Kabul River overflow, "
        "which breached its embankments on 29 July 2010 following continuous heavy rainfall. "
        "The overflow led to the displacement of residents from over 200 villages, with "
        "approximately 500,000 people evacuated to relief camps."
    ),
}

KABUL_RIVER_PAYLOAD = {
    "source": "FFD River Stage Report",
    "title": "Federal Flood Division — Historical Flood Peaks: Kabul River System",
    "text": (
        "Charsadda lies in the floodplain of the Kabul River and is particularly vulnerable "
        "to rapid inundation when embankments fail. The 2010 event at Charsadda resulted in "
        "the displacement of residents across the entire district."
    ),
}


def _make_mock_hit(payload: dict) -> MagicMock:
    hit = MagicMock()
    hit.payload = payload
    return hit


def _make_query_result(*payloads) -> MagicMock:
    """Wrap payloads in a QueryResponse-like mock (results.points)."""
    result = MagicMock()
    result.points = [_make_mock_hit(p) for p in payloads]
    return result


def _mock_llm(prompt: str) -> str:
    """Simulate an LLM that summarises the retrieved context faithfully."""
    return (
        "During the 2010 floods, Charsadda district experienced severe inundation due to "
        "Kabul River overflow after the river breached its embankments on 29 July 2010. "
        "The disaster caused the displacement of residents from over 200 villages, with "
        "approximately 500,000 people evacuated to relief camps."
    )


def test_rag_retrieval_charsadda_2010():
    mock_client = MagicMock()
    mock_client.query_points.return_value = _make_query_result(
        CHARSADDA_PAYLOAD, KABUL_RIVER_PAYLOAD
    )

    mock_embedder = MagicMock()
    mock_embedder.embed.return_value = [[0.1] * 384]

    query = "What happened in Charsadda during the 2010 floods?"
    response, docs = rag_query(query, mock_client, mock_embedder, _mock_llm)

    assert "severe inundation due to Kabul River overflow" in response, (
        "Response must mention the cause of flooding: Kabul River overflow"
    )
    assert "displacement of residents" in response, (
        "Response must mention displacement of residents"
    )


def test_retrieval_returns_docs_from_vector_db():
    mock_client = MagicMock()
    mock_client.query_points.return_value = _make_query_result(CHARSADDA_PAYLOAD)

    mock_embedder = MagicMock()
    mock_embedder.embed.return_value = [[0.2] * 384]

    query = "What happened in Charsadda during the 2010 floods?"
    docs = retrieve(query, mock_client, mock_embedder)

    assert len(docs) == 1
    assert docs[0]["source"] == "NDMA Flood Report 2010"


def test_build_context_formats_docs():
    docs = [CHARSADDA_PAYLOAD, KABUL_RIVER_PAYLOAD]
    context = build_context(docs)

    assert "[1]" in context
    assert "[2]" in context
    assert "NDMA Flood Report 2010" in context
    assert "FFD River Stage Report" in context


def test_rag_query_calls_embedder_and_search():
    mock_client = MagicMock()
    mock_client.query_points.return_value = _make_query_result(CHARSADDA_PAYLOAD)

    mock_embedder = MagicMock()
    mock_embedder.embed.return_value = [[0.3] * 384]

    llm_calls = []

    def recording_llm(prompt: str) -> str:
        llm_calls.append(prompt)
        return "Charsadda experienced severe inundation due to Kabul River overflow, leading to the displacement of residents."

    query = "What happened in Charsadda during the 2010 floods?"
    response, _ = rag_query(query, mock_client, mock_embedder, recording_llm)

    mock_embedder.embed.assert_called_once_with([query])
    mock_client.query_points.assert_called_once()
    assert len(llm_calls) == 1
    assert "Context:" in llm_calls[0]
    assert "severe inundation due to Kabul River overflow" in response
    assert "displacement of residents" in response
