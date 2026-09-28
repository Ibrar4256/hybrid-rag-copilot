"""Integration tests: real Qdrant, real local embedding + reranker models,
real ingestion. No mocks, unlike the unit tests in test_query_rerank.py etc.
Requires a live Qdrant (docker compose up -d qdrant) — see conftest.py.
"""

import pytest

from research_copilot.query import Retriever

pytestmark = pytest.mark.integration


def test_dense_only_search_returns_results(sample_docs_collection):
    retriever = Retriever(collection=sample_docs_collection)
    results = retriever.search("what is retrieval augmented generation", use_hybrid=False, use_reranker=False)

    assert len(results) > 0
    assert all("chunk_id" in r and "text" in r for r in results)


def test_hybrid_search_returns_results(sample_docs_collection):
    retriever = Retriever(collection=sample_docs_collection)
    results = retriever.search("what is retrieval augmented generation", use_hybrid=True, use_reranker=False)

    assert len(results) > 0


def test_hybrid_reranked_search_full_pipeline(sample_docs_collection):
    retriever = Retriever(collection=sample_docs_collection)
    results = retriever.search("what is retrieval augmented generation", use_hybrid=True, use_reranker=True)

    assert len(results) > 0
    # chunk_id format is "source#chunk_index" (query.py's Retriever.search())
    for r in results:
        assert "#" in r["chunk_id"]
        source, index = r["chunk_id"].rsplit("#", 1)
        assert index.isdigit()


def test_search_surfaces_topically_relevant_document(sample_docs_collection):
    # a real content-relevance check, not just "did it return something" —
    # a RAG-specific query should surface rag.md, not vector_databases.md
    retriever = Retriever(collection=sample_docs_collection)
    results = retriever.search(
        "how does retrieval augmented generation reduce hallucination",
        use_hybrid=True, use_reranker=True,
    )

    assert any(r["source"] == "rag.md" for r in results)


def test_search_distinguishes_between_documents(sample_docs_collection):
    retriever = Retriever(collection=sample_docs_collection)
    results = retriever.search(
        "what indexing structure does Qdrant use for approximate nearest neighbor search",
        use_hybrid=True, use_reranker=True,
    )

    assert any(r["source"] == "vector_databases.md" for r in results)


def test_ingested_chunks_are_deduplicated_by_chunk_id(sample_docs_collection):
    retriever = Retriever(collection=sample_docs_collection)
    results = retriever.search("vector database", use_hybrid=True, use_reranker=False)

    chunk_ids = [r["chunk_id"] for r in results]
    assert len(chunk_ids) == len(set(chunk_ids))
