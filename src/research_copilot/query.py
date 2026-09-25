import sys

import numpy as np

from research_copilot.config import settings
from research_copilot.providers import get_embedding_provider, get_reranker_provider
from research_copilot.sparse.bm25 import BM25SparseProvider
from research_copilot.telemetry import log_call, timed
from research_copilot.vector_store import QdrantHybridStore

RETRIEVE_K = 20
RERANK_TOP_N = 7
RERANKER_WEIGHT = 0.7
RRF_POSITION_WEIGHT = 0.3


class Retriever:
    """Wraps the retrieval pipeline. Holds providers/models so the agent loop
    (ADR-005) can call `search()` repeatedly without reloading them.

    `use_hybrid` and `use_reranker` are ablation toggles (dense-only vs +hybrid
    vs +reranking) — both default True, matching the production pipeline; the
    eval harness flips them off independently to build the ablation table."""

    def __init__(self, collection: str | None = None):
        self._embedder = get_embedding_provider()
        self._sparse = BM25SparseProvider()
        self._reranker = get_reranker_provider()
        self._store = QdrantHybridStore(
            url=settings.qdrant_url,
            collection=collection or settings.qdrant_collection,
            dense_dim=self._embedder.dimension,
        )

    def search(
        self, query: str, use_hybrid: bool = True, use_reranker: bool = True
    ) -> list[dict]:
        with timed() as t_embed:
            dense_query = self._embedder.embed_query(query)
        log_call("local", "bge-base-en-v1.5", "embed_query", 0, 0, t_embed.ms, cost_usd=0.0)

        if use_hybrid:
            sparse_query = self._sparse.embed_query(query)
            candidates = self._store.hybrid_search(dense_query, sparse_query, limit=RETRIEVE_K)
        else:
            candidates = self._store.dense_search(dense_query, limit=RETRIEVE_K)

        if not candidates:
            return []

        if use_reranker:
            with timed() as t_rerank:
                ranked_indices = self._rerank_rrf_boosted(
                    query, candidates, top_n=RERANK_TOP_N
                )
            log_call("local", "bge-reranker-base", "rerank", 0, 0, t_rerank.ms, cost_usd=0.0)
            results = [candidates[i] for i in ranked_indices]
        else:
            results = candidates[:RERANK_TOP_N]

        for r in results:
            r["chunk_id"] = f"{r['source']}#{r['chunk_index']}"
        return results

    def _rerank_rrf_boosted(
        self, query: str, candidates: list[dict], top_n: int
    ) -> list[int]:
        """Combine cross-encoder scores with RRF position to break ties.

        The cross-encoder scores same-filing chunks in a compressed 0.94-1.0
        band, making top-N selection effectively random among them. Weighting
        by RRF position anchors the ordering to items both systems agree on,
        while preserving the reranker's ability to rescue correct chunks from
        deep in the candidate set (RRF rank 8-19)."""
        texts = [c["text"] for c in candidates]
        pairs = [(query, t) for t in texts]
        scores = self._reranker._model.predict(pairs)

        s_arr = np.array(scores, dtype=float)
        s_min, s_max = s_arr.min(), s_arr.max()
        if s_max > s_min:
            s_norm = (s_arr - s_min) / (s_max - s_min)
        else:
            s_norm = np.ones_like(s_arr)

        rrf_bonus = np.array([1.0 / (i + 1) for i in range(len(candidates))])
        combined = RERANKER_WEIGHT * s_norm + RRF_POSITION_WEIGHT * rrf_bonus
        ranked = sorted(range(len(candidates)), key=lambda i: combined[i], reverse=True)
        return ranked[:top_n]


if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else "What is retrieval-augmented generation?"
    results = Retriever().search(q)
    for i, r in enumerate(results, 1):
        print(f"\n--- #{i} (id={r['chunk_id']}, score={r['score']:.4f}) ---")
        print(r["text"][:300])
