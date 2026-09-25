from unittest.mock import MagicMock

from research_copilot.query import Retriever


def _bare_retriever_with_scores(fake_scores):
    """Build a Retriever without running __init__ (which loads real local
    models) — only _rerank_rrf_boosted's dependency (self._reranker._model)
    is set, since that's the only attribute this pure-scoring method touches."""
    r = Retriever.__new__(Retriever)
    fake_reranker = MagicMock()
    fake_reranker._model.predict.return_value = fake_scores
    r._reranker = fake_reranker
    return r


def _candidates(n):
    return [{"text": f"chunk {i}"} for i in range(n)]


def test_rerank_returns_top_n_unique_valid_indices():
    r = _bare_retriever_with_scores([0.5, 0.9, 0.1, 0.7, 0.3])
    ranked = r._rerank_rrf_boosted("q", _candidates(5), top_n=3)

    assert len(ranked) == 3
    assert len(set(ranked)) == 3
    assert all(0 <= i < 5 for i in ranked)


def test_rerank_compressed_scores_falls_back_to_rrf_order():
    # this is the exact failure mode ADR-004 documents: when the
    # cross-encoder can't discriminate (near-identical scores), the RRF
    # position bonus should dominate and preserve the original candidate
    # (i.e. RRF) order rather than picking randomly among tied scores
    n = 10
    compressed_scores = [0.97 for _ in range(n)]
    r = _bare_retriever_with_scores(compressed_scores)

    ranked = r._rerank_rrf_boosted("q", _candidates(n), top_n=5)

    assert ranked == [0, 1, 2, 3, 4]


def test_rerank_can_rescue_a_strong_candidate_from_deep_in_rrf_order():
    # candidate at RRF rank 9 (last, smallest RRF bonus) but with a
    # dramatically higher reranker score than everything else should still
    # be rescued into the top_n — this is the reranker's whole job
    n = 10
    scores = [0.5] * (n - 1) + [0.99]  # last candidate scores far higher
    r = _bare_retriever_with_scores(scores)

    ranked = r._rerank_rrf_boosted("q", _candidates(n), top_n=3)

    assert (n - 1) in ranked


def test_rerank_handles_identical_scores_without_division_by_zero():
    # s_max == s_min triggers the fallback branch (s_norm = ones); must not
    # raise and must still return a valid, fully-ordered-by-RRF result
    r = _bare_retriever_with_scores([0.8, 0.8, 0.8, 0.8])
    ranked = r._rerank_rrf_boosted("q", _candidates(4), top_n=4)

    assert ranked == [0, 1, 2, 3]


def test_rerank_top_n_larger_than_candidates_returns_all():
    r = _bare_retriever_with_scores([0.3, 0.6])
    ranked = r._rerank_rrf_boosted("q", _candidates(2), top_n=10)

    assert len(ranked) == 2
