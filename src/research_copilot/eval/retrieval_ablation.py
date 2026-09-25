"""Offline retrieval ablation: dense-only vs +hybrid vs +hybrid+reranking.
No Gemini calls — pure retrieval-stage metrics against hand-verified ground
truth (ground_truth.py). Run with:

    python -m research_copilot.eval.retrieval_ablation
"""

from research_copilot.eval.ground_truth import build_ground_truth
from research_copilot.query import Retriever

MODES = [
    ("dense-only", dict(use_hybrid=False, use_reranker=False)),
    ("+hybrid (RRF)", dict(use_hybrid=True, use_reranker=False)),
    ("+hybrid +reranking", dict(use_hybrid=True, use_reranker=True)),
]


def _hit_rank(results: list[dict], correct_chunk_ids: set[str]) -> int | None:
    """1-indexed rank of the first correct chunk in results, or None if absent."""
    for i, r in enumerate(results, 1):
        if r["chunk_id"] in correct_chunk_ids:
            return i
    return None


def run_ablation(collection: str = "sec_filings_banks_chunk200") -> dict:
    ground_truth = build_ground_truth(chunk_size_tokens=200, chunk_overlap_tokens=40)
    retriever = Retriever(collection=collection)

    mode_results = {}
    for mode_name, kwargs in MODES:
        hits_at_1 = 0
        hits_at_5 = 0
        reciprocal_ranks = []
        per_question = []

        for gt in ground_truth:
            correct = set(gt["chunk_ids"])
            results = retriever.search(gt["question"], **kwargs)
            rank = _hit_rank(results, correct)

            if rank == 1:
                hits_at_1 += 1
            if rank is not None and rank <= 5:
                hits_at_5 += 1
            reciprocal_ranks.append(1 / rank if rank else 0.0)
            per_question.append({"id": gt["id"], "rank": rank})

        n = len(ground_truth)
        mode_results[mode_name] = {
            "hit_rate@1": hits_at_1 / n,
            "hit_rate@5": hits_at_5 / n,
            "mrr": sum(reciprocal_ranks) / n,
            "per_question": per_question,
        }
    return mode_results


def print_report(mode_results: dict) -> None:
    print(f"{'Mode':<22} {'Hit@1':>8} {'Hit@5':>8} {'MRR':>8}")
    print("-" * 48)
    for mode_name, m in mode_results.items():
        print(f"{mode_name:<22} {m['hit_rate@1']:>8.1%} {m['hit_rate@5']:>8.1%} {m['mrr']:>8.3f}")


if __name__ == "__main__":
    results = run_ablation()
    print_report(results)
