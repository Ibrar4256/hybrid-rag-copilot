from typing import Protocol


class RerankerProvider(Protocol):
    def rerank(self, query: str, candidates: list[str], top_n: int) -> list[int]:
        """Return indices into `candidates`, best match first, length top_n."""
        ...

    def score(self, query: str, candidate: str) -> float:
        """Raw relevance score for a single (query, candidate) pair. NOTE: tried
        and rejected for entailment checking (ADR-006) — with realistic full
        chunk text, a confirmed-bad citation and a confirmed-good citation both
        scored 0.97+, since a reranker measures topical relevance ("same
        company/date"), not entailment ("does this text state this specific
        fact"). Kept as a general capability; entailment checking uses an
        LLM-based check instead (synthesis.py's _check_entailment)."""
        ...
