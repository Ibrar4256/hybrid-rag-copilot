import cohere


class CohereRerankProvider:
    """Configurable alternative to LocalCrossEncoderProvider (ADR-004).
    Uses Cohere's free trial tier — not a permanent fallback."""

    def __init__(self, api_key: str, model: str = "rerank-english-v3.0"):
        self._client = cohere.Client(api_key)
        self._model = model

    def rerank(self, query: str, candidates: list[str], top_n: int) -> list[int]:
        result = self._client.rerank(
            query=query, documents=candidates, top_n=top_n, model=self._model
        )
        return [r.index for r in result.results]

    def score(self, query: str, candidate: str) -> float:
        result = self._client.rerank(
            query=query, documents=[candidate], top_n=1, model=self._model
        )
        return float(result.results[0].relevance_score)
