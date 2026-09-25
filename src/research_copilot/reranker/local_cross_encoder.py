from sentence_transformers import CrossEncoder


class LocalCrossEncoderProvider:
    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        self._model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: list[str], top_n: int) -> list[int]:
        pairs = [(query, c) for c in candidates]
        scores = self._model.predict(pairs)
        ranked = sorted(range(len(candidates)), key=lambda i: scores[i], reverse=True)
        return ranked[:top_n]

    def score(self, query: str, candidate: str) -> float:
        return float(self._model.predict([(query, candidate)])[0])
