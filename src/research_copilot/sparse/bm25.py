from fastembed import SparseTextEmbedding, SparseEmbedding


class BM25SparseProvider:
    """Local, free sparse (BM25-style) vectors for hybrid search, via fastembed.
    Pairs with LocalBGEProvider's dense vectors in Qdrant's hybrid query (ADR-001)."""

    def __init__(self, model_name: str = "Qdrant/bm25"):
        self._model = SparseTextEmbedding(model_name=model_name)

    def embed_documents(self, texts: list[str]) -> list[SparseEmbedding]:
        return list(self._model.embed(texts))

    def embed_query(self, text: str) -> SparseEmbedding:
        return list(self._model.query_embed(text))[0]
