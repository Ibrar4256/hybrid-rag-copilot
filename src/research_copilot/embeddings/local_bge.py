from sentence_transformers import SentenceTransformer

# bge models are trained asymmetrically: queries need this instruction
# prefix to align with how passages were indexed; passages get none.
_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class LocalBGEProvider:
    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5"):
        self._model = SentenceTransformer(model_name)
        self.dimension = self._model.get_sentence_embedding_dimension()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self._model.encode(
            _QUERY_INSTRUCTION + text, normalize_embeddings=True
        ).tolist()
