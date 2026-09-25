from typing import Protocol


class EmbeddingProvider(Protocol):
    """Dense embedding backend. Documents and queries are embedded separately
    because asymmetric models (e.g. bge) expect a different instruction
    prefix for queries than for the passages they're matched against."""

    dimension: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...
