import time

import google.generativeai as genai

_MODEL = "models/text-embedding-004"
_DIMENSION = 768
_MAX_RETRIES = 5


def _embed_with_backoff(**kwargs) -> list[float]:
    for attempt in range(_MAX_RETRIES):
        try:
            return genai.embed_content(**kwargs)["embedding"]
        except Exception:
            if attempt == _MAX_RETRIES - 1:
                raise
            time.sleep(2**attempt)  # 1s, 2s, 4s, 8s
    raise RuntimeError("unreachable")


class GeminiEmbeddingProvider:
    """Free-tier configurable alternative to LocalBGEProvider (ADR-002).
    Rate-limited, so callers should batch modestly and expect backoff delays."""

    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.dimension = _DIMENSION

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [
            _embed_with_backoff(
                model=_MODEL, content=t, task_type="retrieval_document"
            )
            for t in texts
        ]

    def embed_query(self, text: str) -> list[float]:
        return _embed_with_backoff(
            model=_MODEL, content=text, task_type="retrieval_query"
        )
