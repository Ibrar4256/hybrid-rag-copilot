from research_copilot.config import settings
from research_copilot.embeddings.base import EmbeddingProvider
from research_copilot.reranker.base import RerankerProvider


def get_embedding_provider() -> EmbeddingProvider:
    if settings.embedding_provider == "local_bge":
        from research_copilot.embeddings.local_bge import LocalBGEProvider

        return LocalBGEProvider()
    if settings.embedding_provider == "gemini":
        from research_copilot.embeddings.gemini import GeminiEmbeddingProvider

        if not settings.gemini_api_key:
            raise ValueError("GEMINI_API_KEY is required when EMBEDDING_PROVIDER=gemini")
        return GeminiEmbeddingProvider(settings.gemini_api_key)
    raise ValueError(f"Unknown embedding provider: {settings.embedding_provider}")


def get_reranker_provider() -> RerankerProvider:
    if settings.reranker_provider == "local_cross_encoder":
        from research_copilot.reranker.local_cross_encoder import (
            LocalCrossEncoderProvider,
        )

        return LocalCrossEncoderProvider()
    if settings.reranker_provider == "cohere":
        from research_copilot.reranker.cohere_rerank import CohereRerankProvider

        if not settings.cohere_api_key:
            raise ValueError("COHERE_API_KEY is required when RERANKER_PROVIDER=cohere")
        return CohereRerankProvider(settings.cohere_api_key)
    raise ValueError(f"Unknown reranker provider: {settings.reranker_provider}")
