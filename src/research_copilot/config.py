from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    embedding_provider: str = "local_bge"
    reranker_provider: str = "local_cross_encoder"

    gemini_api_key: str | None = None
    gemini_api_keys: str | None = None
    cohere_api_key: str | None = None

    @property
    def gemini_key_pool(self) -> list[str]:
        if self.gemini_api_keys:
            return [k.strip() for k in self.gemini_api_keys.split(",") if k.strip()]
        if self.gemini_api_key:
            return [self.gemini_api_key]
        return []

    # LLM fallback providers for the agent loop + synthesis, used when
    # Gemini's daily quota is exhausted (ADR-007). All optional.
    groq_api_key: str | None = None
    openrouter_api_key: str | None = None
    sambanova_api_key: str | None = None
    cerebras_api_key: str | None = None

    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "research_copilot"

    # Tuned via the eval harness's retrieval ablation (see WEEKLY_LOG.md and
    # ADR-003 addendum): 500/75 gave Hit@5=32.1%; 200/40 gave Hit@5=53.6% on the
    # same 28-question ground truth. Root cause was oversized chunks diluting
    # embedding signal for facts buried near the end of topically-mixed text.
    chunk_size_tokens: int = 200
    chunk_overlap_tokens: int = 40

    class Config:
        env_file = ".env"


settings = Settings()
