import pytest
from qdrant_client import QdrantClient

from research_copilot.config import settings
from research_copilot.ingest import ingest_directory

INTEGRATION_COLLECTION = "integration_test_sample_docs"


@pytest.fixture(scope="session")
def sample_docs_collection():
    """Ingests data/sample_docs (12KB, seconds not minutes) into a dedicated
    Qdrant collection once per test session, for integration tests that need
    a real store/embedder/reranker instead of mocks. Requires a live Qdrant
    at settings.qdrant_url (docker compose up -d qdrant)."""
    client = QdrantClient(url=settings.qdrant_url, timeout=120)
    if client.collection_exists(INTEGRATION_COLLECTION):
        client.delete_collection(INTEGRATION_COLLECTION)

    ingest_directory("data/sample_docs", INTEGRATION_COLLECTION)

    yield INTEGRATION_COLLECTION

    client.delete_collection(INTEGRATION_COLLECTION)
