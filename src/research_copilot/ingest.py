import sys
from pathlib import Path

from research_copilot.chunking import chunk_document, chunk_document_with_children
from research_copilot.config import settings
from research_copilot.providers import get_embedding_provider
from research_copilot.sparse.bm25 import BM25SparseProvider
from research_copilot.vector_store import QdrantHybridStore


def ingest_directory(
    directory: str,
    collection: str | None = None,
    chunk_size_tokens: int | None = None,
    chunk_overlap_tokens: int | None = None,
) -> None:
    chunk_size_tokens = chunk_size_tokens or settings.chunk_size_tokens
    chunk_overlap_tokens = chunk_overlap_tokens or settings.chunk_overlap_tokens

    embedder = get_embedding_provider()
    sparse = BM25SparseProvider()
    store = QdrantHybridStore(
        url=settings.qdrant_url,
        collection=collection or settings.qdrant_collection,
        dense_dim=embedder.dimension,
    )

    all_chunks = []
    for path in Path(directory).glob("*.md"):
        text = path.read_text()
        all_chunks.extend(
            chunk_document(text, path.name, chunk_size_tokens, chunk_overlap_tokens)
        )

    if not all_chunks:
        print(f"No .md files found in {directory}")
        return

    print(f"Chunked into {len(all_chunks)} chunks, embedding...")
    texts = [c.text for c in all_chunks]
    dense_vectors = embedder.embed_documents(texts)
    sparse_vectors = sparse.embed_documents(texts)
    store.upsert(all_chunks, dense_vectors, sparse_vectors)
    print(f"Ingested {len(all_chunks)} chunks from {directory} into '{store._collection}'")


def ingest_directory_parent_child(
    directory: str,
    collection: str,
    parent_size_tokens: int | None = None,
    parent_overlap_tokens: int | None = None,
    child_size_tokens: int = 60,
    child_overlap_tokens: int = 15,
) -> None:
    """Parent-Child ingestion (ADR-003): parents use the same boundaries as
    ingest_directory() (so existing ground truth stays valid unchanged);
    children are smaller sub-splits embedded for precise matching, but
    retrieval returns the parent's full text."""
    parent_size_tokens = parent_size_tokens or settings.chunk_size_tokens
    parent_overlap_tokens = parent_overlap_tokens or settings.chunk_overlap_tokens

    embedder = get_embedding_provider()
    sparse = BM25SparseProvider()
    store = QdrantHybridStore(
        url=settings.qdrant_url, collection=collection, dense_dim=embedder.dimension
    )

    all_children = []
    parents_by_key = {}  # (source, chunk_index) -> Chunk
    for path in Path(directory).glob("*.md"):
        text = path.read_text()
        parents, children = chunk_document_with_children(
            text, path.name, parent_size_tokens, parent_overlap_tokens,
            child_size_tokens, child_overlap_tokens,
        )
        for p in parents:
            parents_by_key[(p.source, p.chunk_index)] = p
        all_children.extend(children)

    if not all_children:
        print(f"No .md files found in {directory}")
        return

    print(f"Chunked into {len(parents_by_key)} parents / {len(all_children)} children, embedding...")
    texts = [c.text for c in all_children]
    dense_vectors = embedder.embed_documents(texts)
    sparse_vectors = sparse.embed_documents(texts)
    store.upsert_children(all_children, parents_by_key, dense_vectors, sparse_vectors)
    print(f"Ingested {len(all_children)} children ({len(parents_by_key)} parents) "
          f"from {directory} into '{store._collection}'")


if __name__ == "__main__":
    directory = sys.argv[1] if len(sys.argv) > 1 else "data/sample_docs"
    collection = sys.argv[2] if len(sys.argv) > 2 else None
    ingest_directory(directory, collection)
