import uuid

from qdrant_client import QdrantClient, models

from research_copilot.chunking import ChildChunk, Chunk


UPSERT_BATCH_SIZE = 200


def _point_to_result(payload: dict, score: float) -> dict:
    # Parent-Child collections (ADR-003) store child text for matching but
    # parent text/index for what's actually returned to the LLM — this keeps
    # the Retriever-facing result shape identical regardless of which mode a
    # collection was ingested with.
    if "parent_index" in payload:
        return {
            "text": payload["parent_text"],
            "source": payload["source"],
            "chunk_index": payload["parent_index"],
            "score": score,
        }
    return {
        "text": payload["text"],
        "source": payload["source"],
        "chunk_index": payload["chunk_index"],
        "score": score,
    }


def _dedupe_by_chunk(results) -> list[dict]:
    """In Parent-Child mode, multiple matching children can map to the same
    parent — keep each parent only once, at its best (first, since results
    arrive score-sorted) matching child's score."""
    seen = set()
    deduped = []
    for r in results:
        key = (r["source"], r["chunk_index"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(r)
    return deduped


class QdrantHybridStore:
    def __init__(self, url: str, collection: str, dense_dim: int):
        self._client = QdrantClient(url=url, timeout=120)
        self._collection = collection
        self._dense_dim = dense_dim
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        if self._client.collection_exists(self._collection):
            return
        self._client.create_collection(
            collection_name=self._collection,
            vectors_config={
                "dense": models.VectorParams(
                    size=self._dense_dim, distance=models.Distance.COSINE
                )
            },
            sparse_vectors_config={"sparse": models.SparseVectorParams()},
        )

    def upsert(
        self,
        chunks: list[Chunk],
        dense_vectors: list[list[float]],
        sparse_vectors: list,
    ) -> None:
        points = []
        for chunk, dense, sparse in zip(chunks, dense_vectors, sparse_vectors):
            points.append(
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector={
                        "dense": dense,
                        "sparse": models.SparseVector(
                            indices=sparse.indices.tolist(),
                            values=sparse.values.tolist(),
                        ),
                    },
                    payload={
                        "text": chunk.text,
                        "source": chunk.source,
                        "chunk_index": chunk.chunk_index,
                    },
                )
            )
        self._upsert_points(points)

    def upsert_children(
        self,
        children: list[ChildChunk],
        parents_by_key: dict[tuple[str, int], Chunk],
        dense_vectors: list[list[float]],
        sparse_vectors: list,
    ) -> None:
        """Parent-Child ingestion (ADR-003): embeds/indexes CHILD text (for
        precise matching) but stores the corresponding PARENT's full text in
        the payload (what actually gets returned to the LLM).
        parents_by_key is keyed by (source, chunk_index) — chunk_index alone
        resets per source file, so a plain int key would collide across a
        multi-file corpus."""
        points = []
        for child, dense, sparse in zip(children, dense_vectors, sparse_vectors):
            parent = parents_by_key[(child.source, child.parent_index)]
            points.append(
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector={
                        "dense": dense,
                        "sparse": models.SparseVector(
                            indices=sparse.indices.tolist(),
                            values=sparse.values.tolist(),
                        ),
                    },
                    payload={
                        "child_text": child.text,
                        "source": child.source,
                        "child_index": child.child_index,
                        "parent_index": child.parent_index,
                        "parent_text": parent.text,
                    },
                )
            )
        self._upsert_points(points)

    def _upsert_points(self, points: list) -> None:
        # Sending thousands of points in one request can exceed Qdrant's write
        # timeout (hit in practice ingesting the SEC filings corpus) — batch instead.
        for i in range(0, len(points), UPSERT_BATCH_SIZE):
            batch = points[i : i + UPSERT_BATCH_SIZE]
            self._client.upsert(collection_name=self._collection, points=batch)
            print(f"  upserted {min(i + UPSERT_BATCH_SIZE, len(points))}/{len(points)} points")

    def dense_search(self, dense_query: list[float], limit: int) -> list[dict]:
        """Dense-only search, bypassing BM25/RRF fusion entirely — used as the
        ablation baseline (ADR-001's ablation table: dense-only vs +hybrid vs
        +reranking)."""
        result = self._client.query_points(
            collection_name=self._collection,
            query=dense_query,
            using="dense",
            limit=limit,
        )
        return _dedupe_by_chunk(_point_to_result(p.payload, p.score) for p in result.points)

    def hybrid_search(
        self, dense_query: list[float], sparse_query, limit: int
    ) -> list[dict]:
        result = self._client.query_points(
            collection_name=self._collection,
            prefetch=[
                models.Prefetch(
                    query=dense_query, using="dense", limit=limit * 2
                ),
                models.Prefetch(
                    query=models.SparseVector(
                        indices=sparse_query.indices.tolist(),
                        values=sparse_query.values.tolist(),
                    ),
                    using="sparse",
                    limit=limit * 2,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
        )
        return _dedupe_by_chunk(_point_to_result(p.payload, p.score) for p in result.points)
