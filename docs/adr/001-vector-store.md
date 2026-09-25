# ADR-001: Vector Store for Research Copilot

## Context
Project 1 (Research Copilot) needs a retrieval layer supporting hybrid search (dense + BM25/sparse) with a downstream cross-encoder reranker. This is a learning-focused portfolio project, not a cost/complexity-constrained production system — we are explicitly optimizing for depth of understanding of retrieval infrastructure and relevance to real-world AI engineering job requirements, over minimizing moving parts. Must run free/self-hosted (no paid managed service required) so the full eval suite can be run repeatedly at no cost.

## Options Considered

### Option A: Qdrant — CHOSEN
- How it works: Purpose-built vector database (Rust), self-hosted via Docker, with native support for dense + sparse (BM25-style) vectors and Reciprocal Rank Fusion (RRF) hybrid search, HNSW indexing, payload filtering, and scalar/product quantization.
- Pros: Dedicated learning surface for HNSW tuning, native hybrid fusion (no hand-rolled merge logic), quantization for scale, clean REST/gRPC API, directly comparable to Pinecone/Weaviate in interviews, free via Docker Compose.
- Cons: Extra service to run and monitor; vectors/metadata live separately from any relational store, requiring sync logic if metadata also lives in Postgres.
- Cost/latency/complexity profile: Free self-hosted; low latency at portfolio-scale data volumes; moderate complexity (one more container, one more client library) — acceptable since complexity is not a constraint for this project.

### Option B: pgvector — rejected
- How it works: Postgres extension adding a vector column type + ANN indexes (IVFFlat/HNSW); BM25-style text search via Postgres `tsvector`, combined in a single SQL query.
- Pros: One database for everything (metadata, vectors, full-text), no second service, SQL is a skill already held, easy hybrid queries in one query.
- Cons: Shallower learning surface — mostly "SQL plus an extension" rather than dedicated vector-DB mechanics; ANN performance and hybrid fusion tooling are less mature than a dedicated vector DB; less of a distinct interview story since it's "Postgres" more than "a vector database."
- Why we didn't use it here: We're optimizing for learning breadth and interview signal, and pgvector teaches fewer new concepts than a dedicated vector DB.
- When it WOULD be the better choice: A small team already running Postgres, wanting to avoid operating an extra service, at a data volume of at most a few million chunks, where "one less moving part" outweighs deeper vector-DB expertise.

### Option C: Weaviate — rejected
- How it works: Self-hosted (Docker) vector DB with a GraphQL-first API, built-in modules for hybrid search, vectorization, and even reranking pipelines.
- Pros: Most "batteries included" of the three — can handle vectorization/reranking via modules with less glue code; strong native hybrid search support.
- Cons: GraphQL query surface is a less commonly required skill than REST/gRPC; the module system can obscure the underlying mechanics, which works against the goal of understanding retrieval internals deeply; heavier resource footprint.
- Why we didn't use it here: The built-in modules trade transparency for convenience — we want to hand-write the hybrid fusion and reranking steps ourselves to actually learn them, not have a module do it invisibly.
- When it WOULD be the better choice: A team that wants to minimize custom glue code and is fine with GraphQL and module-based abstractions, prioritizing shipping speed over understanding every internal step.

### Option D: Pinecone — rejected
- How it works: Fully managed SaaS vector database, serverless pricing, no self-hosting option.
- Pros: Best-in-class DX, zero infrastructure to operate, scales transparently, free tier available.
- Cons: Fully proprietary black box — no access to indexing internals, sharding, or quantization; learnings don't transfer to self-hosted operational skills; free tier has real limits.
- Why we didn't use it here: As a managed black box, it teaches the least about how vector retrieval actually works internally, which directly conflicts with this project's learning goal.
- When it WOULD be the better choice: A company with budget that wants to ship fast and has no interest in owning retrieval infrastructure — "buy, don't build" for teams without a dedicated infra function.

### Option E: Local FAISS / in-memory index — rejected
- How it works: Library-only ANN index (e.g., FAISS), no server process, embedded directly in the application.
- Pros: Zero infrastructure, fastest to start.
- Cons: No built-in BM25/hybrid search, no persistence or payload filtering story, doesn't demonstrate a "production-grade" system.
- Why we didn't use it here: Doesn't support the hybrid search requirement natively and isn't defensible as production-grade infrastructure for a portfolio project.
- When it WOULD be the better choice: A throwaway prototype or a benchmark harness where persistence and hybrid search are irrelevant.

## Decision
We chose **Qdrant** because the project's explicit goal is maximum learning depth and interview relevance, not minimal complexity. Qdrant gives hands-on experience with the mechanics that show up in real production RAG stacks — HNSW indexing, native hybrid fusion (RRF), quantization for scale — through a transparent API where we implement the retrieval logic ourselves rather than relying on framework magic (Weaviate) or SQL-adjacent conveniences (pgvector), while avoiding the fully-managed black box (Pinecone) that would teach the least.

## Consequences
We give up the operational simplicity of pgvector (one fewer service) and the built-in convenience modules of Weaviate. We take on running and understanding a second stateful service (Qdrant) alongside whatever relational store holds ticket/user metadata in later projects. This is acceptable because Docker Compose makes local operation trivial and free, and the deeper mechanics learned here transfer directly to Project 5 (observability) and general AI engineering interviews. We would revisit this decision if data volume grew into the tens of millions of chunks and operational overhead became the binding constraint, or if a team context demanded GraphQL/module-based tooling instead.

## Interview-ready summary
"I chose Qdrant over pgvector, Weaviate, and Pinecone because I wanted to actually learn how a dedicated vector database works — HNSW indexing, hybrid dense+sparse fusion via RRF, quantization — rather than treating retrieval as a black box or an afterthought bolted onto Postgres. pgvector would be the better call on a small team already running Postgres that wants one fewer service to operate. Weaviate is a strong alternative if you want built-in modules for vectorization and reranking and are fine with its GraphQL API and the abstraction that comes with it. Pinecone is likely what many funded startups actually use in production because it's fully managed and requires zero infra ownership — but that's exactly why it teaches the least about the internals, which was the point of this project."
