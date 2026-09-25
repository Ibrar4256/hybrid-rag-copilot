# ADR-004: Reranker for Research Copilot

## Context
Hybrid retrieval (ADR-001, dense + BM25 via Qdrant) returns a broad top-k candidate set by cosine/fusion score alone. CLAUDE.md explicitly requires a reranker step ("with a reranker, not just raw cosine similarity top-k") as the quality gate before candidates reach the LLM — this is the layer most responsible for context precision in the RAGAS eval harness. Reranking runs at query time (unlike Contextual Retrieval in ADR-003, which runs at ingestion time), so latency cost here directly affects user-facing response time.

## Options Considered

### Option A: Local cross-encoder (`BAAI/bge-reranker-base`, `sentence-transformers` `CrossEncoder`) — CHOSEN as default
- How it works: Runs locally; scores (query, chunk) pairs jointly through a transformer (true cross-attention between query and candidate, not separate embeddings compared post-hoc), producing a relevance score used to re-sort candidates.
- Pros: Free, unlimited, no rate limits — consistent with ADR-002's iteration-speed rationale; same `bge` model family as our embedder, giving a coherent stack; strong quality for its size; real hands-on learning of why cross-encoders outperform bi-encoder cosine similarity.
- Cons: Slower per-query than a bi-encoder since the full pair must be processed jointly (can't precompute); CPU inference adds latency directly to the query path.
- Cost/latency/complexity profile: Free; latency acceptable at portfolio-scale top-k (e.g., reranking 20-50 candidates); low complexity via `sentence-transformers`.

### Option B: Cohere Rerank API (`rerank-english-v3.0`, free trial tier) — CHOSEN as configurable alternative
- How it works: Send query + candidate texts to Cohere's hosted, purpose-built reranker; receive reordered relevance scores.
- Pros: Best-in-class rerank quality, zero local compute, low latency on Cohere's side; useful for the cost/quality comparison table; exercises the same provider-agnostic swap pattern established in ADR-002.
- Cons: Free trial tier is time/quota-limited (not a permanent free tier), adds network latency to the query path, another provider dependency.
- Cost/latency/complexity profile: Free within trial quota; added network round-trip; requires the same retry/backoff pattern as other API providers.
- Why chosen as configurable, not default: Time-limited trial quota makes it unsuitable as the sole reranker for ongoing development; kept available via config for the cost/quality comparison table and to validate the provider-swap architecture on a second component (embeddings and reranking both now follow this pattern).

### Option C: LLM-as-reranker (prompt an LLM to score/reorder candidates) — rejected as a runtime option, kept as a one-time eval ablation
- How it works: Reuse the existing LLM (Gemini/Groq) to score each candidate chunk's relevance to the query (e.g., "rate relevance 0-10"), then sort by score.
- Pros: No new model/infra, can incorporate more nuanced instructable judgment (e.g., penalize outdated info); genuinely interesting comparison point — does a general LLM reranker beat a dedicated cross-encoder here?
- Cons: Far slower and costlier at query time than a dedicated cross-encoder (N chunks means N or batched LLM calls per query, on the critical path); less calibrated/consistent scoring than a purpose-trained reranker; consumes rate-limited free-tier quota per query rather than per ingestion.
- Why we didn't use it here as a default: Query-time latency and cost make it impractical as a runtime path; the question of whether it outperforms a dedicated cross-encoder is worth answering once, quantitatively, but not worth paying for on every request.
- How we use it instead: Run once as an offline ablation against the same golden set used for RAGAS evaluation, report the accuracy delta vs. Option A, and stop there — it does not get a config flag in the running system.
- When it WOULD be the better choice: A low-QPS system where nuanced, instructable relevance judgment matters more than latency, and query volume is low enough that per-query LLM cost is a non-issue.

### Option D: No reranker (hybrid retrieval score used directly) — rejected
- How it works: Skip reranking; pass hybrid retrieval's top-k directly to the LLM.
- Pros: Simplest, fastest, zero added latency.
- Cons: Gives up the single biggest lever on context precision; directly contradicts the project's stated requirement for a reranking stage; not defensible as production-grade retrieval.
- Why we didn't use it here: The project explicitly requires a reranker; skipping it isn't a genuine option for this system's scope.

## Decision
We implement a `RerankerProvider` interface, mirroring ADR-002's `EmbeddingProvider` pattern, with:
- `LocalCrossEncoderProvider` (`bge-reranker-base`) — **default**, used for all day-to-day development and eval-harness iteration.
- `CohereRerankProvider` (`rerank-english-v3.0`, free trial) — configurable alternative, selected via config, for cost/quality comparison and to validate the provider-swap pattern on a second component.

LLM-as-reranker (Option C) is implemented as a standalone, one-time evaluation script — not part of the `RerankerProvider` interface or the runtime config — used solely to produce a comparison data point in the eval report.

## Consequences
We take on the query-time latency of running a cross-encoder pass over top-k candidates (mitigated by keeping k modest, e.g. 20-50 candidates reranked down to 5-10). We give up the plug-and-play convenience of skipping reranking entirely, and we accept that Cohere's free tier isn't a permanent fallback (it's a comparison/demo path, not a long-term default). In exchange we get: a coherent local model stack (`bge` embedder + `bge` reranker), a second working example of the provider-swap architecture, and a quantified answer to "does an LLM reranker beat a dedicated cross-encoder" without paying its query-time cost in production. We'd revisit this if query volume grew large enough that cross-encoder latency became the dominant bottleneck, at which point a smaller/distilled reranker or Cohere's paid tier might become the default.

## Addendum: RRF-boosted reranking (Phase 4 optimization)

**Problem diagnosed:** The cross-encoder scores same-filing chunks in a compressed
0.94-1.0 band with almost no separation. When the correct chunk scores 0.98 and five
other chunks from the same filing score 0.985-0.999, the top-5 cutoff arbitrarily
excludes the correct result. Net effect: reranking rescued 5 questions from deep in the
candidate set (rank 8-19 → top-5) but randomly ejected 3 questions that RRF had already
ranked #1 (rank 1 → rank 6-8). The reranker helped Hit@5 but hurt Hit@1 and MRR.

**Four strategies tested (all offline, pure retrieval, no LLM calls):**

| Strategy | Hit@1 | Hit@5/7 | MRR |
|---|---|---|---|
| No reranker (RRF only, top-5) | 25.8% | 41.9% | 0.317 |
| Pure reranker (top-5, old default) | 19.4% | 48.4% | 0.286 |
| Pure reranker (top-7) | 19.4% | 54.8% | 0.299 |
| Threshold-based (top-5) | 19.4% | 48.4% | 0.286 |
| **RRF-boosted (top-5)** | **32.3%** | **54.8%** | **0.391** |
| **RRF-boosted (top-7) — CHOSEN** | **32.3%** | **58.1%** | **0.398** |

**How RRF-boosted works:** Normalize reranker scores to [0,1], compute an RRF position
bonus (1/rank for each candidate's hybrid-search position), combine as
`0.7 * reranker_norm + 0.3 * rrf_bonus`, sort by combined score. This preserves the
reranker's ability to rescue correct chunks from deep in the candidate set while
anchoring to RRF's ordering when reranker scores are compressed.

**Why threshold-based didn't work:** Score separation threshold of 0.3 was too high —
nearly all questions had compressed scores, so it fell through to pure reranking in
almost every case.

**Why top-7 over top-5:** Sends 2 more chunks per search round to the agent loop (up to
28 vs 20 across 4 rounds). The token cost is minimal relative to Gemini's context window,
and Hit@7 captures 3.3pp more correct results than Hit@5.

**Implemented in:** `query.py` — `Retriever._rerank_rrf_boosted()` replaces the pure
`reranker.rerank()` call. Constants `RERANKER_WEIGHT=0.7`, `RRF_POSITION_WEIGHT=0.3`,
`RERANK_TOP_N=7` in the same file.

## Interview-ready summary
"I used a local `bge-reranker-base` cross-encoder as the default reranker, reusing the same provider-agnostic interface pattern I built for embeddings, with Cohere's rerank API wired in as a configurable alternative for a cost/quality comparison. I also tested prompting an LLM to rerank candidates directly, as a one-time ablation rather than a runtime path — it's flexible but far too slow and expensive per query to use live, so I only ran it once against my eval set to see whether it beat a dedicated cross-encoder. That's the kind of thing worth measuring once, not something you'd want on your hot path."

"After running the full ablation, I found the cross-encoder was actually hurting Hit@1 — it scored all same-filing chunks 0.94-1.0, so the top-5 cutoff was basically a random shuffle. I tested four strategies: just increasing top-N, threshold-based fallback to RRF, and a combined score that weights 70% reranker + 30% RRF position. The combined approach won on every metric — Hit@1 went from 19% to 32%, MRR from 0.29 to 0.40. The key insight is that when a reranker's scores are compressed, you need a tiebreaker, and the original retrieval ordering is a good one because it's computed from a completely different signal (embedding similarity + BM25 term overlap)."
