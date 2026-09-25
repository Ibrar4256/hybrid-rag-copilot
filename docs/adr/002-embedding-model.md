# ADR-002: Embedding Model for Research Copilot

## Context
Retrieval quality is bounded by embedding quality. This is a learning-focused project — we want unlimited free iteration during eval/chunking tuning, but also want to build a real production pattern: a provider-agnostic interface so swapping embedding backends is a config change, not a refactor (explicitly called for in the project's cost/architecture guidelines, alongside retry/backoff for free-tier rate limits).

## Options Considered

### Option A: Local open embedding model (`BAAI/bge-base-en-v1.5` via `sentence-transformers`) — CHOSEN as default
- How it works: Runs on local CPU/GPU via `sentence-transformers`, no network call, no API key.
- Pros: Free and unlimited — critical since the golden set gets re-embedded repeatedly while tuning chunking/reranking; fully reproducible and offline; teaches embedding-model mechanics (batching, pooling, normalization) that transfer to Project 3's fine-tuning/serving work.
- Cons: Slightly lower ceiling than top proprietary models; CPU inference is slow on large corpora; we own the serving code.
- Cost/latency/complexity profile: Free; latency acceptable at portfolio-scale corpora; low complexity via `sentence-transformers`.

### Option B: Gemini free-tier embedding API (`text-embedding-004`) — CHOSEN as configurable alternative
- How it works: HTTP call to Google's Gemini embedding endpoint, free tier with rate limits.
- Pros: No local compute needed, decent quality, useful for the end-of-project cost/quality comparison table, exercises the provider-agnostic interface and forces real retry/backoff logic for free-tier rate limits (a genuine production concern).
- Cons: Rate-limited (unsuitable as the sole embedding source during heavy iteration), network dependency, free tier terms can change.
- Cost/latency/complexity profile: Free within rate limits; added network latency; requires retry/backoff handling.

### Option C: Paid API embedding (OpenAI `text-embedding-3-*`, Cohere `embed-v3`) — rejected (for now)
- How it works: Same as B but paid, higher quality ceiling.
- Pros: Best-in-class quality, generous throughput.
- Cons: Costs money per call; re-embedding the golden set during tuning would add up; teaches HTTP plumbing more than embedding internals.
- Why we didn't use it here: Not needed given free local + free-tier API already covers the "config-swap" and "compare providers" goals; may be added later purely to populate the frontier-model cost/quality row.
- When it WOULD be the better choice: Final one-time comparison run, or a production deployment where quality matters more than per-call cost.

### Option D: Free-tier hosted API only (no local model) — rejected
- How it works: Rely entirely on Gemini/other free tiers, skip local serving.
- Pros: Zero local compute, simplest to start.
- Cons: Rate limits throttle eval-iteration speed; teaches nothing about serving embedding models; single point of failure if free tier changes.
- Why we didn't use it here: Directly conflicts with the project's learning goal and with needing unlimited-iteration embedding during tuning.
- When it WOULD be the better choice: A team with no interest in ever self-hosting inference and low embedding volume.

## Decision
We implement an `EmbeddingProvider` interface (protocol/abstract base) with two concrete implementations selected via config, not code changes:
- `LocalBGEProvider` (`bge-base-en-v1.5`, `sentence-transformers`) — **default**, used for all day-to-day development, chunking experiments, and eval-harness iteration.
- `GeminiEmbeddingProvider` (`text-embedding-004`, free tier) — configurable alternative, used to validate the provider-swap pattern, to test retry/backoff under real rate limits, and to populate the eventual cost/quality comparison table.

Both providers implement the same `embed(texts: list[str]) -> list[Vector]` contract so Qdrant ingestion and query-time embedding code never know which backend is active.

## Consequences
We take on the cost of building and maintaining a thin abstraction layer plus retry/backoff logic for the Gemini path — a small amount of extra code for a real architectural pattern worth demonstrating. We give up the raw simplicity of hardcoding one provider. We gain: free unlimited local iteration, a working example of provider-swap-by-config, real experience handling rate-limited free-tier APIs, and a straightforward path to compare local vs. API embeddings quantitatively later. We'd revisit this if local inference became a bottleneck (e.g., GPU unavailable) — in which case Gemini could become the default rather than the alternative.

## Interview-ready summary
"I built the embedding layer behind a provider interface with two implementations: a local `bge-base-en-v1.5` model as the default for free, unlimited iteration during eval tuning, and Gemini's free-tier embedding API as a configurable alternative — swappable via config, not a refactor. This let me get hands-on with serving an embedding model myself while also building real retry/backoff logic for free-tier rate limits, and sets up an easy cost/quality comparison later. I'd reach for a paid API like OpenAI's embeddings if quality mattered more than per-call cost in production, or skip local serving entirely on a team with no interest in ever owning inference infrastructure."
