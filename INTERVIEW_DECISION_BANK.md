# Interview Decision Bank

## Research Copilot — Vector Store
**Chosen:** Qdrant
**Rejected:** pgvector, Weaviate, Pinecone, local FAISS
**One-line why:** Optimizing for learning depth and interview signal (not complexity) — Qdrant is a dedicated vector DB with a transparent API, teaching HNSW/hybrid fusion/quantization mechanics that a black box (Pinecone) or SQL-adjacent tool (pgvector) wouldn't.
**"When would you use X instead?" answer:** pgvector if already on Postgres and want one fewer service at modest scale (<a few million chunks). Weaviate if you want built-in hybrid/reranking modules and are fine with GraphQL and more abstraction. Pinecone if you have budget and want zero infra ownership, prioritizing shipping speed over understanding retrieval internals.

## Research Copilot — Embedding Model
**Chosen:** Local `bge-base-en-v1.5` (default) + Gemini free-tier `text-embedding-004` (configurable alternative), behind a shared `EmbeddingProvider` interface
**Rejected (as defaults):** Paid API embeddings (OpenAI/Cohere), free-tier-API-only (no local model)
**One-line why:** Local gives free unlimited iteration for eval tuning and teaches serving mechanics; Gemini free tier exercises the provider-swap pattern and real rate-limit retry/backoff; both live behind one interface so switching is a config change.
**"When would you use X instead?" answer:** Paid API (OpenAI/Cohere) once quality matters more than per-call cost, e.g. production deployment or the final cost/quality comparison table. Free-tier-API-only if the team has no interest in ever self-hosting inference and embedding volume is low.

## Research Copilot — Chunking Strategy
**Chosen:** Recursive/structure-aware chunking (baseline) + Contextual Retrieval and Parent-Child chunking as eval-tested upgrade layers
**Rejected:** Pure semantic chunking, Late Chunking, fixed-size chunking
**One-line why:** Ship the cheap, coherent baseline first, then quantify whether Anthropic's Contextual Retrieval (LLM-generated context per chunk) and/or Parent-Child (small chunks matched, larger context returned) actually improve RAGAS scores before committing to either.
**"When would you use X instead?" answer:** Pure semantic chunking if the corpus has sparse/unreliable structural markers (long-form essays mixing topics). Late Chunking if already standardized on a long-context embedding model with exposed token embeddings (e.g., Jina) — blocked here because our chosen local `bge-base` model doesn't expose pre-pooling token embeddings. Fixed-size only for a throwaway prototype.

## Research Copilot — Reranker
**Chosen:** Local `bge-reranker-base` cross-encoder (default) + Cohere Rerank free tier (configurable alternative), same `RerankerProvider` interface pattern as embeddings
**Rejected:** LLM-as-reranker as a runtime option (kept as one-time eval ablation only), no reranker
**One-line why:** Dedicated cross-encoder gives free/unlimited, low-latency query-time reranking; LLM-as-reranker is too slow/costly per query to run live, so it's measured once offline instead of wired into the runtime config.
**"When would you use X instead?" answer:** Cohere (or another paid rerank API) once quality matters more than the free tier's quota limits, e.g. production traffic. LLM-as-reranker only in a low-QPS system where nuanced instructable judgment outweighs latency/cost concerns.

## Research Copilot — Agentic Loop
**Chosen:** Hand-rolled loop using Gemini's native function-calling API, single `search` tool
**Rejected:** LangGraph, prompt-only JSON action parsing
**One-line why:** A single-tool loop doesn't earn a graph framework's complexity yet; hand-rolling it means understanding the actual mechanics (message history, iteration cap) instead of inheriting a framework's abstraction.
**"When would you use X instead?" answer:** LangGraph once the agent needs multiple tools and non-linear branching (e.g. Project 2's HITL support agent). Prompt-only JSON parsing if the LLM provider lacks reliable native function calling.

## Research Copilot — Citation Synthesis
**Chosen:** Structured output with a per-claim citation schema (`source_chunk_ids` per claim) + deterministic post-check against the retrieved set
**Rejected:** Self-reported inline `[1]`/`[2]` markers alone, post-hoc NLI/embedding attribution (deferred, not rejected outright)
**One-line why:** Schema-enforced citations are machine-checkable by construction — you can programmatically verify every citation and flag/suppress unsupported claims, instead of trusting free-form inline markers the model can misplace.
**"When would you use X instead?" answer:** Post-hoc NLI/embedding attribution once eval numbers show schema-enforced self-reported citations are frequently valid-but-unsupported (chunk exists but doesn't entail the claim) — added as a verification layer, not a replacement.

## Research Copilot — Chunk Size Tuning (ADR-003 addendum)
**Chosen:** 200 tokens / 40 overlap (down from the initial 500/75 placeholder)
**Rejected:** keeping the original 500-token default unvalidated
**One-line why:** diagnosed via a retrieval ablation that showed Hit@5=32.1% at 500 tokens; inspecting actual misses found target facts buried 75-80% into oversized chunks, diluting the embedding signal — re-testing at 200 tokens raised Hit@5 to 53.6% on the same 28-question ground truth, confirming the diagnosis with a number instead of a guess.
**"When would you use X instead?" answer:** Keep chunks larger (500+ tokens) for corpora with genuinely long, single-topic passages where splitting further would fragment coherent reasoning rather than isolate facts — dense financial filings with many short, structurally-similar facts packed together are the case that specifically benefits from smaller chunks; a narrative-prose corpus might not show the same gain.

## Research Copilot — Gemini Model Selection & Reliability Hardening
**Chosen:** `gemini-flash-lite-latest` (non-thinking model, <1.5s/call) + retry-on-`DeadlineExceeded` + fail-fast on daily quota
**Rejected:** `gemini-flash-latest` (silently resolved to a "thinking" model variant, ~12s/call even for trivial prompts), retrying every error type identically, letting per-question exceptions crash the whole eval run
**One-line why:** a model alias can silently start resolving to a slower underlying model with no code change on your end — diagnosed by testing progressively lower-level layers (bare curl, then direct API calls, then inspecting response metadata) until a `thoughtsTokenCount` field in the response explained the 12s latency; fixed by switching to the "lite" alias and confirming both speed and function-calling still worked before trusting it in the eval.
**"When would you use X instead?" answer:** The thinking-model variant is the right choice when answer quality on genuinely hard reasoning matters more than latency/cost — e.g., a single high-stakes synthesis call, not a multi-call agent loop where latency compounds. Retrying identically regardless of error type is fine only for a single error class; once you have both a transient error (rate limit, 504) and a permanent one (daily quota) in the same API, they need different handling or you either waste time retrying the unrecoverable one or give up too early on the recoverable one.

## Research Copilot — Entailment Checking for Citations
**Chosen:** LLM-based entailment check (one batched Gemini call per question, checking all claims' citations at once)
**Rejected:** cross-encoder reranker repurposed as an entailment scorer
**One-line why:** calibrated the reranker on short, clean test snippets and got a clean separation (0.026 bad vs. 0.35-0.9996 good) — but with realistic full 200-token chunk text, a confirmed-bad citation and a confirmed-good citation both scored 0.97+, since a reranker measures topical relevance ("same company, same date"), not entailment ("does this text state this specific fact"); an LLM-based check, tested the same way, correctly separated true/false.
**"When would you use X instead?" answer:** A reranker is the right tool when you need to rank many candidates by relevance cheaply (its actual job); it's the wrong tool for a binary "does this specific text support this specific claim" judgment, which needs either a dedicated NLI model or an LLM prompted for entailment specifically — a real, generalizable lesson about not repurposing a relevance model for an entailment task just because both return a score.

## Research Copilot — Multi-Provider LLM Failover (ADR-007)
**Chosen:** shared OpenAI-compatible module (Groq/OpenRouter/SambaNova/Cerebras) with permanent mid-run failover from Gemini, plus a forced-single-provider mode for clean baselines
**Rejected:** per-call round-robin across all providers, a full provider-agnostic abstraction layer unifying Gemini's native SDK too
**One-line why:** round-robin would let different calls within the same question get answered by different models, making individual results incoherent to attribute — permanent failover (tagged per-result) keeps results attributable even when a run spans two providers; a full abstraction layer would have required refactoring Gemini's already-debugged native-SDK code for a same-day quota problem that didn't need it.
**"When would you use X instead?" answer:** Round-robin makes sense for a production system prioritizing raw throughput over per-request model consistency. The full abstraction layer becomes the right call once multi-provider usage is a standing feature (real provider selection, not just an occasional quota escape hatch) rather than a one-off fix.

## Research Copilot — Parent-Child Chunking Validation (ADR-003 addendum 2)
**Chosen:** not adopted — tested on 3 single-company subsets before committing to a full-corpus rollout, found mixed results (1 win, 1 wash, 1 real regression)
**Rejected:** committing to a full 18-file Parent-Child ingestion based on a single positive (WAL-only) test
**One-line why:** subset-testing on 3 companies caught a real regression (ZION, likely from 100-token children cutting off the segment name that disambiguates near-identical repeated sentences) that a single-company test would have missed entirely — validating small before committing hours of compute changed the actual decision, not just saved time.
**"When would you use X instead?" answer:** Worth revisiting with content-type-aware child sizing, or after directly diagnosing the ZION regression via retrieved-evidence inspection — the technique isn't ruled out permanently, just not adopted as configured against this corpus.

## Research Copilot — Cost/Latency Logging (ADR-009)
**Chosen:** Lightweight custom JSONL logger (`telemetry.py`, one `log_call()` function)
**Rejected:** OpenTelemetry spans, third-party SDK (Langfuse/Helicone), provider dashboards only
**One-line why:** Project 5 is explicitly scoped to build the full observability platform — pre-building OTel spans or adopting Langfuse here would either duplicate that work or undermine the "I built this myself" interview story; a flat JSONL log satisfies this project's cost-reporting bar and becomes Project 5's real ingestion source.
**"When would you use X instead?" answer:** OpenTelemetry once two+ services need shared hierarchical tracing (exactly Project 5's scope). Langfuse/Helicone for a real production team on a deadline that doesn't need the portfolio signal. Provider dashboards only for a throwaway prototype where cost/latency will never be reported as a deliverable.

## Research Copilot — RRF-Boosted Reranking (ADR-004 addendum)
**Chosen:** Combined score: 70% normalized reranker score + 30% RRF position bonus, with RERANK_TOP_N increased from 5 to 7
**Rejected:** Pure reranker ordering (old default), threshold-based fallback to RRF, just increasing top_N
**One-line why:** The cross-encoder scored all same-filing chunks 0.94-1.0, making top-5 selection effectively random — combining with RRF position breaks ties in favor of items both systems agree on. Hit@1 went from 19.4% to 32.3%, MRR from 0.286 to 0.397.
**"When would you use X instead?" answer:** Pure reranker ordering works fine when there's meaningful score separation between relevant and irrelevant chunks (different-topic corpora). Threshold-based fallback needs a corpus where some queries have compressed scores and others don't — ours were compressed across the board. Just increasing top_N helps recall but doesn't fix precision; the combined approach improves both.

## Research Copilot — UI Architecture (ADR-008)
**Chosen:** FastAPI backend + static HTML/JS frontend (no build step) as a walking skeleton
**Rejected:** React/Next.js (CLAUDE.md's originally suggested stack, deferred as an explicit later layer), Streamlit/Gradio
**One-line why:** same incremental-build logic applied to every other component this session — prove the API contract works end-to-end before adding a second new toolchain (Node/React) on top of it; Streamlit/Gradio were rejected as the weakest portfolio signal, demonstrating neither API design nor frontend engineering.
**"When would you use X instead?" answer:** React/Next.js once the API-first version is proven stable and the portfolio value of demonstrating frontend framework skill outweighs the added setup time. Streamlit/Gradio for the fastest possible internal demo where portfolio signal doesn't matter.
