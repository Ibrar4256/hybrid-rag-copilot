# ADR-003: Chunking Strategy for Research Copilot

## Context
Chunking determines what gets embedded and retrieved, and is the most common source of RAG retrieval failure — too small loses meaning, too large dilutes relevance. This decision feeds directly into ADR-001 (Qdrant hybrid search operates on chunks) and ADR-002 (embedding model embeds chunks), and is measured quantitatively via the eval harness's context precision/recall scores. Per CLAUDE.md's "build incrementally" rule, we ship a walking-skeleton chunking approach first, then layer in upgrades measured against the golden eval set — not all five approaches at once.

## Options Considered

### Option A: Recursive / structure-aware chunking — CHOSEN (walking skeleton baseline)
- How it works: Split along natural document boundaries first (headers, paragraphs, sentences), recursing to smaller units only if a segment still exceeds the target size (~400-600 tokens, ~10-15% overlap).
- Pros: Respects document structure, strong topic coherence per chunk, predictable size, simple to implement and reason about, the actual default used by most production RAG systems.
- Cons: Still somewhat arbitrary at the paragraph level — no semantic awareness, just structural boundaries.
- Cost/latency/complexity profile: Free, cheap, low complexity — ideal first implementation.

### Option B: Contextual Retrieval (Anthropic technique) — CHOSEN (first upgrade layer, eval-compared against baseline)
- How it works: Before embedding, send each chunk plus the full document to an LLM, generate 1-2 sentences of context situating the chunk within the document, prepend that context to the chunk, then embed context+chunk together (and index the same contextualized text for BM25).
- Pros: Directly solves the "isolated chunk lost its referents" problem; Anthropic's published benchmarks show ~35% reduction in retrieval failure combined with reranking; works with any embedding model (pure text concatenation, no architecture requirement); highly citable in interviews with real published numbers to compare against.
- Cons: Requires one LLM call per chunk at ingestion time — real cost/latency at index-build time (not query time); mitigated via prompt caching and free-tier Gemini/Groq at portfolio scale.
- Cost/latency/complexity profile: Free at this scale (free-tier LLM), one-time ingestion cost, moderate implementation complexity (LLM call + prompt template + retry/backoff reuse from ADR-002's provider pattern).
- Why chosen: Highest signal-to-effort ratio of all upgrade options — directly citable, works with our existing local `bge` embeddings, no cascading changes to ADR-002.

### Option C: Hierarchical / Parent-Child chunking — CHOSEN (second upgrade layer, eval-compared against baseline)
- How it works: Index small "child" chunks (~100-200 tokens) for precise vector matching; each child links via metadata to a larger "parent" chunk/section. Retrieval searches children, then fetches/expands to parents (deduped) before passing to the LLM.
- Pros: Improves match precision (small chunks give focused embedding signal) while still supplying the LLM enough surrounding context to answer well; widely used in production (LlamaIndex `ParentDocumentRetriever`/`AutoMergingRetriever`).
- Cons: More moving parts — parent-child relationship metadata to store and maintain, a second retrieval step (match → fetch parent → dedupe), effectively doubles indexed content.
- Cost/latency/complexity profile: Free, but higher implementation complexity than A or B; storage roughly doubles (children + parents).
- Why chosen as a second, independent lever: Operates on context-window management rather than embedding content, so it's not redundant with Contextual Retrieval — the eval harness compares both upgrades independently and combined.

### Option D: Pure semantic chunking (sentence-embedding similarity boundaries) — rejected
- How it works: Compute sentence-level embeddings, cut a new chunk boundary wherever consecutive-sentence similarity drops below a threshold.
- Pros: Boundaries align with actual topic shifts rather than structural markers.
- Cons: Expensive (embeds every sentence just to decide boundaries, on top of the real embedding pass), non-deterministic chunk sizes complicate cost/latency planning, threshold-sensitive, and real-world gains over recursive chunking are often marginal per RAG literature.
- Why we didn't use it here: Marginal expected gain over Option A given added cost and complexity; not worth prioritizing over Contextual Retrieval or Parent-Child, which have clearer, more citable wins.
- When it WOULD be the better choice: A corpus with highly variable topic density per document (e.g., long-form essays mixing many unrelated topics) where structural markers (headers/paragraphs) are sparse or unreliable.

### Option E: Late Chunking (Jina AI technique) — rejected
- How it works: Run the entire long document through a long-context embedding model first (token-level embeddings computed with full-document self-attention), then mean-pool those token embeddings into chunks — so each chunk's vector already encodes full-document context before pooling.
- Pros: Elegantly solves referent-loss (e.g., "it grew 20% that year") at the embedding-architecture level rather than via LLM generation; cutting-edge (2024) technique with strong novelty signal.
- Cons: Requires a long-context embedding model that exposes pre-pooling token embeddings — `bge-base-en-v1.5` (ADR-002's chosen local model) does not support this; would require adopting a different embedding model (e.g., `jina-embeddings-v3`), cascading back into ADR-002; document length capped by the model's context window; less battle-tested than parent-child chunking.
- Why we didn't use it here: Blocked by our existing embedding model choice — switching models solely to enable this technique would undo ADR-002's rationale (free local iteration with a well-benchmarked general-purpose model) for a technique whose real-world gains over Contextual Retrieval are not yet well established.
- When it WOULD be the better choice: A project already standardized on a long-context embedding model with exposed token embeddings, or one where referent-loss is a dominant, measured failure mode that Contextual Retrieval doesn't adequately fix.

### Option F: Fixed-size chunking (naive baseline) — rejected
- How it works: Split text into fixed token/character windows with sliding overlap, ignoring structure.
- Pros: Trivial to implement, predictable size.
- Cons: Blindly cuts mid-sentence/mid-idea, splits tables/lists arbitrarily, hurts retrieval precision.
- Why we didn't use it here: Recursive chunking (Option A) achieves the same predictability with far better coherence at negligible extra implementation cost — no reason to accept fixed-size's downside.
- When it WOULD be the better choice: A quick throwaway prototype where implementation time is the only constraint that matters.

## Decision
We ship **Option A (recursive/structure-aware chunking)** as the walking-skeleton default, per CLAUDE.md's incremental-build rule. We then implement **Option B (Contextual Retrieval)** and **Option C (Parent-Child chunking)** as independent, config-swappable upgrade layers, and run all combinations (baseline, +Contextual Retrieval, +Parent-Child, +both) through the RAGAS eval harness on the golden set to report which upgrade(s) actually move context precision/recall and faithfulness, rather than assuming either wins. **Late Chunking (E)** and **pure semantic chunking (D)** are documented but not implemented, for the reasons above.

## Consequences
We take on real implementation and ingestion-time cost for Contextual Retrieval (one LLM call per chunk, free-tier rate limits requiring the retry/backoff pattern from ADR-002) and storage/complexity overhead for Parent-Child (doubled indexed content, a two-step retrieval path). In exchange we get a genuinely interesting, quantified eval story ("baseline vs. +Contextual Retrieval vs. +Parent-Child vs. both, measured on N questions") instead of a single unvalidated choice. We give up the novelty of Late Chunking and the topic-shift precision of semantic chunking; we'd revisit Late Chunking if we ever adopted a long-context embedding model for other reasons, and semantic chunking if the golden set eval showed structural boundaries were unreliable for our corpus.

## Interview-ready summary
"I started with recursive, structure-aware chunking as the baseline — it's what most production RAG systems use, and it's cheap and predictable. I then implemented two upgrade layers as an ablation: Anthropic's own Contextual Retrieval technique, which prepends an LLM-generated context sentence to each chunk before embedding, and Parent-Child chunking, which matches on small precise chunks but returns larger parent context to the LLM. I measured both independently and combined against the baseline using RAGAS on a golden question set, so the choice was backed by numbers, not intuition. I looked at Late Chunking too — it's a clever architecture-level fix for the same referent-loss problem — but it requires a long-context embedding model with exposed token embeddings, which my chosen local embedding model doesn't support, so I'd only reach for it if I were already standardized on a model like Jina's."

## Addendum: chunk-size tuning, empirically validated (post-eval-harness)

**Context:** the original recommendation (~400-600 tokens, ~10-15% overlap) was a general-RAG-practice placeholder, flagged in `KNOWN_TRADEOFFS.md` as "not values tuned against this project's own eval set." Once the 28-question hand-verified ground truth existed (see `src/research_copilot/eval/ground_truth.py`), we tuned it for real.

**How we found the problem:** the initial retrieval ablation (500-token chunks) showed the full pipeline (hybrid + reranking) hitting the correct chunk in the top-5 only 32.1% of the time — low enough to investigate before trusting any downstream citation-verification numbers built on top of it. We ran a diagnostic script comparing, per question, the ground-truth `chunk_id` against what was actually retrieved. The pattern: retrieved chunks were almost always from the *correct source file*, just the wrong `chunk_index` — ruling out a gross retrieval bug (wrong company/filing) and pointing at something chunk-level instead.

Inspecting the actual chunk text confirmed it: in every case checked, the ground-truth chunk was ~1,400-2,000 characters (the 500-token config) and the target fact sat 75-80% of the way through the chunk, preceded by unrelated content — e.g., Western Alliance's "insured deposit ratio strengthened from 45% to 73%" appeared only after a paragraph opening with "the recent volatility in the banking industry... capital and liquidity actions"; M&T's average-deposits table appeared only after a paragraph about "debt investment securities credit-related losses." A chunk's dense embedding is computed over its full text, so when the queried fact is a minority of the chunk's content, the embedding is dominated by the unrelated preceding text — diluting the exact signal a query about that fact needs.

**How we fixed it:** re-ingested the same corpus with `chunk_size_tokens=200, chunk_overlap_tokens=40` into a separate Qdrant collection (`sec_filings_banks_chunk200`), rebuilt ground truth against the new chunk boundaries (all 28 locators still resolved), and re-ran the identical ablation. Result: Hit@5 rose from 32.1% to 53.6% (a ~67% relative improvement), confirming the hypothesis rather than just plausibly explaining it. `config.py`'s defaults were updated to 200/40 accordingly.

**One unexpected secondary finding:** at 200 tokens, hybrid retrieval *without* reranking actually beat hybrid+reranking on MRR (0.354 vs. 0.320) and Hit@1 (28.6% vs. 21.4%), though reranking still won on Hit@5 (53.6% vs. 46.4%). This suggests the reranker's benefit (established at 500-token chunks in the original ablation, ADR-004) is partly compensating for poorly-isolated chunks rather than being a pure, unconditional improvement — worth a closer look during failure analysis rather than assuming ADR-004's conclusion holds unchanged at every chunk size.

**Consequence:** the 200-token config is now the default for all new ingestion. The original 500-token collection (`sec_filings_banks`) is kept only as the ablation baseline for comparison, not used going forward.

## Addendum 2: Parent-Child chunking tested, mixed result, not adopted

**Context:** with the retrieval harness proven out by the chunk-size fix above, we tested Option C (Parent-Child chunking) — parents kept at the existing 200-token boundaries (so ground truth needed zero rework), children sub-split to 100 tokens for more precise matching, with the parent's full text returned to the LLM once a child matches.

**What we found, testing 3 single-company subsets before committing to a full-corpus run:** WAL improved (Hit@5 1/6→2/6, MRR 0.167→0.200), CMA was roughly flat (Hit@5 5/6→5/6, MRR 0.708→0.750), and **ZION regressed** (Hit@5 3/5→1/5, MRR 0.267→0.100) — two previously-correct questions became complete misses. Leading hypothesis: ZION's filings repeat near-identical cross-segment phrasing (Zions Bank, CB&T, Amegy, NBAZ each reporting "income before income taxes decreased $X million" — the same boilerplate-similarity problem documented in the D6 interview-prep question), and a 100-token child chunk can be small enough to cut off *before* the segment name that disambiguates which of four near-identical sentences it is. Not yet confirmed by direct inspection of the regressed retrievals — a real next step if this is revisited.

**Decision: not adopted.** One win, one wash, one real regression across three companies is not evidence a corpus-wide rollout would help on net, and the full 18-file ingestion is a multi-hour job — validating on 3 subsets first (rather than committing to the full run on a single positive result) is what caught this before it shipped as a false positive. Parent-Child chunking, as configured here, is documented as a genuine negative/mixed result, not implemented as the production default. The code (`chunking.chunk_document_with_children`, `vector_store.upsert_children`, `ingest.ingest_directory_parent_child`) remains in the codebase for future revisiting — e.g., with content-type-aware child sizing, or after directly diagnosing the ZION regression — but is not wired into the default ingestion path.

**Interview-ready summary:** "I tested Parent-Child chunking on three single-company subsets before committing to a multi-hour full-corpus run, and found genuinely mixed results — a win on one company, a wash on another, and a real regression on a third, where smaller child chunks likely cut off the exact token that disambiguated otherwise near-identical repeated sentences. I didn't adopt it, and I think that's the correct call given the evidence, not a failure — validating on a subset first is exactly what caught the regression before it would have shipped as a false positive from a single lucky test."
