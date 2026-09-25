# ADR-006: LLM Synthesis with Citations for Research Copilot

## Context
Once the agentic loop (ADR-005) retrieves and reranks chunks, the LLM must synthesize an answer where every claim is traceable to a source chunk, and hallucinated or uncited claims are flagged or suppressed rather than silently included — an explicit project requirement, and the property the RAGAS faithfulness metric will later measure quantitatively.

## Options Considered

### Option A: Prompt-based inline citation (self-reported `[1]`/`[2]` markers) — rejected as the sole mechanism
- How it works: Prepend each retrieved chunk with a visible ID (`[doc1]`, `[doc2]`, ...) in the prompt; instruct the model to cite using those IDs inline as it writes the answer.
- Pros: Simplest to implement, cheap (one LLM call, no schema enforcement needed), works with plain text generation.
- Cons: Self-reported citations are exactly the kind of thing an LLM gets wrong under pressure — it can place a plausible-looking `[2]` next to a claim that chunk doesn't actually support, and nothing catches that without a separate verification step.
- Why we didn't use it here alone: No enforcement mechanism means "traceable or flagged" becomes aspirational rather than actually guaranteed.
- When it WOULD be sufficient: A low-stakes prototype where citation accuracy isn't being measured or enforced.

### Option B: Structured output with a per-claim citation schema — CHOSEN
- How it works: Force schema-valid JSON via function-calling/structured-output mode: `{"claims": [{"text": ..., "source_chunk_ids": [...]}]}`. Each claim in the answer is a discrete object naming which chunk IDs support it, rather than an inline bracket the model can drop or misplace. After generation, verify every `source_chunk_id` referenced actually exists in the retrieved chunk set (cheap, deterministic check); flag or suppress any claim with an empty `source_chunk_ids` list instead of including it silently.
- Pros: Machine-checkable by construction — citations can be programmatically validated against the actual retrieved set; straightforward to compute "% of claims with at least one valid citation" as an eval metric; forces explicit attribution rather than free-form, best-effort citation.
- Cons: More prompt/parsing complexity than Option A; breaks the answer into discrete claim objects that must be reassembled into flowing prose for display.
- Cost/latency/complexity profile: One LLM call (same as Option A) plus a cheap deterministic post-check; moderate added implementation complexity for the schema and reassembly step.

### Option C: Post-hoc citation attribution (embedding similarity or NLI matching after generation) — rejected as default, kept as a future verification layer
- How it works: Let the model write the answer freely, then run a separate pass that matches each output sentence to its best-supporting retrieved chunk via embedding similarity or an NLI/entailment model.
- Pros: Doesn't rely on the generation-time model to self-report correctly at all; can catch fully unsupported sentences the model never intended to cite.
- Cons: Adds a full extra pipeline stage (embedding/NLI pass over every output sentence); attribution can still be wrong in the other direction — falsely matching a chunk on topical similarity without genuine entailment; more moving parts than needed for the walking-skeleton stage.
- Why we didn't use it here as the default: Option B already provides deterministic, cheap verification of self-reported citations without a second model call; Option C's marginal benefit over B is best evaluated once the RAGAS eval harness exists, since RAGAS's faithfulness metric performs a similar entailment-style check already.
- When it WOULD be the better choice: If eval results show schema-enforced self-reported citations (Option B) are frequently valid-but-unsupported (i.e., the cited chunk exists but doesn't actually entail the claim) — at that point Option C becomes a legitimate additional verification layer, not a replacement for B.

## Decision
We implement structured-output generation with a per-claim citation schema (Option B): the LLM returns discrete claims each naming supporting chunk IDs, and a deterministic post-check validates every cited ID against the actual retrieved set, flagging/suppressing claims with no valid citation. Option C is documented as a candidate future verification layer once RAGAS faithfulness scoring is in place to justify the added complexity with real numbers.

## Consequences
We take on the added complexity of schema-constrained generation and reassembling claim objects into a coherent displayed answer, versus the simplicity of free-form inline citation. In exchange we get a citation mechanism that's actually enforceable and measurable rather than best-effort, and a clean hook for the eval harness to compute citation-validity metrics directly. We give up, for now, the stronger guarantee Option C could offer (catching topically-plausible-but-unsupported claims); we'd add it once eval numbers justify the extra pipeline stage.

## Interview-ready summary
"I used structured output to force the model to attach explicit source chunk IDs to each claim, then deterministically verified those IDs against the actual retrieved set — so an unsupported claim gets flagged or suppressed instead of silently shipped. I considered post-hoc citation attribution via embedding or NLI matching, which is a stronger check because it doesn't trust the model's self-report at all, but I held that back until I had eval numbers showing self-reported citations were actually a problem worth the added pipeline stage — measuring first, then adding complexity, rather than assuming I needed the heavier approach."

## Addendum: explicit `unable_to_answer` field added to the schema (2026-09)
The original schema (`{"claims": [...]}`) had no way for the model to cleanly signal "I can't
answer this" — an honest abstention and a fabricated-but-unsupported claim both showed up
identically as `supported=False`. The adversarial eval in `citation_verification.py` inferred
abstention as "zero supported claims," which meant a claim the model *attempted* but that
failed the entailment check scored as "correctly abstained" — the metric couldn't distinguish
an honest hedge from a fabrication that happened to get caught. This is called out in
`KNOWN_TRADEOFFS.md`'s failure-analysis writeup.

Fix: the schema now includes `{"unable_to_answer": bool, "claims": [...]}`. The model is
instructed to set `unable_to_answer=true` and leave `claims` empty when it genuinely can't
answer from the evidence, rather than inventing a hedge claim. `Claim` gained a matching
`unable_to_answer: bool = False` field (default preserves old behavior for any code that
doesn't check it). `citation_verification.py`'s `correctly_abstained` now requires this
explicit signal (or zero claims returned at all) rather than inferring it from
`num_supported == 0` — tightening the metric, which may lower the *measured* adversarial
accuracy number without any change in actual system behavior, since it now requires the
model to have been honest about *why* it produced no supported claims.
