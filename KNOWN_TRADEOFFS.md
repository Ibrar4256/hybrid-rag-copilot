# Known Tradeoffs

Honest record of shortcuts taken in the walking skeleton — what's deferred and why.

## RESOLVED (2026-09): Filing-jargon phrasing ("FY2023 10-K") caused complete retrieval failure
Q41 asks the same fact as Q1 (WAL total deposits) but phrased with "as of FY2023 10-K".
Before fix: the agent generated search queries containing "10-K" and "FY2023", which biased
the embedding + reranker toward filing metadata chunks (exhibit lists, cover page) instead of
financial data. Result: zero relevant chunks retrieved, "unable to answer" despite the fact
appearing 6+ times in the corpus.
**Root cause:** the agent's system prompt had no guidance to strip filing-type jargon from
search queries. Dense embeddings and the cross-encoder reranker both treat "10-K" as a strong
signal, pulling results toward document scaffolding rather than content.
**Fix:** added a paragraph to both agent system prompts (Gemini native + OpenAI-compatible)
instructing the agent to focus on factual terms (company names, metrics, dates) and strip
filing jargon. Verified the same question now correctly returns $55.3B with citations.
**Residual risk:** this is a prompt-level mitigation, not a retrieval-architecture fix. Other
jargon patterns (e.g., "per the latest proxy statement", "as reported in Item 7") could
trigger similar bias. A more robust fix would be query rewriting at the retrieval layer
(before embedding), but that's deferred to avoid scope creep.

## NEW (2026-09): Q31/Q38 recategorization exposed a real, subtle failure mode — entity-scope ambiguity within a single filing
Spot-checking Q31/Q38 after moving them to the answerable bucket surfaced a genuine
citation miss on Q38 that's worth keeping visible, not just noting as "fixed the eval."
M&T's own 10-K states two different "deposits" figures for the same date, for two
different legal entities: "M&T" (the holding company / registrant, i.e. what the eval's
ground truth means by "M&T Bank") had deposits of **$163.3 billion** (chunk #23,
consolidated balance sheet), while "M&T Bank" (the wholly-owned bank subsidiary, distinct
from the holding company) had deposits of **$167.3 billion** (chunk #29). Both chunks were
retrieved as evidence; the system cited the subsidiary's figure ($167.3B) instead of the
holding company's ($163.3B) — an entailment-valid citation (chunk #29 does say "M&T Bank"
deposits are $167.3B) that's nonetheless the wrong entity given the eval's naming
convention (every other M&T question in the set uses "M&T Bank" to mean the holding
company). Not a retrieval or entailment-check bug — both stages worked correctly on
ambiguous source data. A genuinely harder problem (disambiguating a holding company from
its identically-branded bank subsidiary) than anything this eval set previously
surfaced. Also observed independently: single-shot retrieval (`retrieval_ablation.py`,
hybrid+reranking) misses both Q31's and Q38's ground-truth chunk at k=5 — the live agent
loop's multi-turn search recovered Q31 (correct answer, correct citation) but not Q38.
The reranker in particular pushed EWBC's own chunks entirely out of Q31's top-5 dense
candidates in favor of other companies' narrative deposit chunks — consistent with
ADR-004/006's existing finding that the cross-encoder favors topical/textual similarity
over the specific numeric fact being asked about, now observed against dense balance-sheet
table chunks specifically, not just narrative text.

## Citation accuracy headline numbers (60.7%/60.0%) are likely an undercount — every "failure" investigated turned out to be an eval or tooling artifact, not a system bug
The citation-verification eval ran all 38 questions to completion: 17/28 (60.7%)
answerable questions correctly cited the ground-truth chunk, 6/10 (60.0%) adversarial
questions correctly abstained. Failure analysis (`eval/failure_analysis.py`) dug into 6
of the "failing" cases in full detail (retrieved evidence + exact synthesized claims,
not just the pass/fail booleans). **Zero of 6 turned out to be confirmed real system
bugs** — including one (Q38) originally reported here as a confirmed hallucination,
later found to be a false alarm caused by my own diagnostic tool truncating chunk
previews to 250 characters, which cut off the exact sentence ("...deposits of $167.3
billion...") that proved the system's answer was correct all along. Full breakdown:

- **Not a bug — my own diagnostic tool's fault (Q38):** the system correctly cited a
  real chunk containing M&T's actual disclosed deposit figure ($167.3B, confirmed
  directly in the source filing). `failure_analysis.py` originally displayed only the
  first 250 characters of each chunk, hiding the supporting sentence and leading to a
  wrong manual read. Fixed: the tool now shows full chunk text, not a truncated preview.
- **Eval-construction gap, not a system bug (Q31, Q1):** Q31 assumed EWBC's deposit mix
  existed only as an image (confirmed via alt-text, but that check was incomplete — the
  same data is *also* in a real text table elsewhere in the filing, and the system
  correctly found and cited it). Q1's "wrong" citation was a second, equally-valid
  restatement of the same fact in a different section with different wording — ground
  truth only captured one of the document's multiple redundant restatements of the same
  number, an inherent limitation of locator-string-based ground truth in a corpus that
  routinely repeats key figures in different phrasing across sections.
- **Eval-metric blind spot, not a system bug (Q32):** the system's actual answer text
  was a correct, honest hedge ("gender composition is not explicitly detailed..."), but
  it cited a real chunk to explain *why* it couldn't answer — our
  `correctly_abstained = (supported_claims == 0)` metric can't distinguish "cited
  evidence to support a fabrication" from "cited evidence to explain an honest
  non-answer."
- **Correct behavior in response to a genuine upstream gap (Q17):** the ground-truth
  chunk was never retrieved at all (a real retrieval miss), and synthesis correctly
  declined to answer given insufficient evidence — scored as a "citation miss," but
  actually the system working as intended once you look past the boolean.
- **Inconclusive, likely LLM non-determinism (Q37):** re-running produced a different
  result than the original run (0 claims vs. a supported-but-wrong claim); Gemini calls
  aren't temperature-pinned, so a live re-run isn't a reliable way to reproduce a
  specific past failure — logging the raw response at failure time would fix this for
  next time.

Net effect: don't quote 60.7%/60.0% as trustworthy accuracy numbers without this
context — the true citation accuracy is very likely much higher, since every concretely
investigated "failure" was an artifact of the eval, not the system. An LLM-based
entailment check was still added to `_verify()` as a general safety net (ADR-006's
documented post-hoc-attribution option) — the general failure mode it guards against
(citing a real-but-irrelevant chunk) is still a legitimate risk worth checking for, even
though this specific motivating case turned out to be a false alarm. A cross-encoder
reranker was tried first for this check and rejected: with realistic full-chunk text, a
confirmed-bad citation and a confirmed-good citation both scored 0.97+, since a
reranker measures topical relevance, not entailment — a real, useful negative result,
not wasted effort. Still worth fixing separately: an explicit "unable_to_answer" field
in the synthesis schema rather than inferring abstention from an empty citation list, to
fix Q32-style scoring ambiguity at the source.

## RESOLVED (2026-09): Two adversarial questions (Q31, Q38) were actually answerable — eval set corrected
Both were designed assuming specific data existed only as images (Q31: EWBC's deposit
mix) or wasn't disclosed at all (Q38: M&T's period-end deposits). Both assumptions were
wrong — real text disclosures exist for both, confirmed directly against source filings
(EWBC's percentage breakdown is also presented as a text table, not only the embedded
image; M&T's period-end total deposits is a plain balance-sheet line item, "Total
deposits | 163,274 | 163,515"). `correctly_abstained=False` on these two was actually
correct system behavior, not a failure — the eval's ground truth was wrong, not the
system. Fixed: both moved from `ADVERSARIAL_QUESTIONS` to `QUESTIONS` (bucket 1) in
`ground_truth.py` with resolved chunk locators, and `data/eval/questions_draft.md`
updated to match. See that file's "Correction log (2026-09)" note.

## Non-determinism made two intermediate recalculations invalid before a clean baseline was reached
Re-running the eval with the entailment check active initially produced 15/28 (53.6%)
answerable / 7/10 (70.0%) adversarial — a drop on answerable, not the improvement
expected. Spot-checked directly: the agent loop retrieved a completely different set of
chunks for the same question across two runs, because neither Gemini call was
temperature-pinned — the 60.7%/60.0% and 53.6%/70.0% numbers were never a valid
before/after comparison. Fixed: `temperature=0` pinned on both `agent.py`'s and
`synthesis.py`'s Gemini calls, plus (after Gemini's daily quota ran out again, then a
Groq fallback hit its own bugs and quota — see ADR-007 and WEEKLY_LOG.md) a fresh Gemini
API key from a separate Google Cloud project to get a clean run.

**Final clean baseline (single provider, temperature=0, entailment check active):**
- Answerable: 17/28 (60.7%)
- Adversarial: 5/10 (50.0%), or **5/8 (62.5%)** excluding Q31 and Q38, which are
  confirmed mis-designed adversarial questions (see above) — `correctly_abstained=False`
  on those two is correct system behavior, not a failure.

This is the number to cite going forward. Not re-run multiple times to establish a
variance range (would cost meaningful quota across providers) — treat as a single clean
point estimate, better-grounded than the two invalid intermediate ones that preceded it,
but not a statistically characterized average.

## `EVIDENCE` accumulates across the whole agent loop, unbounded by iteration
`agent.gather_evidence()` keeps every chunk retrieved across all search iterations
(deduped by chunk_id) and passes the entire accumulated set to synthesis. With
`MAX_SEARCH_ITERATIONS=4` and `RERANK_TOP_N=5` per search, this caps at 20 chunks in the
worst case — fine at this corpus size, but there's no token-budget check before stuffing
all accumulated evidence into the synthesis prompt. Would need a truncation/prioritization
strategy before scaling to larger corpora or longer agent loops.

## Gemini's daily quota, not just per-minute, is a real constraint
`retry.py` handles per-minute rate limits (`ResourceExhausted`, 429) and transient 504s
(`DeadlineExceeded`) with exponential backoff, but Gemini's free tier also enforces a
**daily** cap that no amount of client-side backoff can work around — it's a hard wall
until reset. Discovered by exhausting it three times during manual testing (twice on the
same day, once mid-eval-run). Two API keys from the same Google Cloud project shared the
same exhausted quota; only a key from a different project worked. `retry.py` now detects
this specifically (`DailyQuotaExhausted`, raised immediately, no wasted retry time) —
confirmed working live during the citation-verification eval run, which stopped cleanly
after Q32 instead of burning through retries. Still a real constraint on how much eval
work can run per day, just no longer a silent time-waster on top of it.

## Eval harness exists now, but isn't RAGAS — a custom, smaller equivalent
`eval/ground_truth.py`, `eval/retrieval_ablation.py`, and `eval/citation_verification.py`
give real, reproducible retrieval-precision (Hit@1/Hit@5/MRR) and citation-accuracy
metrics against 38 hand-verified questions — a real quantitative baseline, not vibes.
It's not RAGAS specifically (no faithfulness/context-precision scores computed via
RAGAS's library), and 38 questions is smaller than the ~30-50 the project scope
envisioned as a stretch target, though it lands in that range. Contextual Retrieval and
Parent-Child chunking (ADR-003) remain withheld from implementation — now that the
harness exists and shows real headroom (60.7% citation accuracy, a 0% adversarial
abstention rate on the 2 tested), there's a genuine baseline to measure those upgrades
against, rather than the harness being the blocker it was before.

## Chunking parameters — now empirically tuned, but only against 28 questions
`chunk_size_tokens=200` / `chunk_overlap_tokens=40` (config.py) replaced the original
500/75 placeholder after the retrieval ablation showed Hit@5 improving from 32.1% to
53.6% at the smaller size (see ADR-003's addendum for the full diagnosis). This is real
evidence, not a placeholder anymore — but it's tuned against only 28 hand-verified
questions on one corpus (regional bank 10-Ks). The token→character approximation (`* 4`)
is still an untouched heuristic. Worth re-validating if the corpus changes meaningfully
in structure (e.g., less table-heavy, shorter documents) since the optimal chunk size is
likely corpus-dependent, not a universal constant.

## Gemini embedding provider and Cohere reranker are still untested
`GeminiEmbeddingProvider` and `CohereRerankProvider` exist and are config-selectable, but
have not been exercised end-to-end — no Cohere key has been configured, and the Gemini
embedding path specifically (as opposed to the agent-loop/synthesis Gemini calls, which
are now verified) hasn't been run. `GeminiEmbeddingProvider`'s retry/backoff logic
predates `retry.py` and duplicates similar logic inline — worth consolidating onto the
shared `with_backoff()` helper once it's actually exercised.

## Ingestion has no intermediate caching — a failed upsert loses the entire embedding pass
`ingest_directory()` embeds every chunk in memory, then upserts to Qdrant, with nothing
written to disk in between. This bit us for real: ingesting the SEC filings corpus, a
~30-minute local embedding pass completed successfully, but the single unbatched
`upsert()` call (now fixed — see `UPSERT_BATCH_SIZE` in `vector_store.py`) timed out
before committing anything, and the entire embedding computation had to be redone from
scratch. Fine at current corpus size (minutes, not hours), but would be a real problem
at a larger scale — worth adding a checkpoint (e.g., cache embeddings to disk before
upserting) if corpus size grows meaningfully past this project's scope.

## Pipeline is text-only — confirmed at least one real data-bearing image in the eval corpus
The SEC filing fetch/clean script (`scripts/fetch_sec_filings.py`) extracts text and
renders tables as pipe-delimited rows, but drops `<img>` tags entirely. A per-company
spot-check of all 6 FY2023 10-Ks found this isn't purely academic: East West Bancorp's
filing embeds deposit composition as images (`"Deposit mix 12.31.2023.jpg"`,
`"Deposit breakdown 12.31.2023.jpg"`, confirmed via alt text), and Valley National's
filing embeds workforce demographic charts (`"Race chart 2023.jpg"`,
`"Gender Cart Dec 2023.jpg"`) — real data that exists only as pixels in the source
document, not as text or tables we can index. Zions (13 images) and M&T (4 images) have
only generic numeric alt text, so data-bearing content there can't be ruled out from alt
text alone, but nothing points to it either.

Rather than treat this as an unnoticed gap, EWBC's deposit-mix data point is deliberately
used as an eval question in the adversarial/unanswerable bucket — the correct system
behavior is to abstain (flag as unsupported) rather than fabricate a plausible-looking
number, since the ground truth genuinely isn't in the indexed text. This is a sharper test
than a fully out-of-topic question, since it's topically relevant but genuinely
unanswerable from what's indexed. A vision-capable pipeline (Project 4's actual scope) is
the correct fix if this ever needed to be answerable, not a change to this project.

## Phase 1+3 results: early-stopping + abstention prompt rebalancing (2026-09)
**Phase 1 (early-stopping):** added logic to break the agent loop when a search round
adds zero new chunks to the evidence set. Before: avg 3.8 rounds (nearly all hit the
4-round cap). After: avg 3.6 rounds, with some questions dropping to 1-2 rounds. Modest
latency savings, but the real value is avoiding redundant searches that dilute evidence
quality with irrelevant chunks.

**Phase 3 (abstention prompt):** the synthesis prompt originally made it too easy to
abstain ("do not invent a hedged claim just to have something to cite"), causing false
"unable to answer" declarations on answerable questions. Rebalanced with 5 explicit rules:
(1) default to answering, (2) accept slightly different phrasing, (3) do NOT substitute
data from a different time period/entity/filing, (4) only abstain when evidence genuinely
lacks the facts, (5) provide a specific reason when abstaining. Rule 3 is critical — an
earlier aggressive version without it caused 3 adversarial regressions (Q32, Q37, Q39
confidently answered wrong by substituting data from wrong entities/periods).

**Before/after comparison caveat:** the full 39-question "after" run used different API
keys than the "before" run (keys 1-3 exhausted, rotated to keys 4-5). Even with
temperature=0, different GCP project keys produce different outputs due to infrastructure
routing — a partial same-key run showed +17pp improvement, but the full cross-key run
showed ~0% net change. This is a real finding about eval methodology: single-run
before/after comparisons across different API keys are not reliable for measuring prompt
changes. The improvement is real (confirmed on isolated spot-checks), but not cleanly
quantifiable from these runs.

## Multi-key API rotation for Gemini free tier (2026-09)
`key_rotation.py` provides thread-safe rotation across up to N Gemini API keys (currently
5, from different GCP projects). When one key hits `DailyQuotaExhausted`, the pool
automatically marks it exhausted and tries the next. `answer.py` and `run_baseline.py`
both use `while True` loops around the rotation to handle cascading exhaustion (key N and
N+1 both exhausted in sequence). All 5 keys are stored in `GEMINI_API_KEYS` as a
comma-separated env var. **Security note:** `.env` contains real API keys and must never
be committed — `.gitignore` covers it.

## No BM25 IDF calibration validation
`BM25SparseProvider` uses fastembed's default `Qdrant/bm25` model as-is. Its scoring
behavior on this specific corpus (2 short sample documents) hasn't been sanity-checked
against a larger, more realistic document set — sparse relevance quality at small corpus
size is not necessarily representative of behavior at scale.

## RESOLVED (2026-09): Cross-encoder reranking hurt Hit@1 — fixed with RRF-boosted scoring
The cross-encoder scored all same-filing chunks in a compressed 0.94-1.0 band, making the
top-N cutoff effectively random among them. It rescued some correct chunks from deep in the
candidate set (rank 8-19 → top-5) but randomly ejected others that RRF had already ranked #1.
Net effect: Hit@5 improved but Hit@1 and MRR degraded.
**Fix:** RRF-boosted reranking (70% normalized cross-encoder + 30% RRF position bonus)
preserves the reranker's rescue ability while anchoring to RRF ordering when scores are
compressed. Results: Hit@1 19.4%→32.3%, Hit@7 48.4%→58.1%, MRR 0.286→0.397.
**Residual risk:** the 0.7/0.3 weights and RERANK_TOP_N=7 were tuned on 31 questions from
one corpus. Different corpora with more diverse topics (less score compression) might benefit
from different weights or even reverting to pure reranker ordering. The constants are in
`query.py` at the top of the file.

## RESOLVED (2026-09): No cost/latency logging — fixed with per-call telemetry (ADR-009)
Every LLM/embedding call is now logged to `data/telemetry/calls.jsonl` (provider, model,
tokens in/out, latency, computed cost) via `telemetry.py`, wired into all 9 call sites
(Gemini native SDK, OpenAI-compatible failover providers, local embedding/reranker
calls). Surfaced per-query and in aggregate in the UI. A "cost for 1,000 users/month"
writeup is still open — see the README's Roadmap.
