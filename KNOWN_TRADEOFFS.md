# Known Tradeoffs

Honest, current record of what's deferred and why — not a historical log. Full build
history and debugging trail lives in `WEEKLY_LOG.md`.

## Retrieval

- **Reranker miscalibration (fixed).** The cross-encoder scored same-filing chunks in a
  compressed 0.94-1.0 band, making the top-N cutoff effectively random and hurting Hit@1/
  MRR despite improving Hit@5. Fixed with RRF-boosted scoring (70% normalized cross-
  encoder + 30% RRF position bonus): Hit@1 19.4%→32.3%, Hit@7 48.4%→58.1%, MRR
  0.286→0.398 (ADR-004). The 0.7/0.3 weights were tuned on 31 questions from one corpus —
  may need retuning on a more topically diverse corpus.
- **Filing-jargon queries (fixed).** Search queries containing filing-metadata terms
  ("10-K", "FY2023") biased retrieval toward document scaffolding instead of financial
  content, causing false "unable to answer" results. Fixed via agent system-prompt
  guidance to strip filing jargon from search queries. Residual risk: this is a prompt-
  level mitigation, not a retrieval-architecture fix — other jargon patterns could
  trigger the same bias.
- **Entity-scope ambiguity (open).** A filing can report different "deposits" figures for
  a holding company vs. its identically-branded bank subsidiary. The system once cited an
  entailment-valid but entity-mismatched figure — retrieval and entailment both worked
  correctly against genuinely ambiguous source data; disambiguating near-identical entity
  names is a harder problem this pipeline doesn't yet solve.
- **Chunk size tuned on a small sample.** `chunk_size_tokens=200` / `chunk_overlap=40`
  (empirically replaced a 500/75 placeholder, +21.5pp Hit@5 — ADR-003) is tuned against
  28 hand-verified questions on one corpus type (table-heavy bank 10-Ks). Likely corpus-
  dependent, not a universal constant.
- **No BM25 IDF validation at scale.** `BM25SparseProvider` uses fastembed's default
  model as-is; IDF statistics are computed over the current 18-filing corpus, and sparse
  relevance quality hasn't been sanity-checked against a meaningfully larger corpus
  (100s+ documents).

## Citation accuracy & eval methodology

- **Current clean baseline (2026-09, RRF-boosted retrieval, single provider, no
  failover): 19/31 (61.3%) answerable correctly cited, 7/8 (87.5%) adversarial correctly
  abstained** — raw results in `data/eval/citation_verification_rrf_boosted.json`.
  Adversarial jumped from 62.5% on the previous retrieval pipeline; answerable is
  roughly flat (60.7% → 61.3%). This baseline has **not** yet been through the same
  case-by-case failure audit the previous one got (below) — treat it as directionally
  reliable, not fully verified.
- **The previous baseline's headline numbers likely undercounted real accuracy**, and
  the same is probably still true here. Failure analysis on 6 "failing" cases from that
  run found zero confirmed system bugs — every one was an eval-construction artifact
  (ground truth missing a valid alternate phrasing of the same fact; an abstention
  metric that can't distinguish "cited evidence to fabricate" from "cited evidence to
  explain an honest non-answer"; a truncated preview in the diagnostic tool itself
  hiding the supporting text) or genuinely ambiguous source data. An LLM-based
  entailment check was added to `_verify()` as a general safety net regardless — a
  cross-encoder reranker was tried first for this and rejected (it measures topical
  relevance, not entailment; a confirmed-bad and a confirmed-good citation both scored
  0.97+).
- **Temperature=0 reduces but doesn't eliminate run-to-run variance.** Spot-checking Q40
  (highest-CET1-ratio-across-6-banks, adversarial) after the full run: the original run
  didn't abstain (scored incorrect), but re-running the identical question moments later
  correctly declared `unable_to_answer` (evidence had 5 of 6 banks' ratios, missing
  Valley National — correctly hedging rather than guessing). Same temperature=0 setting,
  different outcome. Consistent with the earlier finding that different Gemini
  infrastructure routing can produce different outputs even at temperature=0 — pinning
  temperature narrows variance, it doesn't guarantee determinism. Not re-run multiple
  times to establish a full variance range (would cost real quota) — treat every
  point-estimate baseline in this doc as a single clean run, not a statistically
  characterized average.
- **Not RAGAS.** The eval harness (`eval/ground_truth.py`, `retrieval_ablation.py`,
  `citation_verification.py`) gives real Hit@1/Hit@5/MRR and citation-accuracy metrics
  against 38 hand-verified questions, but doesn't compute RAGAS's faithfulness/context-
  precision scores specifically, and 38 questions is on the low end of the ~30-50
  question range originally scoped.
## Infrastructure & operations

- **Gemini's daily quota (not just per-minute) is a hard constraint** that client-side
  backoff can't work around. `retry.py` detects and fails fast on `DailyQuotaExhausted`
  instead of burning retries. Multi-key rotation (`key_rotation.py`, 5 keys across
  separate GCP projects — quota is project-scoped, so keys sharing a project don't help)
  extends daily eval-run capacity but doesn't remove the underlying constraint. `.env`
  holds real keys and is gitignored.
- **Evidence accumulation is unbounded by token budget.** `agent.gather_evidence()` caps
  at 20 chunks in the worst case (`MAX_SEARCH_ITERATIONS=4` × `RERANK_TOP_N=5`) — fine at
  this corpus size, but there's no truncation/prioritization strategy before scaling to
  larger corpora or longer agent loops.
- **No ingestion checkpointing.** A failed upsert loses the entire embedding pass, since
  nothing is cached to disk mid-ingestion — hit this for real once (an unbatched upsert
  timed out after a full embedding pass completed; fixed with `UPSERT_BATCH_SIZE`, but no
  disk checkpoint exists for future failures at larger scale).
- **Gemini embedding provider and Cohere reranker are wired but untested end-to-end** —
  no Cohere key configured, Gemini embedding path never exercised.

## Data coverage

- **Pipeline is text-only.** At least one filing (East West Bancorp) embeds real, data-
  bearing content as images (a deposit-mix breakdown), confirmed via alt text — not
  indexed. Used deliberately as an adversarial eval question (correct behavior: abstain
  rather than fabricate). A vision-capable pipeline would be the correct fix — out of
  scope for this project.

## Resolved

- **Explicit `unable_to_answer` field** — the synthesis schema now returns
  `{"unable_to_answer": bool, "claims": [...]}` instead of inferring abstention from an
  empty citation list (ADR-006 addendum), so an honest hedge and a fabrication that
  happened to get caught are no longer scored identically. Tightened
  `citation_verification.py`'s `correctly_abstained` metric accordingly — may lower the
  *measured* adversarial number without any change in actual system behavior.
- **Cost/latency logging** — every LLM/embedding call is now logged to
  `data/telemetry/calls.jsonl` (provider, model, tokens in/out, latency, computed cost)
  via `telemetry.py`, wired into all 9 call sites, surfaced per-query and in aggregate in
  the UI (ADR-009).
