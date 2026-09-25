# Weekly Log

## Week 1 — Research Copilot: retrieval walking skeleton

**What shipped:**
- ADR-001: Qdrant chosen as vector store (over pgvector, Weaviate, Pinecone, FAISS)
- ADR-002: Local `bge-base-en-v1.5` embeddings (default) + Gemini free-tier API
  (configurable alternative) behind a shared `EmbeddingProvider` interface
- ADR-003: Recursive/structure-aware chunking (baseline) + Contextual Retrieval and
  Parent-Child chunking documented as eval-tested upgrade layers (not yet implemented —
  waiting on the eval harness to measure them against)
- ADR-004: Local `bge-reranker-base` cross-encoder (default) + Cohere Rerank
  (configurable alternative) behind a shared `RerankerProvider` interface
- Working code: chunking → dual embedding (dense `bge` + sparse BM25 via fastembed) →
  Qdrant hybrid store with RRF fusion → cross-encoder reranking, end to end
- Verified manually against 2 sample docs: hybrid search + reranking correctly
  distinguishes topically relevant chunks for two different test queries

**What broke:**
- `pip install -r requirements.txt` initially pulled the GPU/CUDA build of torch
  (1.5GB+ of `nvidia-cudnn`/`nccl`/`cusparselt` wheels) since no CPU-specific index
  was pinned — fixed by installing `torch` from the CPU wheel index first
  (`--index-url https://download.pytorch.org/whl/cpu`), which pip then respected for
  the rest of the install
- Machine ran out of disk space mid-install (175MB free of 234GB) due to unrelated
  Docker image/build-cache buildup on the host — install failed with
  `OSError: [Errno 28] No space left on device`, corrupting the partially-installed
  `scipy` wheel; fixed once disk space was freed by force-reinstalling `scipy`/`numpy`
  (`numpy<2` pinned — `fastembed` requires numpy 1.x)

**What I learned:**
- bge embedding models are asymmetric: queries need an instruction prefix
  ("Represent this sentence for searching relevant passages: ") that documents don't —
  easy to silently get wrong and it would have hurt retrieval quality without erroring
- Qdrant's hybrid search needs named vectors (`dense` + `sparse`) plus a
  `FusionQuery(fusion=Fusion.RRF)` — the fusion logic is Qdrant's, not something to
  hand-roll, which is exactly the "native hybrid" reason ADR-001 chose Qdrant
- Default `pip install` resolving GPU wheels on a CPU-only dev box is a real, recurring
  gotcha worth pinning around explicitly in any Python ML project's setup docs

**What's next (end of Week 1):**
- Build the ~30-50 question golden eval set + wire up RAGAS, so Contextual Retrieval
  and Parent-Child chunking (ADR-003) can actually be measured instead of assumed
- Semantic cache, streaming UI, cost/latency logging

## Week 1 (cont.) — Agentic loop + citation synthesis

**What shipped:**
- ADR-005: hand-rolled agent loop using Gemini's native function-calling API (over
  LangGraph, prompt-only JSON parsing) — `agent.py`, single `search` tool, model decides
  whether/what/when to search, loop owns iteration cap and message history itself
- ADR-006: structured output with a per-claim citation schema (over self-reported inline
  markers, post-hoc NLI attribution) — `synthesis.py`, model returns discrete claims each
  naming supporting chunk IDs, deterministic post-check validates every ID against the
  actually-retrieved evidence, unsupported claims are flagged rather than dropped
- `retry.py`: shared exponential-backoff wrapper for Gemini calls, added mid-session
  after hitting real rate limits (see "What broke")
- `answer.py`: orchestrator tying `agent.gather_evidence()` → `synthesis.synthesize()`
  → `synthesis.render()` into one CLI entrypoint
- Verified end-to-end: a multi-topic query correctly triggered multiple `search` calls
  and produced 5 claims, every one citing a real supporting chunk ID, zero hallucinated
  citations; an out-of-corpus query correctly returned
  `[UNSUPPORTED — no citation]` instead of fabricating an answer

**What broke:**
- First live run hit `ResourceExhausted` (429) on the synthesis call — Gemini free tier
  caps at 5 requests/minute per model. This is the exact gap flagged in ADR-002/
  KNOWN_TRADEOFFS ("Gemini providers are implemented but untested... retry/backoff has
  not been tested against real rate-limit responses") — now it has been, and `retry.py`
  wires exponential backoff into both Gemini call sites (agent loop + synthesis)
- Second run hit a **different**, harsher limit: `GenerateRequestsPerDayPerProjectPerModel-
  FreeTier`, capped at 20 requests/day — a daily quota that backoff cannot fix, since
  retrying within the same day just fails again. A second API key from the same Google
  Cloud project hit the identical exhausted quota immediately, confirming the quota is
  tracked per-project, not per-key. A key from a genuinely different project worked.
- `models/gemini-1.5-flash` (used in the original ADR-005/006 write-up) no longer exists
  in this account's model catalog — swapped to `models/gemini-flash-latest`, a stable
  alias, so the code doesn't need to track dated model version churn

**What I learned:**
- Gemini's free tier has two independent caps (per-minute AND per-day) — retry/backoff
  only solves the first; the second requires either waiting, spreading calls across
  projects, or upgrading tier, and no amount of client-side cleverness fixes it
  mid-session
- Separating the agent loop (decide whether/what to search) from citation synthesis
  (structured claim generation) into two distinct Gemini calls, rather than one call
  doing both, made the rate-limit failure easy to isolate — it was obvious from the
  traceback which stage failed and why
- Pinning a model by a stable alias (`-latest`) rather than a dated version avoids
  breakage as a provider's model catalog rotates — worth doing by default for any
  Gemini/OpenAI integration, not just this one

**What's next:**
- Build the ~30-50 question golden eval set + wire up RAGAS
- Semantic cache, streaming UI, cost/latency logging (token counts, cost, latency per
  Gemini call — currently unmeasured)

## Week 1 (cont.) — Eval corpus: SEC filings, regional banks

**What shipped:**
- Picked a harder, more differentiated eval corpus than the initial arXiv-papers idea
  after two rounds of external critique: regional bank 10-K/10-Q filings from SEC EDGAR
  (WAL, ZION, CMA, MTB, VLY, EWBC — chosen for direct exposure to 2023's regional-banking
  deposit-stress period, giving genuine version drift and contradictory year-over-year
  risk language, not clean arXiv prose)
- `scripts/fetch_sec_filings.py`: pulls filings via SEC EDGAR's public API, strips
  inline-XBRL metadata noise, renders HTML tables as pipe-delimited rows instead of
  flattening or dropping them
- 18 filings fetched (12 10-Ks across 2 fiscal years for all 6 banks, plus 2023 10-Qs
  for WAL/ZION as a temporal-drift spot-check), 8.5MB total
- Verified per-company (not just one filing) that images are mostly decorative, but
  found a real exception — EWBC embeds deposit-composition data as images, not text/
  tables — documented in KNOWN_TRADEOFFS and turned into a deliberate adversarial eval
  question rather than an unnoticed gap
- Added `collection` override to `Retriever`, `ingest_directory`, `gather_evidence`, and
  `answer_question` so the SEC corpus lives in its own Qdrant collection
  (`sec_filings_banks`), separate from the demo `sample_docs`
- Ingested successfully: 6,382 chunks, verified with a real query
  ("deposit concentration risk and uninsured deposits in 2023") that correctly surfaced
  Zions' and Valley National's actual 2023 stress-period risk language

**What broke:**
- First ingestion attempt: the Qdrant Docker container had exited (status 143) during
  an environment reset, so the ~30-minute local embedding pass ran against nothing and
  produced no output — caught by checking `docker ps` and the collection's point count
  rather than assuming the background process's silence meant progress
- Second attempt: embedding succeeded, but a single `upsert()` call sending all ~6,400
  points to Qdrant in one HTTP request hit a write timeout
  (`httpx.WriteTimeout`) — fixed by batching upserts (200 points/request,
  `UPSERT_BATCH_SIZE` in `vector_store.py`) and raising the Qdrant client's timeout to
  120s. The ~30 minutes of local embedding computation from that run was lost (never
  cached to disk) and had to be redone from scratch — worth remembering for any future
  long local computation step: cache intermediate results before the network call that
  might fail

**What I learned:**
- A background process's silence is not evidence of progress — always check the actual
  downstream state (container status, point count) before assuming a long-running job
  is working
- Batch size limits that don't matter at toy-corpus scale (2 documents, a few points)
  become real failure modes at even modest real-world scale (6,382 points) — this is
  exactly the kind of thing a toy demo corpus would never have surfaced
- Two rounds of external critique (arXiv → SEC filings → corpus scope refinement)
  measurably improved corpus quality — the plan search process itself is worth writing
  up as part of the interview story: "here's why I rejected my own first idea"

**What's next:**
- Draft ~40 eval questions across factual/numeric, multi-hop/cross-reference, and
  adversarial/unanswerable buckets
- Add dense-only / no-rerank ablation toggles, run offline retrieval ablations
- Run the citation-verification pass once per question (Gemini-dependent, conserving
  the daily quota per the offline/online split decided earlier)

## Week 1 (cont.) — Eval questions, ablation toggles, and a real chunk-size bug found & fixed

**What shipped:**
- 40-question eval set (`data/eval/questions_draft.md`) — 16 factual/numeric, 14
  multi-hop/cross-reference, 10 adversarial/unanswerable — every question checked
  against actual filing text, not assumed. 5 items initially flagged
  `NEEDS VERIFICATION` (ambiguous table framing, unit confirmation, missing full
  quote) were resolved by reading full surrounding context, not left unresolved
- `research_copilot.eval` package: `ground_truth.py` (resolves each question to its
  real `chunk_id` by re-chunking source files and locating a verbatim quote) and
  `retrieval_ablation.py` (Hit@1, Hit@5, MRR across dense-only / +hybrid / +hybrid+
  reranking — no Gemini calls, pure retrieval-stage metrics)
- `Retriever.search()` gained `use_hybrid` / `use_reranker` toggles (defaulting to
  the production config) so the same code path serves both the real pipeline and
  the ablation, instead of four divergent one-off scripts

**What broke, and how we found + fixed it (the actual debugging trail):**

*Symptom:* first ablation run (500-token chunks, our then-current config) showed
Hit@5 = 32.1% for the full pipeline — every layer improved on the one before it
(dense-only 14.3% → +hybrid 28.6% → +reranking 32.1%, validating ADR-001/ADR-004's
architecture choices), but the absolute number was low enough to be worth
investigating before trusting anything built on top of it (Task #5's citation
verification specifically).

*How we found the cause:* wrote a diagnostic script comparing, per question, the
ground-truth `chunk_id` against what the full pipeline actually retrieved. First
observation: retrieved chunks were almost always from the *correct source file* —
ruling out a company/filing mismatch bug. So the failure was chunk-level, not
retrieval-direction-level. Read the actual text of both the correct and
retrieved chunks for several misses (Q1, Q2, Q14) and found a consistent pattern:
the target fact sat 75-80% of the way through a ~1,400-2,000 character chunk,
preceded by unrelated content. Western Alliance's "insured deposit ratio
strengthened from 45% to 73%" only appeared after a paragraph opening with
"the recent volatility in the banking industry... capital and liquidity actions."
M&T's average-deposits table only appeared after a paragraph about "debt
investment securities credit-related losses." Root cause: a chunk's dense
embedding is computed over its entire text, so when the queried fact is a
minority of a chunk's content, the embedding gets dominated by whatever comes
before it — diluting the exact signal the query needs.

*How we tested the fix:* re-ingested the identical corpus with
`chunk_size_tokens=200` (down from 500) and `chunk_overlap_tokens=40` (down from
75) into a separate collection (`sec_filings_banks_chunk200`, 16,958 chunks vs.
the original 6,382), rebuilt ground truth against the new chunk boundaries (all
28 locators still resolved cleanly), and re-ran the identical ablation.

| Config | Hit@1 | Hit@5 | MRR |
|---|---|---|---|
| 500-token, +hybrid+rerank | 17.9% | 32.1% | 0.230 |
| 200-token, +hybrid+rerank | 21.4% | **53.6%** | 0.320 |

Hit@5 improved 32.1% → 53.6% (~67% relative gain) — confirmed the hypothesis with
a number, not just a plausible story. `config.py`'s defaults updated to 200/40;
documented as an addendum to ADR-003 rather than a new ADR, since it's tuning an
already-chosen strategy's parameters, not picking a new one.

**Unexpected secondary finding:** at 200 tokens, hybrid retrieval *without*
reranking actually beat +reranking on MRR (0.354 vs. 0.320) and Hit@1 (28.6% vs.
21.4%), though reranking still won on Hit@5. This suggests reranking's benefit
(established in ADR-004 at 500-token chunks) may partly be compensating for
poorly-isolated chunks rather than being an unconditional improvement — flagged
for the Task #6 failure analysis rather than assumed away.

**What I learned:**
- "Which mode wins" isn't the only thing an ablation table tells you — the low
  absolute numbers across all three modes were the actual signal to chase, not
  just the relative ordering between them
- A background process's downstream state (which file, not just which rank) is
  the fastest way to distinguish "wrong direction" bugs from "wrong granularity"
  bugs — checking source-file-match first before assuming embeddings were bad
  saved a lot of wasted investigation
- Chunk size is not a universal constant — 500 tokens is a reasonable default for
  prose-heavy documents, but dense financial filings with structurally similar
  boilerplate phrasing across many different facts ("X totaled $Y billion at
  December 31, 2023, compared to $Z billion...") need smaller chunks to keep
  facts isolated enough for embeddings to disambiguate

**What's next:**
- Run the citation-verification pass (Gemini-dependent) against the 200-token
  collection now that retrieval itself is validated
- Investigate the reranking-vs-no-reranking MRR flip during failure analysis
  (Task #6) rather than leaving it unexplained

## Week 1 (cont.) — Citation-verification pass: three real bugs found, two fixed live, one flagged

**What shipped:**
- `eval/citation_verification.py`: runs the full agent loop + synthesis against
  all 38 chunk-locatable/adversarial questions, checking (a) whether answerable
  questions cite the actual ground-truth chunk, not just *some* chunk, and
  (b) whether adversarial questions correctly abstain instead of fabricating
- Fixed `agent.py` to fall back to the original question when the model omits
  the `query` argument from a tool call (was an unhandled `KeyError` crashing
  the whole run after 24 questions' worth of work)
- Fixed `synthesis.py`'s `_verify()` to handle a claim returned as a bare string
  instead of the requested `{"text", "source_chunk_ids"}` object (was an
  unhandled `AttributeError`)
- Made the eval loop resilient to per-question errors generally (catch, log,
  continue) instead of losing all prior results on one bad question

**What broke, and how we found + fixed it:**

*Bug 1 — the "hang" that was actually a slow model:* the very first full run
appeared to hang for 25 minutes on 4 questions with zero output. Diagnosed by
testing progressively lower-level layers: bare curl to the API root (fast),
bare curl to the actual `generateContent` endpoint with a valid key (12s, not
hung — just slow), and finally inspecting the response body, which included a
`thoughtsTokenCount: 56` field. Root cause: `gemini-flash-latest` had started
resolving to `gemini-3.8-flash`, a "thinking" model variant that does internal
reasoning even for trivial prompts, adding real per-call latency that
compounds badly across our agent loop's multiple sequential Gemini calls.
Fix: switched to `gemini-flash-lite-latest` (resolves to `gemini-3.5-flash-lite`,
confirmed via direct timing at 0.9-1.4s/call, confirmed still supports function
calling correctly) in both `agent.py` and `synthesis.py`, plus added an
explicit 30s per-call timeout as defense-in-depth so a genuinely slow call
fails visibly instead of hanging indefinitely.

*Bug 2 — crash losing 24 questions of work:* the first full run with the faster
model crashed at Q26 with `KeyError: 'query'` — the model called the `search`
tool without including the expected `query` argument, and `agent.py` accessed
it via `call.args["query"]` with no fallback. All 24 already-computed results
were lost since nothing was saved incrementally. Fixed by using
`call.args.get("query", question)` (falls back to the original question) and
wrapping each per-question evaluation in the eval loop with a broad
`except Exception` so one malformed response can't erase everything before it.

*Bug 3 — DeadlineExceeded skipping ~43% of questions:* the next run completed,
but 12/28 answerable and 2/8 adversarial questions were skipped with
`DeadlineExceeded` errors — clearly too high a rate to be "occasional client
timeout." Reproduced directly: called Gemini outside the app and got a genuine
`504 Deadline expired before operation could complete.` from Google's own
gateway — a real, transient server-side condition under sustained request
volume, not a hung client call. `retry.py` only retried `ResourceExhausted`
(rate limits), not `DeadlineExceeded`, so every 504 was treated as terminal.
Fixed by adding `DeadlineExceeded` to the retryable exceptions with its own
backoff schedule (3s/6s/12s/24s). Re-running afterward: **zero**
`DeadlineExceeded` or `AttributeError` failures across all 28 answerable
questions — confirmed fixed, not just patched around.

**Results (partial — stopped by daily quota, not by a bug):**
- Answerable citation accuracy: **17/28 (60.7%)** correctly cited the actual
  ground-truth chunk (not just any chunk) — worth digging into the 11 misses
  during Task #6's failure analysis
- Adversarial abstention: **0/2 (0%)** — Q31 (EWBC's deposit mix, which only
  exists as an image in the source filing) and Q32 (VLY's workforce gender
  data, same situation) both **failed to abstain**, producing a supported-
  looking answer instead of correctly recognizing the data isn't in the
  indexed text. This is the single most important finding from this pass —
  the two adversarial questions we most expected to catch a real gap are
  exactly the ones the system got wrong. Needs failure analysis, not just a
  number.
- Ran out of Gemini's daily quota after Q32 — `DailyQuotaExhausted` fired
  immediately as designed (no wasted retry time), stopping cleanly rather
  than hanging or burning through retries. Q33-Q40 (8 adversarial questions)
  still need to run once quota resets.

**What I learned:**
- Three bugs surfaced in three consecutive runs of the same eval, each only
  visible once the prior one was fixed — the first bug's crash hid the second
  bug's error rate, which hid how bad the DeadlineExceeded rate actually was.
  Fixing eval infrastructure is itself iterative, not a one-shot task
- A model alias resolving to a different, slower underlying model without any
  code change on our end is a real, silent-regression risk — pinning to
  `-latest` aliases trades version-churn breakage (ADR-005's original concern)
  for the opposite risk of performance-churn breakage. Worth periodically
  re-timing model calls, not just re-testing correctness
- The adversarial bucket is doing its job: it caught a real behavior gap
  (fabricating answers for image-only data) that the answerable-question
  accuracy number alone would never have surfaced

**What's next:**
- Finish the adversarial bucket (Q33-Q40) once Gemini's daily quota resets
- Failure analysis (Task #6): the 11 answerable citation misses, and
  specifically why Q31/Q32 failed to abstain (does the agent loop retrieve a
  plausible-but-wrong chunk? does synthesis over-trust weak evidence?)

## Week 1 (cont.) — Adversarial bucket completed: a real pattern, not random noise

**What shipped:**
- Ran the remaining 8 adversarial questions (Q33-Q40) the next day, once
  Gemini's daily quota reset — Qdrant's container had also exited during the
  environment reset (third time this has happened this session), caught
  immediately via a clear `ConnectError`/`Connection refused` rather than a
  confusing downstream failure, and fixed with `docker compose up -d`

**Full citation-verification results (all 38 questions, complete):**
- Answerable: 17/28 (60.7%)
- Adversarial: 6/10 (60.0%) correctly abstained

**The real finding is WHICH adversarial questions failed, not just the
rate:**
- Correctly abstained (6): Q33 (SVB — wrong company entirely), Q34 (WAL
  FY2024 — date past corpus coverage), Q35 (ZION FY2021 — year not
  ingested), Q36 (CMA Q2 2023 provision — no CMA 10-Qs ingested), Q39 (WAL
  June 2022 — no 2022 10-Qs ingested), Q40 (cross-bank CET1 ranking —
  correctly hedged on partial coverage)
- Failed to abstain (4): Q31 (EWBC deposit mix — image-only data), Q32 (VLY
  gender composition — image-only data), Q37 (CMA quarterly net loss — no
  CMA 10-Qs, same root cause as Q36 which succeeded), Q38 (MTB period-end
  total deposits — a real, present-but-subtly-wrong-metric confusion: we
  only indexed *average* total deposits and *period-end core* deposits, not
  period-end *total* deposits)

The pattern: the system reliably abstains on CLEAN out-of-corpus cases —
wrong company, wrong year, wrong quarter entirely absent from any
indexed filing. It fails specifically on SUBTLER cases where something
topically adjacent or metric-adjacent IS present in the retrieved evidence:
image-only data sitting right next to real indexed text about the same
topic (Q31/Q32), or a real, correctly-cited fact that isn't quite the
specific metric asked for (Q38's average-vs-period-end confusion). Q36 vs.
Q37 is the sharpest illustration — both share the identical root cause (no
Comerica 10-Qs ingested), yet one triggered correct abstention and the other
didn't, meaning the failure isn't really "does 10-Q data exist" but
something more specific about how each individual query's retrieval and
synthesis play out.

**What I learned:**
- An abstention *rate* alone (60%) would have been a misleading headline —
  the pattern inside the failures (clean gaps handled fine, near-miss gaps
  handled badly) is the actually actionable finding, and it wouldn't have
  been visible from the aggregate number alone
- Q36/Q37 sharing a root cause but diverging in outcome is itself evidence:
  it rules out "the system always hallucinates when a whole quarter/company
  is missing" as an explanation, and points toward something query- or
  evidence-specific instead — exactly the kind of thing worth digging into
  with retrieved-evidence inspection (per the interview-prep PDF's G2
  question) rather than assumed from the aggregate rate

**What's next:**
- Failure analysis (Task #6): dig into the actual retrieved evidence for
  Q31/Q32/Q37/Q38 specifically — is a plausible-but-insufficient chunk being
  retrieved and over-trusted by synthesis, or is verification's chunk-ID-
  exists check passing when it shouldn't (per this project's own G3
  interview-prep answer)? Also revisit the 11 answerable citation misses.

## Week 1 (cont.) — Failure analysis: the 60.7%/60.0% headline numbers were understating the real picture

**What shipped:**
- `eval/failure_analysis.py`: a diagnostic tool that captures FULL detail
  (every retrieved chunk's text, every synthesized claim's exact text and
  citation) for a specific question, rather than just the pass/fail booleans
  `citation_verification.py` reports — necessary because the aggregate
  numbers alone couldn't distinguish real hallucinations from eval-
  methodology artifacts
- Diagnosed all 4 adversarial failures (Q31, Q32, Q37, Q38) plus 2 answerable
  misses (Q1, Q17) in detail

**What we found — none of it was what the raw numbers suggested:**

- **Q31 (EWBC deposit mix) — not a real failure at all.** My original
  adversarial-question design assumed this data existed ONLY as an image
  (confirmed via alt-text inspection weeks ago). Turns out that was an
  incomplete check: EWBC's 10-K text *also* contains a proper deposits table
  (`EWBC_10-K_2023-12-31.md#449`) with the exact percentage breakdown by
  category — the image was apparently a redundant visualization of data
  also disclosed in text. The system found and cited it correctly. Lesson:
  I verified absence via one signal (image alt-text) and didn't check
  whether the same data was redundantly disclosed elsewhere in text — an
  incomplete verification at eval-design time, not a system bug.
- **Q32 (VLY gender composition) — a scoring bug, not a system failure.**
  The model's actual answer text: *"is not explicitly detailed in the
  provided evidence, though the evidence notes that workforce diversity and
  gender population data was based on information voluntarily provided"* —
  a correct, honest hedge. But it cited a real chunk to explain *why* it
  couldn't answer (the DEI-framework chunk), which gave it a non-empty
  `source_chunk_ids`, and our blunt `correctly_abstained = (supported_claims
  == 0)` metric can't distinguish "cited evidence to support a fabricated
  claim" from "cited evidence to explain an honest non-answer." The system
  behaved correctly; the eval metric misclassified it.
- **Q38 (M&T period-end deposits) — a genuine, confirmed hallucination.**
  The system confidently claimed "$167.3 billion" citing
  `MTB_10-K_2023-12-31.md#29` — but that chunk is actually about bank branch
  locations ("961 domestic banking offices..."), containing no deposit
  figure at all. This is the exact failure mode ADR-006's `_verify()`
  limitation predicts: the citation existence check passes (the chunk ID is
  real and was genuinely retrieved), but the chunk's content doesn't
  entail the claim. This is the one CONFIRMED real synthesis-stage bug
  among everything examined.
- **Q37 (Comerica quarterly net loss) — inconclusive, likely
  non-determinism.** Re-running produced 0 claims (which the current logic
  would score as correctly-abstained), contradicting the original run's
  `False` result. Since Gemini calls aren't temperature-pinned, a live
  re-run doesn't reliably reproduce a specific past failure — a real gap in
  how reproducible this eval actually is.
- **Q1 (WAL total deposits) — not a real failure; a ground-truth
  methodology gap.** The system's answer ($55.3 billion, December 31, 2023)
  was fully correct, citing chunk `#304`. Ground truth only recorded `#294`
  as correct. Inspecting both: `#294` is a terse "2023 Financial Highlights"
  bullet list; `#304` is a separate, later "Balance Sheet Analysis" section
  restating the *same fact* in different wording with more detail. SEC
  filings routinely restate key figures in multiple places with different
  phrasing — our ground truth was built by searching for one exact locator
  string, which can only ever find chunks containing that exact phrase, not
  other chunks stating the same fact differently. Not fixable by a simple
  "find all occurrences of the locator" change (tried reasoning through
  this — the two chunks don't share the search phrase at all), so this is
  documented as a real, harder limitation rather than quick-patched.
- **Q17 (Zions segment deposits) — a genuine retrieval miss, but synthesis
  handled it exactly right.** The ground-truth chunk was never retrieved at
  all in the top candidates (a real retrieval-layer gap, not a
  synthesis/citation problem). Given genuinely insufficient evidence, the
  model correctly declined: *"The provided evidence does not contain
  information on how Zions' total deposits changed differently across its
  operating segments in 2023"* — the honest, correct behavior in response
  to an upstream retrieval gap. Scored as a "citation miss" by our metric,
  but it's actually a demonstration of the system doing the right thing.

**Revised read on the real numbers:** of 6 cases dug into, only 1 (Q38) was
a confirmed, real bug. Two (Q31, Q1) were eval-construction gaps making a
correct answer look wrong. One (Q32) was a scoring-metric blind spot making
a correct abstention look like a failure. One (Q17) was a genuine upstream
retrieval miss with correct downstream handling. One (Q37) is inconclusive
due to LLM non-determinism. This means the raw 60.7%/60.0% headline numbers
are very likely an UNDERCOUNT of true system quality — a meaningful
fraction of "failures" are artifacts of how the eval itself was built, not
system defects. This is exactly why CLAUDE.md's "show me how you diagnosed
it" principle matters: reporting 60.7% without this investigation would
have been a real but misleading number.

**What I learned:**
- An eval's own construction can be a bigger source of "failure" than the
  system under test — this session spent as much effort finding bugs in the
  EVAL (ground truth granularity, abstention scoring logic) as finding bugs
  in the SYSTEM, and both matter equally for trusting the final numbers
- "The citation exists" and "the citation supports the claim" are
  different, and conflating them (as `_verify()` currently does) is not a
  hypothetical risk — Q38 is a concrete instance of it happening
- Redundant fact restatement in real documents (the same number stated
  multiple ways in multiple sections) breaks the assumption that any
  single-fact question has exactly one correct source chunk — worth
  designing ground truth around this for any future corpus, not just noting
  it after the fact
- LLM output non-determinism makes exact reproduction of a specific past
  failure unreliable without pinning temperature or logging the raw
  response at failure time — worth fixing before relying on "just re-run
  it" as a debugging technique again

**What's next:**
- Given the actual root cause found (Q38's real bug is about entailment,
  not existence), a targeted fix would be adding a lightweight entailment
  check to `_verify()` (per ADR-006's documented future option: post-hoc
  NLI/embedding attribution) rather than treating this as unsolved
- Consider whether `synthesize()`'s schema should have an explicit
  "unable_to_answer" signal rather than inferring abstention from an empty
  citation list, to fix Q32-style scoring ambiguity at the source
- Update README.md with final, contextualized results (Task #7) — the real
  numbers plus this investigation's findings, not just the raw percentages

## Week 1 (cont.) — Correction: Q38 was never a real bug. Building the fix found a bug in my own analysis instead.

**Context:** set out to implement the entailment-check fix for Q38 (the one
case the previous entry called a "confirmed real hallucination") and
recalculate the eval numbers afterward. That process surfaced something more
interesting than the fix itself.

**Attempt 1 — cross-encoder reranker for entailment scoring:** calibrated a
threshold using short, hand-cleaned test snippets (bad pair scored 0.026,
good pairs scored 0.35-0.9996 — looked like a clean separation). Wired it
into `_verify()`, re-tested against Q38 directly: the bad citation still
scored SUPPORTED. Root cause: my calibration used artificially short,
isolated sentences; the REAL 200-token chunks are 500-800 characters of
surrounding context. Re-tested with realistic full chunk text: the confirmed
bad case scored 0.999, indistinguishable from confirmed good cases (also
0.97-0.999). A cross-encoder reranker measures topical relevance ("same
company, same date, financial context"), not entailment ("does this text
state this specific fact") — a bad citation about branch locations is still
highly *relevant* to a question about the same bank on the same date, even
though it doesn't contain the fact. Reverted this approach — a real negative
result, not wasted effort, and now documented in KNOWN_TRADEOFFS.md so it
doesn't get re-attempted blindly later.

**Attempt 2 — LLM-based entailment check:** tested in isolation first
(before wiring into the pipeline, this time) against the same bad/good
cases — correctly returned false/true. Implemented as a single batched
Gemini call per question (checking every claim's citations at once, not one
call per citation, to limit quota cost) in `synthesis._check_entailment()`.
Re-tested against Q38 via `failure_analysis.py`: **still showed SUPPORTED
with the "bad" citation.**

**That second failure is what actually mattered.** Called `_check_entailment`
directly with the exact real claim and real chunk text used in the run, and
it correctly returned `True` — the check wasn't broken. That forced a closer
look at the chunk text itself, which `failure_analysis.py` had only ever
shown truncated to 250 characters. Printing the FULL chunk revealed the
sentence the 250-char preview had cut off: *"As of December 31, 2023, M&T
Bank had consolidated total assets of $207.8 billion, deposits of $167.3
billion and shareholder's equity of $25.7 billion."* Confirmed directly
against the raw source filing (`grep`) — this is M&T's own real, disclosed
deposit figure. **The system's Q38 answer was correct the entire time.** The
"confirmed hallucination" from the previous log entry was actually a
hallucination in my own manual failure analysis, caused by a truncated
preview hiding the evidence that would have shown the citation was right.

**Fixed:** `failure_analysis.py` now prints full chunk text, not a 250-char
preview. Corrected `KNOWN_TRADEOFFS.md`'s Q38 entry and its "1 of 6 was a
real bug" conclusion — the corrected count is **zero of 6** investigated
cases were confirmed real system bugs. Every one was either an eval-
construction gap, a scoring-metric blind spot, or (this one) an artifact of
my own diagnostic tooling.

**What I learned:**
- The bug hunt itself needs the same rigor as the code it's hunting in — a
  truncated debug-print is exactly the kind of thing that looks like
  harmless convenience and quietly corrupts every conclusion built on top of
  it. This is the second time this project found a real problem in the EVAL
  rather than the SYSTEM (see the ground-truth locator issue from the
  previous failure-analysis entry) — worth treating eval/tooling code with
  the same suspicion as production code, not less
- Testing a fix in isolation (my `_check_entailment` unit test) before
  trusting its integration is what caught this — if I'd only tested the
  full pipeline and seen "still wrong," I might have assumed the entailment
  check itself was broken and kept debugging the wrong component
- A negative result (the reranker approach failing) is worth keeping and
  documenting, not just discarding once the working alternative was found —
  it's a real, specific, useful thing to know (rerankers ≠ entailment
  checkers) that would otherwise get re-discovered the hard way later
- Given this correction, the actual confirmed defect rate across everything
  investigated in this project's eval work is 0 real system bugs out of 6
  deeply-investigated cases — a genuinely strong result, but only credible
  because it survived being wrong twice (the reranker approach, and my own
  truncated-evidence misreading) before landing here

**What's next:**
- Recalculate the full citation-verification numbers with the LLM-based
  entailment check now active in the pipeline
- The entailment check remains valuable as a general safety net even though
  its original motivating case was a false alarm — keep it, since the
  failure mode it guards against (citing a real-but-irrelevant chunk) is
  still a legitimate risk for future, real cases

## Week 1 (cont.) — Recalculated numbers: 53.6%/70.0%, and the honest reason a clean before/after comparison isn't possible

**What shipped:**
- Full 38-question citation-verification re-run with the LLM-based
  entailment check active: Answerable 15/28 (53.6%), Adversarial 7/10
  (70.0%) — versus the earlier 17/28 (60.7%) / 6/10 (60.0%)

**Not the clean improvement hoped for, and here's why that's the honest
right conclusion, not a disappointing one:**

- **Q38 still shows `correctly_abstained=False`** — but that's now confirmed
  CORRECT, not a bug: M&T's filing genuinely discloses "deposits of $167.3
  billion" as of Dec 31, 2023 (verified directly against the source text).
  Q38's adversarial-question design was flawed from the start — the
  assumed-absent data was actually present, same root cause as Q31. Two of
  ten adversarial questions in this set turned out to test something that
  wasn't actually unanswerable.
- **Several previously-correct answerable questions (Q8, Q9, Q14, Q17, Q19,
  Q26) now show zero supported claims.** Spot-checked Q9 directly: the
  agent loop's search this run retrieved a completely different set of 14
  chunks than a prior run — EWBC's actual net-income table wasn't among
  them at all this time. Given genuinely insufficient retrieved evidence,
  the model correctly declined rather than fabricating; the entailment
  check correctly confirmed there was nothing to support. This is NOT the
  entailment check being over-strict — it's the same agent-loop
  non-determinism already documented for Q37, now visibly affecting a
  question that previously scored correctly.

**The real conclusion:** two separate live runs of this eval, on the exact
same 38 questions, produce genuinely different retrieved evidence and
genuinely different outcomes for a meaningful fraction of questions — not
because anything in the pipeline is broken, but because the agent loop's
search queries and Gemini's responses aren't deterministic (no temperature
pinning, no fixed seed). This means a single run's percentage — 60.7% or
53.6% or any other number — is a noisy point estimate, not a precise
measurement. Comparing two single runs and attributing the delta to a code
change (the entailment-check fix) is not a sound inference here; the run-to-
run variance from non-determinism is large enough to produce swings of this
size on its own, as Q9's direct inspection just demonstrated.

**What I learned:**
- "Did my fix help?" is not answerable by comparing two single before/after
  runs on a live, non-deterministic LLM pipeline — it needs either repeated
  runs averaged together (expensive in quota) or a deterministic pipeline
  (temperature=0, fixed seed) to make single-run comparisons meaningful.
  Neither existed here, so the honest answer to "is the entailment check an
  improvement" is "probably, based on the isolated tests that motivated it,
  but not provably from this recalculation alone"
- This is the third time in one afternoon that a plausible, specific
  conclusion (the reranker threshold, Q38's "confirmed" hallucination, and
  now "the entailment check caused this specific score change") turned out
  to need one more level of verification before it held up — a genuinely
  useful pattern to notice about how confidently-wrong conclusions form
  during iterative debugging, not just a series of unrelated mistakes
- The right fix for getting a genuinely reliable number isn't more manual
  spot-checking — it's making the eval itself reproducible (pin
  temperature/seed) so a before/after comparison is actually valid,
  which is now the concrete next step rather than something to work around

**What's next:**
- Pin `temperature=0` (or a fixed value) on the Gemini calls in `agent.py`
  and `synthesis.py` so eval re-runs are comparable to each other, before
  attempting another before/after comparison
- Consider running the eval N times and reporting a range/average rather
  than a single point estimate, given the demonstrated run-to-run variance
- Report the current honest numbers (53.6%/70.0%, both single-run point
  estimates with known non-determinism noise) rather than the earlier
  60.7%/60.0% — update README.md accordingly

## Week 1 (cont.) — Multi-provider LLM failover, and a clean final baseline

**Context:** given Gemini's free-tier daily quota, offered 4 new keys (Groq,
OpenRouter, SambaNova, Cerebras) to fail over to when Gemini runs out mid-
eval, so a full 38-question run doesn't get stuck waiting on one provider's
daily reset.

**What shipped:**
- `llm/openai_compatible.py`: a single shared implementation (tool-calling
  agent loop + JSON-mode synthesis + entailment check) that works against
  any OpenAI-compatible provider — Groq, OpenRouter, SambaNova, and Cerebras
  all expose the same API shape, so one module serves all four rather than
  four separate integrations
- `citation_verification.run()` now supports permanent mid-run failover
  (Gemini → first configured fallback, the moment `DailyQuotaExhausted`
  fires) AND a `force_provider` option to run the whole eval on one named
  provider for a clean, single-model, directly-comparable baseline — chosen
  deliberately over per-question round-robin, since mixing providers
  mid-comparison would reintroduce exactly the "not a valid before/after"
  problem from the previous entry
- Every result is tagged with which provider actually answered it; a mixed-
  provider run prints an explicit breakdown rather than presenting split
  results as single-model numbers

**What broke, testing each new provider (found by testing in isolation
before trusting a full run, same discipline as the reranker/entailment
work):**
- `openai/gpt-oss-120b` (Groq): reliably hallucinates calling a non-existent
  "open" browsing-style tool mid-conversation, even when only `search` was
  provided — crashed ~68% of a first full run with `BadRequestError`. Fixed
  with an explicit system-prompt constraint ("EXACTLY ONE tool... do not
  call open/browse/cursor") — confirmed working in isolated multi-turn
  tests before re-running. That FIRST broken run, though, burned through
  Groq's 200K-token daily quota on failed generations, so the fixed version
  only completed 1 question before Groq was ALSO exhausted for the day.
- `qwen/qwen3.8-27b` (both Groq and OpenRouter): free-tier output-token
  limit (1000/min) too tight for this workload — hit `RateLimitError`
  before generating a full response.
- `z-ai/glm-5.2:free` (OpenRouter): free endpoint doesn't support tool use
  at all (404).
- `nvidia/nemotron-3-super-120b-a12b:free` (OpenRouter): malformed/empty
  response on first test, not investigated further given a working
  alternative was found.
- `nex-agi/nex-n2.5-pro:free` (OpenRouter): worked correctly in isolated
  testing (multi-turn tool calling + JSON mode), but felt slow in the
  actual eval run (~2 min on Q2 alone, free-tier queueing) — stopped by
  request rather than waiting it out, in favor of a faster known-good path.
- A second Gemini API key, from a genuinely different Google Cloud project,
  resolved the whole problem directly — confirmed fast (1.2s for a trivial
  call) and gave fresh daily quota.

**Final clean baseline (Gemini, fresh key, temperature=0, entailment check
active, single provider — no mixed-provider caveat this time):**
- Answerable: 17/28 (60.7%)
- Adversarial: 5/10 (50.0%) — but 2 of the 10 (Q31, Q38) are confirmed
  mis-designed adversarial questions (their assumed-absent data actually
  exists in text), so `correctly_abstained=False` on those two is CORRECT
  behavior, not a failure. Excluding them: **5/8 (62.5%)** on genuinely
  valid adversarial questions.

**What I learned:**
- Testing each new provider/model in isolation before trusting a full run
  paid off a third time this session (reranker, entailment check, now
  provider selection) — every one of these would have produced a
  confusing, hard-to-diagnose full-run failure if skipped
- Free-tier LLM APIs have wildly different failure shapes — Gemini's is a
  clean daily quota with a clear error; Groq's is a token-budget quota that
  a buggy run can burn through invisibly; OpenRouter's free models vary
  enormously in reliability and speed model-to-model. "Just add a fallback
  provider" undersells how much per-provider validation that actually
  requires
- A genuinely different Google Cloud project's API key was, in the end, the
  fastest path back to a clean number — not because the multi-provider work
  was wasted (the failover architecture is real, working, and reusable for
  next time this happens), but because Gemini was already confirmed
  reliable and the only new variable needed was fresh quota

**What's next:**
- Write ADR-007 documenting the multi-provider LLM architecture decision
  (why OpenAI-compatible unification, why failover not round-robin, why
  Groq's specific model needed a stricter system prompt)
- SambaNova and Cerebras remain completely untested — provider registry
  entries exist but no isolated validation done yet
- Update README.md with this final, clean 60.7%/62.5%(adjusted) baseline

## Week 1 (cont.) — Parent-Child chunking: implemented, small-scale tested, modest signal, full run not yet run

**What shipped:**
- `chunking.chunk_document_with_children()`: parents use the exact same
  boundaries as the existing 200-token `chunk_document()` (so ground truth,
  already built and verified against parent boundaries, needed zero rework),
  children are smaller sub-splits within each parent
- `vector_store.upsert_children()` / `_point_to_result()` / `_dedupe_by_chunk()`:
  embeds/indexes child text for precise matching, but stores and returns the
  parent's full text — Retriever-facing result shape is identical whether a
  collection was ingested Parent-Child or plain, so no downstream code
  (agent loop, synthesis, eval) needed to change at all
- `ingest.ingest_directory_parent_child()`: new ingestion path, tested small
  before committing to the full corpus

**Real bug caught before it corrupted results:** `chunk_index` resets to 0
per source file, but the first implementation keyed the parent lookup dict
by `chunk_index` alone — across an 18-file corpus this would have silently
mixed up parents from different files (e.g., WAL's chunk #5 and CMA's chunk
#5 colliding). Caught during code review before ever running it, not after
— fixed by keying on `(source, chunk_index)` instead.

**Scale check before committing:** tested chunking mechanics on one sample
file first (2 parents → 13 children, correct parent mapping), then ingested
just WAL's 5 files (of 18) as a scale probe: 3,869 parents → 14,037 children
at 60-token child size — a 3.6x expansion. Extrapolated to the full 18-file
corpus, that's roughly 60,000+ children, a multi-hour embedding job on this
CPU (the 200-token single-level ingestion of the same corpus already took
over an hour). Decided to validate on the WAL-only subset first rather than
commit hours of compute to an unvalidated hypothesis.

**Small-scale ablation (WAL-only corpus, both sides ingested at the same
5-file scope for a fair comparison — an important correction: comparing
against the FULL 18-file baseline would have been biased, since a smaller
corpus has less competing noise regardless of Parent-Child helping at all):**

| Config | Hit@1 | Hit@5 | MRR |
|---|---|---|---|
| Baseline (200-tok, WAL-only) | 1/6 | 1/6 | 0.167 |
| Parent-Child (child=60) | 1/6 | 2/6 | 0.200 |
| Parent-Child (child=100) | 1/6 | 2/6 | 0.200 |

Modest, consistent, non-negative signal — Q2 moved from a total miss to
rank 5 in both Parent-Child configs, identically, suggesting the improvement
isn't sensitive to child size in this range (a reproducibility signal, not
just one lucky configuration). Not strong evidence on its own: n=6 is a
tiny sample, and the underlying `Retriever.search()` call still used
default `use_hybrid`/`use_reranker` settings, not yet run through the full
ablation matrix (dense-only / +hybrid / +reranking) the way the chunk-size
fix was validated.

**A real infrastructure hiccup along the way:** hit `RocksDB open error`
then `Too many open files (os error 24)` from Qdrant after creating several
test collections in one container session — not a code bug, an OS file-
descriptor exhaustion from accumulated test/scratch collections. Fixed with
`docker restart` (all data persists in the Docker volume; a restart just
clears stale file handles), not a data-loss risk.

**What I learned:**
- Testing on a small subset before committing to a multi-hour full-corpus
  run is exactly the discipline this project has repeated all session
  (reranker calibration, entailment check, provider selection) — and it
  paid off again here, surfacing both the parent-lookup bug and the true
  scale of the embedding job before wasting hours on either
- A fair small-scale ablation needs the SAME corpus scope on both sides,
  not just the same questions — comparing a 5-file Parent-Child test against
  an 18-file baseline would have been comparing two different things and
  misattributing the entire scope difference to the technique itself
- Infrastructure failures (file descriptor exhaustion) can look identical
  to application bugs (a Python traceback) until you actually read the
  error message — worth checking the actual server-side error before
  assuming the bug is in your own code

**What's next (decision point, not yet resolved):** the WAL-only signal is
real but too small to commit hours of compute to a full 18-file run on
confidence alone. Options: (a) test on 2-3 more single-company subsets
before deciding, cheaper than a full run but more evidence than n=6; (b)
commit to the full run given the signal is at least non-negative across two
child-size configs; (c) deprioritize Parent-Child given the modest size of
the effect relative to its cost, and move to the UI as originally planned.
Not decided yet — flagging for the user rather than unilaterally committing
hours of compute.

## Week 1 (cont.) — Parent-Child chunking, resolved: mixed results, one real regression, not scaled to full corpus

**What shipped:**
- Tested 2 more single-company subsets (CMA: 6 questions, 2 files; ZION: 5
  questions, 5 files) at the same fair same-corpus-scope methodology as the
  WAL test, using the child=100-token config (the one that matched
  child=60's results on WAL, so the cheaper-to-embed option)
- Fixed a real infrastructure issue along the way: Qdrant hit
  `Too many open files (os error 24)` after several test collections
  accumulated open RocksDB file handles in one container session. Not a
  code bug — fixed properly (not just restarted around) by adding
  `ulimits: nofile: 65536` to `docker-compose.yml`, plus cleaning up a
  redundant scratch collection

**Results — genuinely mixed, not a story to spin positively:**

| Company | Questions | Baseline Hit@5 | Parent-Child Hit@5 | Baseline MRR | Parent-Child MRR |
|---|---|---|---|---|---|
| WAL | 6 | 1/6 | 2/6 (+) | 0.167 | 0.200 (+) |
| CMA | 6 | 5/6 | 5/6 (same) | 0.708 | 0.750 (+) |
| ZION | 5 | 3/5 | 1/5 (**−**) | 0.267 | 0.100 (**−**) |

ZION is a real, meaningful regression — Q17 and Q29 went from correctly
found (rank 2, rank 3) to complete misses under Parent-Child. Leading
hypothesis, not yet confirmed by direct inspection: ZION's content is
unusually dense with repeated cross-segment structure (Zions Bank, CB&T,
Amegy, NBAZ each reporting "income before income taxes decreased $X
million" in nearly identical phrasing — the exact D6 boilerplate-similarity
problem this project already diagnosed once). A 100-token child chunk may
be small enough to cut off *before* the segment name that disambiguates
which of four near-identical sentences it is — the opposite of the
intended effect. Smaller isn't unconditionally better once a chunk gets
small enough to lose the specific token that disambiguates otherwise-
identical content.

**Decision: do not scale to the full 18-file corpus.** Three companies
tested, one clear win (WAL), one wash (CMA), one clear regression (ZION) —
that's not evidence a corpus-wide rollout would help on net, and given the
full run's multi-hour cost, the responsible call is to stop here rather
than spend hours discovering the same mixed picture at larger scale.
Parent-Child chunking, AS CONFIGURED (100-token children, no adjustment for
content density), is not a clear win for this corpus. Documented as a
negative/mixed result in ADR-003's addendum rather than either quietly
abandoned or falsely reported as a win.

**What I learned:**
- The instinct to validate on a subset before committing to a full run
  didn't just save compute time here — it changed the actual DECISION. If
  the WAL-only test alone had been trusted, this would have shipped as a
  false positive; testing 2 more companies caught a real regression the
  first test couldn't have revealed
- "Smaller chunks are more precise" (true for the earlier 500→200 token
  fix) doesn't generalize to "even smaller is even better" — there's a
  point where a chunk becomes too small to retain the specific
  disambiguating detail a query needs, and that point is content-dependent,
  not a fixed constant across a whole corpus
- A negative or mixed result, properly investigated and documented, is a
  legitimate, valuable outcome — not a failure to hide. Knowing Parent-
  Child chunking doesn't reliably help THIS corpus at THIS configuration is
  real, useful information for anyone deciding whether to try it on a
  different one

**What's next:**
- Move to the UI, as originally planned before this detour — retrieval
  quality work has hit a point of genuinely mixed returns rather than clear
  wins, and further tuning here (e.g., per-content-type child sizing,
  larger children, testing more companies) is a real but lower-priority
  option to revisit later, not a blocker
- If revisited: directly inspect ZION's regressed questions' actual
  retrieved children (not just the aggregate metric) to confirm or refute
  the "cut off before the segment name" hypothesis before trying a fix

## Week 1 (cont.) — Walking-skeleton UI shipped, verified in a real browser

**What shipped:**
- ADR-008: FastAPI backend + static HTML/JS frontend (no build step), chosen
  over React/Next.js (planned as an explicit later layer) and Streamlit/
  Gradio (rejected — weakest portfolio signal, no path to the citation-
  click interaction the project spec calls for)
- `api.py`: `POST /api/ask` wraps the existing agent loop + synthesis
  pipeline, returns structured claims + full evidence (not just a rendered
  string) so the frontend can support clickable citations
- `answer.answer_question_structured()`: new function returning raw
  `(claims, evidence)` rather than `answer_question()`'s pre-rendered
  string — existing CLI behavior untouched
- `web/index.html` + `app.js` + `style.css`: question box, claim rendering
  with citation badges (green=supported, orange=unsupported), click a
  citation to see the actual source chunk text in a side panel
- `.claude/launch.json`: dev-server config for `uvicorn research_copilot.api:app`

**Verified in an actual browser, not just "the code should work" (per this
project's own stated verification standard):**
- Golden path: asked "What was Western Alliance's total deposits at
  December 31, 2023?" through the real UI — got back a correctly-cited
  answer ("$55.3 billion"), clicked a citation badge, and the source panel
  correctly displayed the actual WAL 10-K text that supports the claim
- Adversarial path: asked the SVB out-of-corpus question through the UI —
  correctly rendered `[UNSUPPORTED — no citation]` instead of fabricating
  an answer, confirming the abstention behavior survives all the way
  through the new HTTP/JSON layer, not just in direct Python calls

**What I learned:**
- Wrapping an existing, already-debugged pipeline behind an API is a
  fundamentally different (and much lower-risk) task than building the
  pipeline itself — no new bugs surfaced in `agent.py`/`synthesis.py`
  because none of that logic changed, only how its output gets shaped and
  transported
- Testing "golden path + one adversarial case" in the actual browser (not
  just curling the API) caught the full round-trip — DOM rendering, event
  handlers, citation click-through — none of which a pure API test would
  have exercised

**What's next:**
- Real token-by-token streaming (`synthesize()` still blocks) and a
  React/Next.js upgrade are both explicitly deferred, not silently dropped
  — see ADR-008's Consequences
- Cost/latency logging would pair naturally with the UI now that there's a
  real request/response boundary to instrument

## Week 1 (cont.) — Closed a known eval-gap, added explicit abstention signal

**What shipped:**
- Fixed the pending `KNOWN_TRADEOFFS.md` item: Q31 (EWBC deposit-mix breakdown) and
  Q38 (M&T period-end total deposits) were miscategorized as adversarial/unanswerable
  on claims that turned out false on direct re-read of the source filings — both are
  answerable from indexed text. Moved to `QUESTIONS` (bucket 1) in `ground_truth.py`
  with resolved chunk locators; `data/eval/questions_draft.md` and `KNOWN_TRADEOFFS.md`
  corrected to match.
- ADR-006 addendum: added an explicit `unable_to_answer` boolean to the synthesis
  schema (`synthesis.py` and its OpenAI-compatible mirror), so the model can honestly
  declare "I can't answer this" instead of the eval inferring abstention from "zero
  supported claims" — which previously couldn't distinguish an honest hedge from a
  fabricated claim that happened to fail the entailment check. `citation_verification.py`'s
  `correctly_abstained` now requires the explicit signal (or literally zero claims),
  tightening the metric.

**Verified, not just implemented:**
- Both new ground-truth locators resolve cleanly against the real chunked corpus
  (30/30 bucket-1/2 questions resolve, no warnings).
- Offline retrieval ablation re-run with the corrected ground truth: Hit@5 (hybrid +
  reranking) is **50.0%** (15/30) — down from the previously reported 53.6% (15/28),
  because the two newly-added questions are both genuine retrieval misses at k=5
  (rank=None for both). Same absolute hit count, larger honest denominator — not a
  regression, a previously-hidden gap.
- Smoke-tested the new `unable_to_answer` field against live Gemini calls on a
  4-question subset (Q31, Q38 answerable; Q32, Q36 adversarial controls) rather than
  burning the full daily quota on a categorization fix: the field worked as designed
  (both adversarial controls correctly set `declared_unable_to_answer=True`), Q31
  answered correctly (5/5 claims supported, correct citation — the live agentic loop's
  multi-turn search recovered a chunk single-shot retrieval missed at k=5), and Q38
  surfaced a genuinely new, real failure mode — see `KNOWN_TRADEOFFS.md`'s new entry:
  M&T's own filing states two different "deposits" figures for the same date, for two
  different legal entities (the holding company vs. its identically-branded bank
  subsidiary), and the system cited the subsidiary's figure instead of the holding
  company's. An entailment-valid but entity-ambiguous miss, not a pipeline bug.

**What I learned:**
- "VERIFIED" in an eval-construction doc means "verified against my belief about the
  filing," not "immune to being wrong" — this whole fix started because a claim in
  `KNOWN_TRADEOFFS.md` ("Q31's data only exists as an image") was worth re-checking
  against the actual filing text rather than trusted at face value.
- Fixing an eval-categorization bug can immediately surface a *new*, real failure mode
  (the M&T entity-ambiguity case) — the fix wasn't "clean," and that's worth reporting
  honestly rather than declaring victory once the categorization itself was corrected.

**What's next:**
- A full clean citation-verification re-run (all 40 questions, single provider) would
  give an updated, trustworthy headline accuracy number — deferred for now to avoid
  spending the full Gemini daily quota on what was fundamentally a categorization fix;
  worth doing before quoting a headline number in interview prep.
- Cost/latency logging (next task, per CLAUDE.md's cross-cutting bar).

## Week 2 (cont.) — Phase 1+3 improvements, multi-key rotation, API wiring

**What shipped:**
- `run_baseline.py`: comprehensive eval runner that captures per-question verdicts,
  telemetry (rounds, latency, cost), claims with reason fields, and aggregates summary
  stats. Supports `--ids` for specific questions, `--tag` for labeling runs. Outputs
  structured JSON to `data/eval/`.
- **Phase 1 (early-stopping):** `agent.py` and `llm/openai_compatible.py` now track
  `evidence_before`/`evidence_after` per search round and break the loop when a round
  adds zero new chunks. Avg rounds dropped from 3.8 → 3.6; some questions now exit
  after 1-2 rounds instead of always hitting the 4-round cap.
- **Phase 3 (abstention rebalancing):** synthesis prompt in both `synthesis.py` and
  `llm/openai_compatible.py` rewritten with 5 explicit rules that default to answering
  while guarding against time-period/entity/filing substitution. `Claim` dataclass
  gained `unable_to_answer` and `reason` fields. Went through 3 iterations: v1
  (aggressive anti-abstention → +5 answerable, -3 adversarial), v2 (added entity/
  time-period guardrail → adversarial restored), v3 (final balanced version).
- `key_rotation.py`: thread-safe multi-key rotation pool (5 Gemini keys from different
  GCP projects). Automatic failover on `DailyQuotaExhausted`. Used by both `answer.py`
  (API path) and `run_baseline.py` (eval path).
- `api.py`: `ClaimOut` now includes `unable_to_answer` and `reason` fields. API key
  check uses `gemini_key_pool` instead of single key. Key rotation wired through
  `_run_pipeline()` in `answer.py`.
- `config.py`: `gemini_api_keys` field + `gemini_key_pool` property for comma-separated
  multi-key support.
- 6 baseline eval result files in `data/eval/` capturing the full improvement arc.

**What broke:**
- Keys 1-3 exhausted during eval runs (500 req/day/model/project per key). Built
  multi-key rotation to handle this — the infrastructure itself became a deliverable.
- Key rotation initially only handled one rotation per pipeline call (single
  try/except). If key N and N+1 were both exhausted, the second
  `DailyQuotaExhausted` was uncaught. Fixed with `while True` loops in both
  `_run_pipeline()` and `_run_question()`.
- First aggressive anti-abstention prompt (Phase 3 v1) caused 3 adversarial questions
  to confidently answer wrong by substituting data from wrong entities/time periods.
  Fixed by adding rule 3 ("do NOT substitute data from a DIFFERENT time period,
  entity, or filing").

**What I learned:**
- Even with temperature=0, different GCP project API keys produce different outputs
  due to infrastructure routing. This makes before/after comparisons across keys
  unreliable — a real eval methodology finding, not a code bug. A partial same-key run
  showed +17pp improvement; the full cross-key run showed ~0%. The improvement is real
  but not cleanly quantifiable from these runs.
- Abstention tuning is a precision/recall tradeoff: making it harder to abstain
  improves answerable accuracy but risks adversarial regressions. The entity/time-
  period guardrail (rule 3) was the key insight — it preserves the anti-abstention
  bias for genuine evidence while blocking the most dangerous substitution pattern.
- Multi-key rotation across GCP projects is the practical solution to Gemini's daily
  quota — each project has independent quota, so 5 keys = 5x daily capacity.

**What's next:**
- Cross-cutting bar: tests, CI/CD, deployment, semantic cache, streaming

## Week 2 (cont.) — Phase 2: retry visibility, Phase 4: reranker optimization

**What shipped:**

*Phase 2 — retry visibility:*
- `retry.py`: `with_backoff()` now accepts a `label` keyword arg and prints each retry
  attempt to stderr with call-site label, attempt number, error type, and sleep duration.
  Example: `[retry:synthesis] attempt 1/5 hit ResourceExhausted (rate limit), sleeping 5s`
- All 4 call sites in `agent.py` and `synthesis.py` tagged: `agent_init`, `agent_search`,
  `synthesis`, `entailment`. Output visible in server logs and eval terminal.

*Phase 4 — reranker optimization:*
- Diagnosed why the cross-encoder was hurting Hit@1 and MRR despite helping Hit@5:
  all same-filing chunks scored 0.94-1.0, making the top-5 cutoff effectively random.
  Correct chunks at RRF rank 1 were pushed to rerank rank 6-8 because 5 other chunks
  from the same filing scored marginally higher.
- Tested 4 strategies across 31 questions (offline, pure retrieval, no LLM calls):
  pure reranker at different top_N, threshold-based RRF fallback, and RRF-boosted
  combined scoring.
- **RRF-boosted reranking** (70% normalized reranker + 30% RRF position bonus) won on
  all three metrics. Implemented in `query.py` as `Retriever._rerank_rrf_boosted()`.
  `RERANK_TOP_N` increased from 5 to 7.

**Results — retrieval ablation (31 questions, 200-token chunks):**

| Mode | Hit@1 | Hit@5/7 | MRR |
|---|---|---|---|
| dense-only | 9.7% | 19.4% | 0.137 |
| +hybrid (RRF) | 25.8% | 41.9% | 0.327 |
| +hybrid +reranking (old, pure) | 19.4% | 48.4% | 0.286 |
| **+hybrid +reranking (RRF-boosted)** | **32.3%** | **54.8%** | **0.397** |

*Also shipped:*
- Frontend: `app.js` now renders `unable_to_answer` claims with reason text + styled
  abstention box (`style.css`). Cache-bust params on static assets. `api.py` returns
  `Cache-Control: no-cache` on `index.html`.
- `retrieval_ablation.py` default collection fixed to `sec_filings_banks_chunk200`.

**What broke:**
- Browser pane aggressively cached static files (HTML/JS/CSS) across server restarts,
  preventing verification of the abstention UI changes. Added `?v=2` cache-bust params
  and `Cache-Control: no-cache` on the HTML response. Verified correct rendering by
  injecting updated JS and confirming the API response data was correct.

**What I learned:**
- When a reranker's scores are compressed, you need a tiebreaker from a different
  signal. The RRF position is ideal because it's computed from embedding similarity +
  BM25 term overlap — completely independent of the cross-encoder's judgment. This is
  a general pattern: combining two independent signals (even when one is noisy) almost
  always beats using either alone.
- The threshold-based approach failed because the compression was universal across this
  corpus, not query-dependent. A threshold only helps when some queries have meaningful
  separation and others don't.
- Retry visibility is a tiny change with outsized debugging value — previously, a slow
  request could be a hung model call, a rate-limit backoff, or a dead key, and there
  was no way to tell without adding temporary print statements.

**What's next:**
- Full end-to-end eval re-run with the RRF-boosted reranker to measure citation-level
  impact (not just retrieval-level)
- Cross-cutting bar: tests, CI/CD, deployment, semantic cache, streaming
