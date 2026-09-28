# ADR-011: Eval-as-CI-Infra — Retrieval Regression Gate

**STATUS: ACCEPTED.** Option A (cached ingestion, path-filtered trigger) implemented in
`.github/workflows/eval.yml` and `eval/ci_gate.py`.

## Context
CLAUDE.md's cross-cutting bar wants "GitHub Actions running tests + evals on every
push." Unit tests (23, mocked dependencies) already run on every push. The natural next
step is wiring the eval harness itself into CI, so a retrieval regression — like the
one ADR-004 found and fixed (naive reranking hurting Hit@1 despite helping Hit@5) — gets
caught automatically next time, not discovered by hand months later.

The blocking constraint, discovered by measuring rather than assuming: full corpus
ingestion (18 SEC filings, ~17,000 chunks, local `bge-base` CPU embedding) takes **~55
minutes**, confirmed by timing a real run before designing this workflow. A CI job that
adds 55 minutes to every single push is not "fast feedback," it's a tax that would get
disabled or ignored within a week. Whatever the eval-in-CI design is, it has to not pay
that cost on every trigger.

A second, related question: `citation_verification.py` (the LLM-dependent eval,
measuring citation/faithfulness accuracy, not just retrieval) is the more complete
"eval-based test for LLM output quality" the cross-cutting bar literally asks for — but
it needs real Gemini calls, and this project's Gemini usage is entirely free-tier, with
a shared daily quota also consumed by manual development/testing. This ADR only covers
the retrieval-side gate; the LLM-output-quality question is addressed separately as a
documented trade-off in KNOWN_TRADEOFFS.md (not wired into CI at all, for the same
quota-sharing reason explored in Option D below).

## Options Considered

### Option A: Cache the ingested Qdrant data, path-filter the trigger — CHOSEN
- How it works: `.github/workflows/eval.yml` triggers only on pushes/PRs touching
  retrieval-relevant files (chunking, query, vector store, embeddings, reranker, the
  eval harness itself, or the corpus). Qdrant runs via `docker run` (not a `services:`
  block, so its volume can be tied to a cache-restored path) with the storage directory
  cached via `actions/cache`, keyed by a hash of the corpus files + `config.py`'s
  chunking parameters. First run pays the ~55 minute ingestion cost once; every run
  after that (until the corpus or chunk config actually changes) restores from cache in
  seconds and only pays for the ablation itself. `eval/ci_gate.py` runs the same offline
  ablation as `retrieval_ablation.py` and fails the job if the production config's
  (RRF-boosted reranking) Hit@1 drops below 25% — a floor set comfortably below the
  current 32.3% baseline but above the pre-fix regressed configs (19.4% pure reranker,
  9.7% dense-only), so it actually catches a real regression back toward either, not
  just normal run-to-run noise.
- Pros: real regression signal against the actual production corpus and config, not a
  toy stand-in; genuinely fast after the first run (cache hit skips ingestion entirely);
  only triggers when the files that could actually cause a retrieval regression change,
  not on every README edit.
- Cons: the first run on any given corpus/config combination is still slow (~55min,
  confirmed: the real first GitHub Actions run took 1h10m42s including setup overhead);
  cache invalidation is coarse — any corpus or config change forces a full re-ingestion,
  even a change that couldn't plausibly affect retrieval quality (e.g., fixing a typo in
  one filing); adds real complexity (two cache steps, a custom hash key, a `docker run`
  step instead of a simple `services:` block) compared to a naive always-ingest design.
- Cost/latency/complexity profile: ~2 hours to design and validate (including the local
  timing measurement that drove the design); $0 (GitHub Actions minutes are free for
  public repos); cached runs complete in a few minutes, uncached runs in over an hour.

### Option B: Small smoke-test corpus (data/sample_docs) on every push — rejected
- How it works: ingest the 12KB `data/sample_docs` corpus (seconds, not minutes) on
  every push, run a lightweight ablation against it as a fast sanity check.
- Pros: genuinely fast on every single push, no caching complexity needed at all;
  catches total pipeline breakage (a crash, an import error, a broken Qdrant query)
  immediately.
- Cons: `data/sample_docs` isn't the real eval corpus and has no corresponding
  hand-verified ground-truth question set at this project's scale — a check against it
  wouldn't have caught the actual regression ADR-004 found (that needed real financial
  disclosures with genuine cross-company/cross-filing ambiguity, not two short reference
  documents). Weak signal dressed up as a regression gate is arguably worse than an
  honest "we don't have automated regression detection" — it would pass while a real
  regression on the production corpus went undetected.
- When it WOULD be the better choice: as a *fast pre-check* layered before Option A's
  expensive-first-time gate, to catch total breakage before spending CI minutes on the
  real corpus — not attempted here, a reasonable future addition if the full eval gate's
  first-run cost ever becomes a real friction point.

### Option C: Scheduled (nightly) or manual-only, not gating pushes — rejected
- How it works: run the full corpus ablation on a cron schedule or via
  `workflow_dispatch`, decoupled from push/PR events entirely.
- Pros: simplest to build — no cache-key design, no path filtering, no `docker run`
  volume juggling; sidesteps the "first run is slow" problem since nothing is blocking
  a PR merge waiting on it.
- Cons: loses the actual value CLAUDE.md's cross-cutting bar is pointing at — catching a
  regression *before* it merges, in the PR itself, not discovering it the next morning
  after it's already on `master`. Given the caching approach in Option A makes
  post-first-run checks fast anyway, this gives up real signal for a simplicity gain
  that Option A's caching already captures without the trade-off.
- When it WOULD be the better choice: a corpus large/volatile enough that even cached
  runs stay expensive (e.g., re-embedding is triggered by something other than a clean
  content hash, like a rotating live document feed) — not the case here, this corpus is
  static.

### Option D: Wire citation_verification.py (LLM-dependent) into the same gate — rejected
- How it works: add `GEMINI_API_KEY` as a GitHub secret, run the citation-accuracy eval
  (or a small fixed subset of it) as part of this same workflow.
- Pros: would close the "eval-based test for LLM output quality" cross-cutting-bar item
  literally, not just its retrieval-only proxy; catches synthesis/entailment regressions
  the retrieval gate structurally can't see.
- Cons: needs a real, shared, exhaustible resource — this project's Gemini usage is
  entirely free-tier with a **daily** quota (not just per-minute), also consumed by this
  project's own manual development and browser-based testing. A CI run that goes red
  because someone was testing the UI earlier that day and burned the quota is noise, not
  a regression signal — worse than no gate, because a red CI run that's actually
  meaningless erodes trust in the gate for when it matters. This is the same
  free-tier-sharing reasoning behind ADR-010's deployment decision.
- When it WOULD be the better choice: once this project (or a later one) moves off a
  shared free-tier quota — a dedicated CI-only API key with its own quota allocation, or
  a paid tier with enough headroom that manual testing and CI don't compete for it.

## Decision
Option A. Real regression signal against the production corpus, made fast on every
run *after* the first by caching the expensive part (ingestion) and only re-paying it
when the underlying data or config could plausibly have changed. Path-filtering keeps
it from firing on changes that couldn't affect retrieval quality at all.

## Consequences
We get an actual, working regression gate that would have caught ADR-004's own
regression automatically, running on GitHub's infrastructure for $0. We accept a slow
first run per corpus/config combination (~55min-1h10m, measured not estimated) as an
unavoidable one-time cost of testing against the real corpus rather than a toy one. We
explicitly do NOT get LLM-output-quality regression detection in CI — that gap is
documented, not hidden, in KNOWN_TRADEOFFS.md. We'd revisit Option D the moment quota
stops being a shared, contested resource.

## Interview-ready summary
"I wired the offline retrieval eval into CI as a real regression gate, but the design
took two real constraints seriously instead of just wiring it up naively. First, I
measured — not guessed — that ingesting the real 18-filing corpus takes about 55
minutes on CPU, which ruled out 'ingest fresh on every push' immediately; I solved that
with content-hashed caching so only an actual corpus or config change pays that cost
again. Second, I deliberately did NOT wire the LLM-dependent citation-accuracy eval into
the same gate, because it needs Gemini calls against a shared, exhaustible free-tier
daily quota — a CI run failing because manual testing burned that day's quota isn't a
real signal, and I'd rather have an honest gap than a gate people learn to ignore."
