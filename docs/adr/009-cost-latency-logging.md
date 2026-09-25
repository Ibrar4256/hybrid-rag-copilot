# ADR-009: Cost/Latency Logging for LLM and Model Calls

**STATUS: ACCEPTED.** Option A confirmed by user; implementation wired into all 9 call sites.

## Context
CLAUDE.md's cross-cutting bar requires every LLM call logged with token counts, cost, and
latency, so the project can answer "what does 1,000 users cost us per month" — not yet
implemented. There are two real call-site families today:
- **Metered (has a real $ cost or a real quota to track):** Gemini native SDK calls in
  `agent.py` (tool-calling loop, 2 call sites) and `synthesis.py` (structured-output +
  entailment check, 2 call sites); OpenAI-compatible calls in `llm/openai_compatible.py`
  (agent loop, entailment, synthesis — 3 call sites) for the Groq/OpenRouter/SambaNova/
  Cerebras failover path (ADR-007).
- **Unmetered but latency-relevant:** local `bge-base` embeddings and local
  `bge-reranker-base` cross-encoder calls (`embeddings/local_bge.py`,
  `reranker/local_cross_encoder.py`) — free (CPU/GPU-bound), but real production latency
  budget worth tracking once cost/latency dashboards exist.

Constraints: this project runs entirely on free-tier providers (Groq/Gemini) by design
(see README/CLAUDE.md's cost note) — actual $ cost today is $0, so "cost logging" mostly
means computing what it *would* cost against published per-token pricing, not reconciling
against a real bill. Project 5 (LLM Observability Platform) is explicitly the "meta" project
that builds a real hierarchical tracing + dashboard system across this project and Project 2
— this ADR should not attempt to pre-build Project 5's whole scope, only satisfy this
project's own cross-cutting bar with something Project 5 can later ingest from.

## Options Considered

### Option A: Lightweight custom logger — one wrapper function, JSONL sink — RECOMMENDED
- How it works: a single `research_copilot/telemetry.py` module exposes one function,
  `log_call(provider, model, call_type, tokens_in, tokens_out, latency_ms, cost_usd=None)`,
  called from each of the ~7 metered call sites (and the 2 local-model call sites, with
  `cost_usd=0`) right after the call returns. Each call site already has the response object
  in hand (Gemini's `response.usage_metadata`, OpenAI SDK's `response.usage`) plus a
  `time.monotonic()` delta wrapped around the existing `with_backoff()` call. Appends one
  JSON line per call to `data/telemetry/calls.jsonl`; cost computed from a small hardcoded
  per-model $/1K-token table (Gemini flash-lite, Groq/OpenRouter/SambaNova/Cerebras free-tier
  models all currently $0 — the table exists so a future paid-tier comparison run, already
  planned per CLAUDE.md's "spend a few dollars against one frontier model," has somewhere
  real to log actual cost).
- Pros: zero new dependencies; ~7 one-line call sites plus one new ~40-line module; JSONL is
  trivially greppable/loadable into pandas for the eval-cost-per-1000-users analysis
  CLAUDE.md asks for; directly reusable as Project 5's *ingestion source* later (Project 5's
  own ADR can decide dashboard/storage on top of these same JSONL records) without having
  built Project 5 early.
- Cons: no dashboard, no querying beyond "load the JSONL and compute stats yourself"; not
  hierarchical (a multi-call agent-loop run produces N flat log lines, not a trace tree) —
  acceptable now, explicitly Project 5's job to add nesting.
- Cost/latency/complexity profile: ~1 hour of work; zero added runtime cost; zero new
  dependencies.

### Option B: OpenTelemetry spans from day one — rejected for now, revisit in Project 5
- How it works: instrument every call site with `opentelemetry-api`/`sdk` spans
  (`tracer.start_as_current_span(...)`), attaching token/cost/latency as span attributes;
  export via the OTLP console/file exporter for now.
- Pros: industry-standard instrumentation format; hierarchical by construction (a span tree
  naturally represents "agent loop called synthesis called entailment check"); directly
  matches Project 5's own stated stack ("OpenTelemetry or a custom tracing SDK").
- Cons: real setup overhead (SDK init, exporter config, context propagation across the
  agent loop's iterative tool-calling structure) for a walking-skeleton-stage need that's
  currently just "log tokens/cost/latency somewhere"; adds a dependency this project doesn't
  otherwise need yet; risks half-building Project 5 inside Project 1 before Project 5's own
  ADR process gets to make this decision with full context (multiple projects feeding one
  observability platform, not just this one).
- When it WOULD be the better choice: exactly this project's own future Project 5 —
  when there are two+ real applications (Project 1 + Project 2) that need a shared,
  hierarchical, cross-service tracing format, not just one project logging its own calls.

### Option C: Third-party observability SDK (Langfuse self-hosted, Helicone) — rejected
- How it works: drop in the Langfuse or Helicone Python SDK, wrap LLM calls with their
  decorator/proxy, view cost/latency in their (self-hostable) dashboard UI.
- Pros: fastest to a working dashboard — zero dashboard code to write; battle-tested cost
  tables per model, maintained by someone else.
- Cons: this project's *own* CLAUDE.md scopes Project 5 as "replaces Langfuse Cloud /
  Helicone" — adopting one of them here would be building on top of the exact category of
  tool the portfolio's next project is meant to demonstrate you can build yourself; adds an
  external service/self-hosted container dependency for a need Option A satisfies in ~40
  lines.
- When it WOULD be the better choice: a real production team on a deadline that doesn't
  need the "I built this myself" interview story — the correct choice for almost everyone
  who isn't specifically building an AI-engineering portfolio.

### Option D: Rely on each provider's own dashboard (Gemini/Groq/OpenRouter usage pages) — rejected
- How it works: no custom code at all; pull cost/usage numbers from each provider's web
  console when needed.
- Pros: zero implementation cost.
- Cons: doesn't unify across 5 providers into one view; can't compute latency percentiles
  (most provider dashboards don't expose per-call latency at all); doesn't cover the two
  local-model call sites (no provider dashboard exists for code running on your own
  machine); doesn't satisfy CLAUDE.md's explicit "every LLM call should be logged" bar,
  which implies application-level logging, not after-the-fact console lookups.
- When it WOULD be sufficient: a quick prototype where cost/latency will never be reported
  as a project deliverable.

## Decision
Option A: a single lightweight `telemetry.py` module with one `log_call()` function, called
from all metered + local-model call sites, sinking to `data/telemetry/calls.jsonl`. Chosen
because it satisfies this project's cross-cutting bar with minimal added complexity, produces
a format Project 5 can ingest directly rather than replacing, and deliberately avoids
pre-building Project 5's own OpenTelemetry-based scope before that project's ADR process gets
to make that call with full cross-project context.

## Consequences
We get token/cost/latency visibility now, in a form that directly answers CLAUDE.md's
"cost per 1,000 users" question, without taking on OpenTelemetry's setup cost or an external
observability service. We give up hierarchical trace structure (an agent-loop run's 3-4
calls appear as separate flat log lines, not a linked trace tree) until Project 5 adds that
layer on top of these same records. We'd revisit this (adopt Option B) the moment Project 2's
agent needs the same instrumentation and duplicating flat-JSONL logging across two projects
starts to feel worse than the OTel setup cost it currently avoids.

## Interview-ready summary
"I logged every LLM call — tokens, cost, latency — to a flat JSONL file behind one shared
logging call, rather than reaching for OpenTelemetry or a third-party observability SDK like
Langfuse immediately. That's because this project's own portfolio plan has a dedicated later
project — an LLM observability platform — whose whole point is building real hierarchical
tracing and dashboards across multiple applications. Building OpenTelemetry spans here first
would have meant either redoing that work in the observability project or making Project 5's
architecture decision prematurely, with only one project's needs in view instead of two. The
lightweight logger satisfies this project's own cost-reporting requirement now and hands
Project 5 a real ingestion source instead of a toy one."
