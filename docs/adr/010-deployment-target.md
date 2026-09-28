# ADR-010: Deployment Target

**STATUS: ACCEPTED.** Option D (deliberately not deployed) chosen; deployment effort
deferred to a later project in the portfolio.

## Context
CLAUDE.md's cross-cutting bar asks for "actually deployed somewhere reachable by a URL
(not just 'runs on my machine')." Docker + docker-compose already prove the app builds
and runs correctly as a container (verified end-to-end: a real `/api/ask` query against
the SEC filings corpus, from inside the container, returns a correct cited answer with
telemetry logged — see WEEKLY_LOG.md). The remaining question is whether to take the
next step and put that container somewhere publicly reachable.

The constraint that dominates this decision: the agent loop runs on Gemini's **free**
tier, which has two real production-visible consequences —
- **Latency:** individual queries range 15-115s depending on free-tier load (measured
  live, not estimated — see the Cost section in README.md), and multi-fact questions can
  take 4 search rounds even when 1 would suffice (see KNOWN_TRADEOFFS.md's agent
  over-search finding).
- **A hard daily quota**, not just per-minute rate limits — once exhausted, the app
  either fails over to a secondary provider (ADR-007) or stops answering until reset.

A public link that a recruiter or interviewer might click at any time, with no control
over when, means either of those failure modes could be the very first thing they see.

## Options Considered

### Option A: Fly.io — two apps (api + qdrant) on Fly's private network — rejected for now
- How it works: `flyctl launch` for the API (using the existing Dockerfile as-is) and a
  second Fly app for Qdrant with a persistent volume, connected over Fly's internal
  6PN network.
- Pros: closest match to the existing Docker setup (minimal new config); free-tier
  allowance covers a low-traffic portfolio demo; persistent volumes exist on the free
  Hobby plan, unlike Render's.
- Cons: doesn't fix the underlying latency/quota problem — a slow or quota-exhausted
  response is just as visible on a real URL as it would be locally; requires an account
  and a card on file (Fly's abuse-prevention requirement) that only the project owner
  can create, not something automatable from this environment (no `flyctl` installed or
  authenticated when this decision was made).
- When it WOULD be the better choice: the moment this project's LLM calls move off a
  free tier (e.g., after Project 3's fine-tuned-serving work makes cheap, low-latency
  inference available), or when a specific interview loop asks to see it live and a
  short-lived deploy is worth spinning up for that one occasion.

### Option B: Render — simpler dashboard, no card required — rejected
- How it works: connect the GitHub repo, deploy the Dockerfile as a Web Service.
- Pros: no payment method needed for the free tier; simplest setup of the three options.
- Cons: free-tier services sleep after ~15 minutes of inactivity — a cold start on top of
  already-slow free-tier Gemini calls compounds the exact latency problem this decision
  is trying to avoid; free-tier disks are ephemeral, so Qdrant's ingested data wouldn't
  survive a restart, meaning either re-ingesting on every cold start (expensive, ~55min
  for the full corpus per eval.ci_gate's measurement) or paying for a persistent disk
  (real recurring cost) just to get back to where Fly.io's free tier already is.
- When it WOULD be the better choice: a lower-stakes internal demo where cold starts and
  occasional data loss are acceptable, and avoiding a card-on-file matters more than
  latency or persistence.

### Option C: Cheap VPS (DigitalOcean/Hetzner) — rejected
- How it works: provision a small droplet, run `docker compose up -d` directly — nearly
  identical to local dev.
- Pros: most control; the existing docker-compose.yml works almost unchanged; no
  platform-specific quirks (no sleep-on-idle, no ephemeral disk surprises).
- Cons: real recurring cost (~$5-6/month) starting immediately, not just past a free
  allowance — the only option of the three with no free tier at all; still doesn't
  address the underlying free-tier LLM latency/quota problem; requires the same
  account-creation step only the project owner can do.
- When it WOULD be the better choice: once there's a reason to pay for infrastructure
  regardless (e.g., running Project 5's observability stack, which needs a real
  persistent store anyway), self-hosting everything on one VPS could make sense to
  amortize the cost across projects.

### Option D: Don't deploy Project 1; defer to a later project — CHOSEN
- How it works: no live deployment for this project. Docker + docker-compose remain the
  proof that the app is deployment-ready (builds, runs, serves real answers end-to-end
  in a container) without actually exposing free-tier latency/quota to a public URL.
  Deployment effort gets spent on a later portfolio project better suited to it —
  larger scope, and by then likely running on infrastructure (fine-tuned/self-served
  models from Project 3, or the observability stack from Project 5) that doesn't carry
  the same free-tier-latency risk.
- Pros: doesn't spend real money or an account-creation step on a demo that would likely
  read as broken/slow rather than impressive; keeps the honest-tradeoffs story
  consistent — this project already documents its free-tier constraints openly rather
  than hiding them, and a live demo exhibiting exactly those constraints in front of an
  unpredictable audience undercuts that credibility rather than reinforcing it.
- Cons: literally does not satisfy CLAUDE.md's cross-cutting-bar checkbox as written
  ("actually deployed somewhere reachable by a URL") — this is a real, acknowledged gap
  against the checklist, not a reinterpretation of it.
- Cost/latency/complexity profile: $0, zero setup time now; real cost deferred to
  whichever later project actually gets deployed.

## Decision
Option D. Deployment is explicitly deferred, not abandoned — CLAUDE.md's own timeline
table describes an adjustable, portfolio-wide pace, and Weeks 6-8's "Final Packaging"
phase is where all 5 projects get deployed and polished together. Spending that effort
now, on infrastructure that would mostly showcase free-tier LLM latency rather than
this project's actual retrieval/agentic/eval work, is lower leverage than spending it
later on a project (or the final packaging pass) where deployment doesn't fight the
demo's own credibility.

## Consequences
We give up the literal "reachable by a URL" checkbox for this project, and accept that
as an honest, documented gap rather than working around it. We keep $0 spent and no
cloud accounts created for Project 1 specifically. We'd revisit this immediately if: an
actual interview loop asks to see it live (worth a short-lived Fly.io deploy for that
one occasion), or a later project's infrastructure (self-served fine-tuned models,
Project 5's observability stack) removes the free-tier-latency problem enough that a
standing deployment stops being a liability.

## Interview-ready summary
"I didn't deploy this one live, and I can defend that choice specifically: it runs
entirely on free-tier Gemini, which means 15-115 second response times and a hard daily
quota that either fails over to a backup provider or stops answering until reset. A
public link a recruiter clicks at an unpredictable moment would be as likely to show a
timeout or a quota error as a real answer — that's not a demo, that's a coin flip. Docker
and docker-compose already prove the container builds and serves correct, cited answers
end-to-end; I chose to spend deployment effort later, on a project where the
infrastructure doesn't fight the demo, rather than force a checkbox here at the cost of
the exact reliability story this project is built around."
