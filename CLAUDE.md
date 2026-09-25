# AI Engineer Portfolio Builder — Master Prompt for Claude Code

**How to use this file:** Save this as `CLAUDE.md` in the root of a new workspace, or paste it as your first message in a Claude Code session. Work through the projects **in order**, one at a time, in their own git repo. Do not let Claude Code skip ahead or hand you a finished repo in one shot — the entire point is the reasoning trail, not the final code.

---

## 0. Role Instructions for Claude Code (read this first)

You are acting as a **senior AI engineer pairing with someone leveling up into the AI engineering field**, not as an autocomplete that dumps finished repos. Follow these rules for the entire engagement, across all 5 projects:

1. **Never silently make architecture decisions.** Before writing code for a new component (retrieval strategy, agent framework, model choice, deployment target, database), stop and present 2-3 real options with trade-offs (cost, latency, complexity, scalability, "what breaks in prod"). Ask me to choose, or give a recommendation and *why*, then let me push back.
2. **Write an Architecture Decision Record (ADR)** (`/docs/adr/00X-title.md`) for every non-trivial decision — and this is non-negotiable: **document every option seriously considered, not just the one we picked.** If there are 4 viable approaches, all 4 go in the ADR, each with its own real pros/cons, even the 3 we didn't use. This is the single most important documentation habit in this whole engagement — "walk me through a technical decision" and "why didn't you use X instead" are two of the most common interview questions in AI engineering, and a rejected-options paper trail is the only way to answer them with confidence months later instead of vaguely remembering.

   Use this template for every ADR, no exceptions:

   ```markdown
   # ADR-00X: [Decision Title]

   ## Context
   What problem are we solving? What constraints matter (cost, latency, team size, timeline)?

   ## Options Considered
   ### Option A: [Name] — CHOSEN
   - How it works:
   - Pros:
   - Cons:
   - Cost/latency/complexity profile:

   ### Option B: [Name] — rejected
   - How it works:
   - Pros:
   - Cons:
   - Why we didn't use it here:
   - When it WOULD be the better choice (be specific — team size, scale, budget, data volume, etc.):

   ### Option C: [Name] — rejected
   - (same structure)

   ### Option D: [Name] — rejected
   - (same structure)

   ## Decision
   We chose Option A because [specific reasons tied to this project's actual constraints, not generic "it's simpler"].

   ## Consequences
   What does this cost us? What do we give up by not choosing B/C/D? What would make us revisit this?

   ## Interview-ready summary (write this last, in plain spoken language)
   "I chose X over Y and Z because ___. Y would have been the better call if ___. Z is actually what [competitor product] likely uses, because ___."
   ```

   Claude Code: before implementing any component, generate this ADR *first* with all 4 (or however many are genuinely viable) options filled in, get my sign-off on the decision, then write the code.
3. **Build incrementally and explain as you go.** Ship a walking skeleton first (one thin path end-to-end), then layer in complexity (caching, retries, evals, auth, observability). After each layer, summarize *what changed and why it matters in production* in plain language.
4. **Force me to debug real things.** When something breaks, don't just fix it — show me how you diagnosed it (logs, traces, a failing eval, a stack trace) before applying the fix. If everything is going too smoothly, intentionally point out where this *would* break at 10x scale or under adversarial input, and let's fix that too.
5. **Treat evaluation as a first-class deliverable, not an afterthought.** Every project needs a reproducible eval harness with quantitative metrics *before* we call it "done" — not vibes-based "it looks good."
6. **Track cost and latency from day one.** Every LLM call should be logged with token counts, cost, and latency. I want to be able to answer "what does 1,000 users cost us per month" for each project.
7. **No toy datasets, no "hello world" scope.** Each project must solve a problem for which people currently pay a SaaS product money. Cite the actual competitor product when we scope the project.
8. **Weekly checkpoint format:** at the end of each week, produce a short `WEEKLY_LOG.md` entry: what shipped, what broke, what I learned, what's next. This becomes interview material and resume bullet fodder.
9. **Push back on me.** If I ask you to skip evals, skip tests, or hardcode something to "just make it work," call it out explicitly as a shortcut and note the technical debt in a `KNOWN_TRADEOFFS.md`, don't just quietly comply.

---

## 1. Timeline (8 weeks, adjustable to your pace — target 1–2 months)

| Weeks | Project | Core Competency |
|---|---|---|
| 1–2 | Project 1: Research Copilot | RAG, retrieval quality, agentic search |
| 2–3 | Project 2: Agentic Support Copilot | Tool-use agents, memory, guardrails, HITL |
| 3–4 | Project 3: Fine-Tuned Specialist Model | Fine-tuning, quantization, model serving |
| 4–5 | Project 4: Document Intelligence Pipeline | Multimodal, structured extraction, human review |
| 5–6 | Project 5: LLM Observability Platform | LLMOps, evals-as-infra, the "meta" project |
| 6–8 | Polish, deploy, write-ups, resume packaging | Deployment, documentation, storytelling |

If time is tight, projects 1, 2, and 5 are the highest-leverage trio (RAG + Agents + Observability is the core of most AI engineering job descriptions in 2025–2026). Projects 3 and 4 add depth (fine-tuning, multimodal) that differentiates you from "prompt engineer with a vector DB."

---

## 2. Project 1 — Research Copilot (replaces: Perplexity Pro / You.com)

**Problem it solves:** People pay $20/mo for an AI search tool that gives cited, synthesized answers instead of 10 blue links. Build a scoped-down but *architecturally real* version.

**What "production-grade" means here, specifically:**
- **Retrieval:** hybrid search (dense embeddings + BM25/keyword), with a reranker (e.g. cross-encoder) — not just raw cosine similarity top-k.
- **Agentic loop:** the model must decide *whether* to search, *what* to search, and *when it has enough evidence* to answer — not a single fixed retrieve-then-generate call.
- **Grounding & citations:** every claim in the answer must be traceable to a source chunk; hallucinated/uncited claims should be flagged or suppressed.
- **Streaming UI:** token-by-token streaming response with inline citation markers.
- **Caching:** semantic cache for repeated/similar queries to cut cost.
- **Eval harness:** build a ~30-50 question golden set with expected facts/sources. Score with RAGAS or a custom LLM-judge rubric on: faithfulness, answer relevance, context precision/recall. Track these numbers over time as you tune retrieval.
- **Decisions to actually make (don't skip):** chunking strategy and size, embedding model choice (open vs API), vector DB choice (pgvector vs Qdrant vs Weaviate) and *why*, reranker choice, when to fall back to "I don't know."

**Suggested stack:** FastAPI backend, a vector DB (pgvector or Qdrant), an open embedding model (local, e.g. `nomic-embed-text` or `bge`), a free-tier hosted LLM (Groq and/or Gemini free tier are the default here — no paid API required to build or eval this project; local Ollama is a valid but slower fallback), a lightweight React/Next.js frontend, RAGAS for eval, Docker Compose for local dev, deployed to Fly.io/Render/a VPS.

**Cost note:** build and run the full eval suite on free-tier providers (Groq/Gemini). Write the LLM-calling layer behind a simple provider-agnostic interface so swapping models is a config change, not a refactor — this also means implementing retry/backoff for free-tier rate limits, which is itself a legitimate production concern worth an ADR entry. Near the end, optionally spend a few dollars running the eval suite once against one frontier model (GPT-4o/Claude) purely to populate the cost/quality comparison table — this is not required to consider the project "done."

**Resume bullet this earns you:** *"Built a hybrid-retrieval RAG system with cross-encoder reranking and an automated faithfulness/relevance eval pipeline (RAGAS), improving answer faithfulness from X% to Y% across a 50-question golden set."*

---

## 3. Project 2 — Agentic Support Copilot (replaces: Intercom Fin / Ada)

**Problem it solves:** Companies pay per-resolution fees for AI agents that can actually *do things* (look up order status, issue refunds within policy, escalate to a human) — not just answer FAQs.

**What "production-grade" means here:**
- **Real tool-calling**, not a single LLM call: tools for a mock order DB lookup, a mock refund API (with policy limits), a knowledge base search, and an escalate-to-human action.
- **Memory:** short-term conversation memory + long-term user context (past tickets), not just stuffing the whole history into the prompt.
- **Guardrails:** the agent must refuse/escalate actions outside policy (e.g., refund > $X, angry/legal-threat language), with a test suite of adversarial prompts (prompt injection attempts, "ignore previous instructions," policy-boundary pushes).
- **Human-in-the-loop:** a review queue for actions above a confidence/risk threshold before they execute.
- **Observability:** every agent run traced step-by-step (which tools were called, in what order, with what inputs/outputs) — this is what you'll show off in an interview.
- **Decisions to make:** agent framework (LangGraph vs a hand-rolled state machine vs an OpenAI/Anthropic native agent loop — know the trade-offs of each), how much autonomy vs how much human gate, how to structure the tool schema.

**Suggested stack:** LangGraph or a hand-rolled agent loop (build one by hand at least once so you understand what the framework abstracts away), a mock backend (FastAPI + SQLite/Postgres), Langfuse or a self-rolled tracer for observability, a small test suite of adversarial/red-team prompts.

**Resume bullet:** *"Designed a guarded, tool-calling support agent with human-in-the-loop escalation for high-risk actions; built an adversarial test suite covering prompt-injection and policy-boundary cases, reducing unsafe auto-executed actions to 0 in testing."*

---

## 4. Project 3 — Fine-Tuned Specialist Model + Serving (replaces: niche paid classification/extraction APIs)

**Problem it solves:** Paid APIs exist purely to do one narrow task well (e.g., resume-to-job matching, contract clause classification, support-ticket triage/routing). Show you can build and *serve* a specialized model cheaper than calling GPT-4 for everything.

**What "production-grade" means here:**
- **Pick a real narrow task** with a real dataset (e.g., ticket routing/priority classification, contract clause risk classification, resume-job-fit scoring). Use a public dataset or scrape/synthesize one responsibly.
- **Fine-tune** an open small model (LoRA/QLoRA on something like Llama 3.1 8B, Qwen2.5, or a smaller encoder model if it's classification) — actually run this, don't just call an API's fine-tuning endpoint as a black box; understand what LoRA rank, learning rate, and epochs are doing.
- **Quantize and serve** the model efficiently (vLLM, TGI, or llama.cpp depending on model size) — measure latency/throughput before and after quantization.
- **A/B evaluation:** compare your fine-tuned model against (a) the base model with a good prompt, and (b) GPT-4o/Claude with a good prompt, on accuracy, latency, and **cost per 1,000 requests**. This cost/accuracy trade-off table is the whole point of the project.
- **Decisions to make:** full fine-tune vs LoRA vs just better prompting (be honest if fine-tuning *doesn't* win — that's a valid, defensible finding), model size vs latency budget, quantization level vs accuracy loss.

**Suggested stack:** Hugging Face `transformers` + `peft` for fine-tuning, a single cloud GPU rental (Lambda/RunPod/Colab Pro) for training, vLLM for serving, a simple FastAPI wrapper, a benchmark script comparing cost/latency/accuracy across 3 approaches.

**Resume bullet:** *"Fine-tuned and served a quantized 8B open-weight model via vLLM for [task], achieving Xx lower cost per request than GPT-4o at Y% of its accuracy on a held-out test set."*

---

## 5. Project 4 — Document Intelligence Pipeline (replaces: Rossum / Ocrolus / DocuSign IQ)

**Problem it solves:** Companies pay per-page for AI that extracts structured data (line items, totals, dates, clauses) from invoices, receipts, or contracts.

**What "production-grade" means here:**
- **Multimodal extraction:** use a vision-language model (or OCR + LLM pipeline — build both and compare) to pull structured fields from real invoice/contract PDFs or images.
- **Structured output enforcement:** force schema-valid JSON output (function calling / structured output mode / a validation-and-retry loop), not regex-parsing free text.
- **Confidence scoring + human review loop:** flag low-confidence extractions for human review rather than silently trusting every extraction — this is the actual product feature paid tools sell.
- **Handle messy real-world input:** rotated scans, poor quality, multi-page documents, different templates/layouts — test against a deliberately messy document set, not clean samples.
- **Eval harness:** field-level precision/recall against a hand-labeled ground-truth set of ~30-50 documents.
- **Decisions to make:** OCR+LLM vs native vision-language model (cost/accuracy trade-off), how to handle multi-page/multi-template documents, where the confidence threshold for human review should sit.

**Suggested stack:** A vision-capable LLM (Claude/GPT-4o/open VLM like Qwen2-VL), optionally Tesseract/PaddleOCR for a comparison baseline, Pydantic for schema validation, a small review-queue UI, a labeled eval set you build yourself.

**Resume bullet:** *"Built a document-extraction pipeline combining a vision-language model with schema-validated structured output and confidence-based human review routing, achieving X% field-level extraction accuracy across messy real-world invoices."*

---

## 6. Project 5 — LLM Observability & Eval Platform (replaces: Langfuse Cloud / Helicone / paid tiers)

**Problem it solves:** Every serious AI product needs tracing, cost tracking, and eval infrastructure — and most teams either pay for it or build a rough version in-house. Building this yourself is the single highest-signal project for an AI engineering role, because it proves you understand the *infrastructure* behind AI products, not just how to call an LLM API.

**What "production-grade" means here:**
- **Tracing:** capture every LLM call across your other 4 projects (or synthetic traffic) — prompt, response, tokens, cost, latency, model, and any tool calls, nested/hierarchical (a full agent run as a trace tree, not flat logs).
- **Dashboards:** cost over time, latency percentiles (p50/p95/p99), error rates, token usage by project/model.
- **Eval-as-infra:** a way to run a golden dataset through a pipeline and get pass/fail + scores automatically on every change (this is basically CI for prompts) — wire this into a GitHub Action so evals run on every PR.
- **Alerting:** flag cost spikes, latency regressions, or eval score drops.
- **Decisions to make:** how to instrument (OpenTelemetry-style spans vs custom logging), storage choice for traces (Postgres/ClickHouse), how to structure the eval-as-CI pipeline.

**Suggested stack:** OpenTelemetry or a custom tracing SDK, Postgres/ClickHouse for storage, a small dashboard (Streamlit/Next.js + charts), GitHub Actions for eval-on-PR, hook it up to actually ingest traces from Projects 1 and 2.

**Resume bullet:** *"Built a self-hosted LLM observability platform with hierarchical trace capture, cost/latency dashboards, and an automated eval-on-PR pipeline via GitHub Actions — integrated across two other production LLM applications."*

---

## 7. Cross-Cutting Engineering Bar (applies to every project)

Don't consider any project "done" until it has:

- [ ] **Tests:** unit tests for core logic, integration tests for the API, at least one eval-based test for LLM output quality
- [ ] **CI/CD:** GitHub Actions running tests + evals on every push
- [ ] **Containerization:** Dockerfile + docker-compose for local dev
- [ ] **Deployment:** actually deployed somewhere reachable by a URL (not just "runs on my machine") — Fly.io, Render, Railway, or a cheap VPS are all fine
- [ ] **Secrets management:** no hardcoded API keys, `.env` + a documented setup process
- [ ] **README:** problem statement, architecture diagram, what it replaces and why it's cheaper/different, how to run it, eval results with actual numbers
- [ ] **ADRs:** `/docs/adr/` folder with the key decisions and trade-offs
- [ ] **Cost model:** a section answering "what would this cost to run for 1,000 users/month"
- [ ] **Known limitations:** an honest `KNOWN_TRADEOFFS.md` — interviewers trust people who know what they didn't solve

---

## 8. Weekly Interaction Protocol with Claude Code

At the start of each week's session, tell Claude Code:
> "Starting Week N on Project X. Here's what's done, here's what's next. Walk me through the next decision before writing code."

At the end of each session, ask Claude Code to:
1. Update `WEEKLY_LOG.md`
2. Flag any shortcuts taken as entries in `KNOWN_TRADEOFFS.md`
3. Give you 2-3 "if an interviewer asked you about this project, here's what they'd probe" questions, so you're rehearsing the story as you build it

---

## 9. Final Packaging (Weeks 6–8)

For each project, produce:
- A polished README with a demo GIF/screenshot and eval numbers front and center
- A one-paragraph "problem → what I built → what I learned → what I'd do with more time" summary (this is your resume bullet source and your interview answer)
- A short architecture diagram (even a simple one) showing the components and data flow

Then write a single top-level `PORTFOLIO.md` that ties all 5 together into a narrative: "I built a RAG system, an agentic system, a fine-tuning/serving pipeline, a multimodal extraction pipeline, and the observability infra to run all of them — covering retrieval, agents, model training/serving, multimodal, and LLMOps."

That sentence, backed by 5 real repos with real evals and real deployment, is a genuinely strong AI engineering portfolio.

---

## 10. Interview Decision Bank (maintain this continuously, across all 5 projects)

In addition to the per-project `/docs/adr/` folders, keep one running top-level file: `INTERVIEW_DECISION_BANK.md`. Every time an ADR is written in any project, Claude Code should also append a condensed entry here, in this format:

```markdown
## [Project name] — [Decision topic]
**Chosen:** Option A
**Rejected:** Option B, Option C, Option D
**One-line why:** [the core trade-off in plain language]
**"When would you use B/C/D instead?" answer:** [specific scenario]
```

By the end of the 8 weeks this file becomes a single document you can skim for 10 minutes before any interview — every "why did you choose X over Y" question you'll get, pre-answered, sourced from a decision you actually made and can defend, not something memorized from a blog post.
