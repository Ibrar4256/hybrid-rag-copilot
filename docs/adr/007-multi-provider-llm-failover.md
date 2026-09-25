# ADR-007: Multi-Provider LLM Failover for Agent Loop & Synthesis

## Context
Gemini's free tier enforces a hard daily request quota (ADR-005/006's chosen model), and this project has repeatedly exhausted it mid-eval-run — most recently blocking a full 38-question citation-verification baseline from completing on a single day. The user supplied API keys for four additional providers (Groq, OpenRouter, SambaNova, Cerebras) to eliminate this as a recurring blocker. All four expose OpenAI-compatible chat-completions APIs (tool calling + JSON mode), unlike Gemini's native SDK.

## Options Considered

### Option A: Permanent mid-run failover + optional forced single-provider mode — CHOSEN
- How it works: a shared `llm/openai_compatible.py` module implements the agent loop and synthesis once, generically, against any OpenAI-compatible provider (base_url + model swap only). `citation_verification.run()` starts on Gemini; the moment `DailyQuotaExhausted` fires, it permanently switches to the first configured fallback provider for the rest of that run — not per-question round-robin. A separate `force_provider` parameter allows running the entire eval on one named provider from the start, for a clean, single-model, directly-comparable baseline.
- Pros: eliminates the "stuck until tomorrow" failure mode entirely; every result is tagged with which provider actually answered it, so a mixed-provider run is never silently presented as single-model numbers; one shared implementation serves four providers instead of four bespoke integrations, since they share an API shape.
- Cons: a mid-run failover still produces a result set that isn't a clean single-model comparison (explicitly flagged in output, not hidden, but still a real limitation for before/after analysis); different providers' models have materially different behavior (see Consequences), so failover doesn't guarantee equivalent answer quality, only availability.
- Cost/latency/complexity profile: free across all five providers at this usage scale; moderate implementation complexity (one shared module, a provider registry, fail-over state tracking in the eval runner).

### Option B: Per-call round-robin across all five providers — rejected
- How it works: distribute each individual Gemini/agent-loop call across all configured providers in rotation, regardless of quota state.
- Pros: maximizes total available throughput/quota across providers; no single provider's exhaustion blocks anything.
- Cons: within a single question, different calls (search decision, synthesis, entailment check) could be answered by different models with different capabilities and biases, making individual question results incoherent and impossible to attribute to "the system's" behavior — a worse version of the "mixed-provider" caveat this project already learned to avoid the hard way (see WEEKLY_LOG.md's non-determinism entries).
- Why we didn't use it here: this project's current priority is getting a comparable, attributable baseline number, which round-robin actively undermines.
- When it WOULD be the better choice: a production system prioritizing raw throughput/availability over per-request model consistency, where downstream consumers don't need to reason about "which model answered this."

### Option C: Provider-agnostic abstraction layer (a shared interface Gemini also implements) — rejected for now
- How it works: define one `LLMProvider` protocol (mirroring `EmbeddingProvider`/`RerankerProvider`) that Gemini's native SDK and the OpenAI-compatible providers both implement, so `agent.py`/`synthesis.py` call through one interface regardless of backend.
- Pros: the architecturally "correct" long-term design — true provider-agnosticism, matching this project's existing pattern for embeddings/reranking; would let Gemini participate in round-robin or failover without special-casing.
- Cons: Gemini's native SDK (function-calling response shape, JSON-mode configuration, tool-call argument access) doesn't map 1:1 onto the OpenAI-compatible shape without a translation layer — a real refactor of `agent.py`/`synthesis.py`'s existing, working, well-tested code, not just an addition alongside it.
- Why we didn't use it here: the immediate goal (get one clean baseline run done today) didn't require touching Gemini's already-working path at all; Option A achieves failover without risking regression in code that's been debugged through three real incidents already this session (model-speed, retry logic, entailment).
- When it WOULD be the better choice: once multi-provider usage becomes a standing feature rather than an occasional quota escape hatch — worth revisiting alongside adding genuine provider selection (not just failover) to `config.py`, matching the embedding/reranker pattern.

## Decision
We implement Option A: a shared OpenAI-compatible module serving all four new providers, wired into the eval runner as permanent (not per-call) failover, with an explicit single-provider override for clean baseline runs. Gemini's existing native-SDK implementation is untouched.

## Consequences
Every new provider required real, individual validation before it could be trusted — not a hypothetical risk, but something that happened three times in one session: `openai/gpt-oss-120b` (Groq) reliably hallucinated calling a non-existent "open" tool, crashing most multi-turn runs, until a stricter system prompt fixed it; `qwen/qwen3.8-27b` had a free-tier output-token limit too tight for this workload; `z-ai/glm-5.2:free` didn't support tool calling on its free endpoint at all. This means "add a fallback provider" is not a one-line config change in practice — each one needs the same test-in-isolation-before-trusting discipline this project has applied to every other component. SambaNova and Cerebras remain completely unvalidated as of this ADR. We accept the mid-run mixed-provider caveat as a real, documented limitation rather than pretending failover produces clean comparisons — `print_summary()` explicitly flags it. We'd revisit Option C once multi-provider selection becomes a standing need rather than an occasional escape hatch.

## Interview-ready summary
"I built a shared OpenAI-compatible module so one implementation could serve four different fallback providers, wired in as permanent failover — not per-call round-robin, because mixing models mid-comparison would have reintroduced the exact 'not a valid before/after' problem I'd just spent time diagnosing with Gemini's own non-determinism. Every result gets tagged with which provider actually answered it, so a mixed-provider run is never silently reported as clean single-model numbers. The real lesson was that each new provider needed individual validation before I could trust it in the eval — one model reliably hallucinated calling a tool that didn't exist, another had a token limit too tight for the workload — 'just add a fallback' undersells how much per-provider testing that actually requires in practice."
