# ADR-008: UI Architecture for Research Copilot

## Context
Every prior interaction with this project has been CLI-only (`research_copilot.answer`). CLAUDE.md's original project scope calls for a "lightweight React/Next.js frontend" with "token-by-token streaming response with inline citation markers." No backend API layer exists yet either — the agent loop and citation synthesis are only reachable as direct Python function calls, not over HTTP.

## Options Considered

### Option A: FastAPI backend + vanilla HTML/JS frontend (no build step) — CHOSEN (walking skeleton)
- How it works: FastAPI serves both a JSON API endpoint (wrapping the existing `answer_question()` pipeline) and a single static HTML page with plain JS (fetch + DOM updates). No Node.js, no npm, no build tooling.
- Pros: fastest path to something working end-to-end, zero new toolchain, easy to debug, matches this project's own incremental-build philosophy (thin path first, then layer complexity — the same reasoning used for every prior component).
- Cons: doesn't demonstrate React/modern-frontend-framework skills.
- Cost/latency/complexity profile: free, low complexity, minimal new dependencies (fastapi + uvicorn).

### Option B: FastAPI backend + React/Next.js frontend — rejected for now, planned as a later layer
- How it works: matches CLAUDE.md's originally suggested stack exactly — a separate React/Next.js frontend calling the FastAPI backend.
- Pros: stronger portfolio signal (demonstrates real frontend engineering, not just a script with a form), room to grow (routing, component structure, richer streaming UI patterns).
- Cons: more moving parts (Node toolchain, CORS between dev servers, more to debug) before anything is demoable.
- Why we didn't use it here: per this project's own established pattern, ship the walking skeleton first and layer in complexity once it works — same logic applied to retrieval (dense-only → hybrid → reranking) and chunking (baseline → tuned → Parent-Child ablation). Jumping straight to a React frontend before the API layer itself is proven risks debugging two new things (API design AND a new frontend toolchain) at once.
- When it WOULD be the better choice: once Option A's API contract is proven out and stable, and the portfolio value of demonstrating frontend framework skill outweighs the added setup time.

### Option C: Streamlit or Gradio — rejected
- How it works: pure-Python UI framework: the "UI" is Python code rendering widgets, no separate frontend code at all.
- Pros: extremely fast to build.
- Cons: weakest portfolio signal of the three options — demonstrates neither API design nor frontend engineering; doesn't naturally support the inline clickable citation-marker interaction CLAUDE.md's spec calls for.
- Why we didn't use it here: this project has consistently optimized for learning depth and interview signal over raw build speed (see ADR-001's vector-store reasoning, for the same logic applied here) — Streamlit optimizes purely for speed at the cost of that signal.

## Decision
We build Option A: a FastAPI backend exposing the existing agent-loop + citation-synthesis pipeline as a JSON API, served alongside a single static HTML/JS page with no build step. True token-by-token streaming is explicitly deferred — `synthesize()` currently uses a blocking `generate_content()` call, not Gemini's streaming API, and switching to real SSE-based token streaming is separate, later work, not bundled into this walking skeleton.

## Consequences
We give up, for now, the stronger frontend-engineering portfolio signal a React/Next.js build would provide — explicitly planned as Option B, a follow-up layer once this API-first version is proven working. We also ship without real token-by-token streaming despite CLAUDE.md's original spec calling for it — the v1 UI shows a loading state then the full answer at once, a real, acknowledged gap (see KNOWN_TRADEOFFS.md), not a silent scope cut.

## Interview-ready summary
"I built the UI as a walking skeleton — a FastAPI backend wrapping the existing agent loop, served alongside a plain HTML/JS page with no build step — before reaching for React or real token streaming, both of which were on the original plan. That's the same incremental-build pattern I used throughout the rest of the project: prove the thin path works end-to-end, then layer in complexity, rather than debugging a new frontend toolchain and a new API contract at the same time. React and real SSE streaming are both planned next layers, not things I decided the project didn't need."
