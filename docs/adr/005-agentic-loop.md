# ADR-005: Agentic Loop for Research Copilot

## Context
The current pipeline (ADR-001 through ADR-004) performs one fixed retrieve → rerank pass regardless of the question. The project requires the model to decide whether to search at all, what to search for (possibly multiple or reformulated queries), and when it has gathered enough evidence to answer — not a single hardcoded retrieve-then-generate call. This is also the first hands-on agent-loop implementation in the portfolio; Project 2 later asks for a hand-rolled agent loop specifically so the mechanics under any framework are understood, and there's no reason to defer that learning past Project 1 when the spec already requires equivalent capability here.

## Options Considered

### Option A: Hand-rolled loop using native LLM tool-calling (Gemini function calling) — CHOSEN
- How it works: Expose a `search(query: str)` tool to the LLM via Gemini's native function-calling API. Loop: call the LLM with the user question and the tool definition; if it calls `search`, run retrieval and feed the results back as a tool result; repeat until the model returns a final answer instead of another tool call, or a max-iteration cap is hit.
- Pros: Full ownership of the state machine (message history, loop termination, iteration cap) — genuine understanding of what a framework like LangGraph abstracts; no extra dependency; supported on Gemini's free tier; directly transfers to Project 2's explicit hand-rolled-agent requirement.
- Cons: Must handle edge cases directly — malformed tool-call responses, infinite-loop guards, retrying a dropped tool call.
- Cost/latency/complexity profile: Free (Gemini free tier); latency scales with number of search iterations (bounded by the iteration cap); moderate implementation complexity (a message-history loop, not a full framework).

### Option B: LangGraph — rejected (for this project)
- How it works: Model the loop as a graph (decide/search/answer nodes, conditional routing edges) using LangGraph's state management and checkpointing.
- Pros: Handles state/checkpointing/routing out of the box, less boilerplate, a legitimate and widely-used production framework.
- Cons: Abstracts away the exact mechanics this project is meant to teach; adds a fairly heavy dependency for what is still a simple 2-3 step loop at this scope.
- Why we didn't use it here: The value of a graph framework shows up with multiple distinct tools and branching human-approval paths — Project 1's single-tool search loop doesn't need it yet.
- When it WOULD be the better choice: Project 2 (multi-tool support agent with human-in-the-loop escalation), or any agent with more than 2-3 tools and non-linear branching.

### Option C: Prompt-only loop (provider-agnostic JSON action parsing, no native tool-calling) — rejected
- How it works: No dependency on provider-specific function-calling support; instruct the model via system prompt to emit `{"action": "search"|"answer", ...}` JSON each turn, parsed manually.
- Pros: Portable across any LLM provider, including ones with weak or no native function-calling support.
- Cons: Less reliable than native tool calling — free-form JSON output is more prone to malformed responses with no schema enforcement from the provider; effectively reimplements what native tool-calling already provides.
- Why we didn't use it here: Gemini's free tier has solid native function-calling support, so there's no portability problem to solve yet; adopting this fallback preemptively would trade away schema enforcement for a portability benefit we don't currently need.
- When it WOULD be the better choice: If the chosen LLM provider lacks reliable native function calling and a provider-agnostic action format is required.

## Decision
We implement a hand-rolled agent loop using Gemini's native function-calling API with a single `search` tool. The loop maintains message history across turns, feeds tool results back to the model, and terminates when the model emits a final answer or a max-iteration cap is reached (guarding against runaway loops on ambiguous queries).

## Consequences
We take on directly implementing and testing the loop's edge cases (malformed tool calls, iteration caps, dropped tool results) rather than inheriting them from a framework. In exchange we get a fully transparent, debuggable agent loop and a hands-on understanding of the mechanics that transfers directly to Project 2. We'd revisit this if Project 1's tool surface grew beyond a single `search` tool with non-linear branching, at which point LangGraph's graph-based routing would earn its complexity.

## Interview-ready summary
"I hand-rolled the agent loop using Gemini's native function-calling API rather than reaching for LangGraph, because with a single `search` tool the framework's state-management and routing features weren't earning their complexity yet — and building the loop by hand meant I actually understand the mechanics a framework would otherwise hide. I'd reach for LangGraph on my next project, which needs multiple tools and human-in-the-loop branching, where a graph-based state machine genuinely pays for itself."
