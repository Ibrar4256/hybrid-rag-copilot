"""Shared implementation for any OpenAI-compatible LLM provider (Groq,
OpenRouter, SambaNova, Cerebras — all expose the same chat-completions API
shape). Used as Gemini's failover when its daily quota is exhausted (see
ADR-007). Mirrors agent.py's tool-calling loop and synthesis.py's JSON-mode
+ entailment check, but via the `openai` SDK's provider-agnostic client
rather than `google.generativeai`.
"""

import json
from dataclasses import dataclass

from openai import OpenAI

from research_copilot.agent import Evidence
from research_copilot.query import Retriever
from research_copilot.synthesis import Claim
from research_copilot.telemetry import log_call, timed

MAX_SEARCH_ITERATIONS = 4

# (base_url, default model). Models chosen for confirmed tool-calling +
# JSON-mode support — not exhaustively benchmarked for quality.
PROVIDERS = {
    "groq": ("https://api.groq.com/openai/v1", "openai/gpt-oss-120b"),
    "openrouter": ("https://openrouter.ai/api/v1", "nex-agi/nex-n2.5-pro:free"),
    "sambanova": ("https://api.sambanova.ai/v1", "Meta-Llama-3.3-70B-Instruct"),
    "cerebras": ("https://api.cerebras.ai/v1", "llama-3.3-70b"),
}

_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search",
        "description": (
            "Search the document corpus for passages relevant to a query. Call this "
            "whenever you need evidence to answer the user's question. You may call it "
            "more than once with reformulated or follow-up queries if the first results "
            "are insufficient."
        ),
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
}

_SYSTEM_INSTRUCTION = (
    "You are a research assistant with access to EXACTLY ONE tool: search(query: "
    "string). Do not call any other tool (e.g. open, browse, cursor) under any "
    "circumstances — only search is available, calling anything else will fail. "
    "Decide whether you need to search at all, and if so, what to search "
    "for. Call `search` with focused queries. If the results already contain the "
    "facts you need, STOP searching and respond with plain text confirming you're "
    "ready to answer. Only search again if the previous results were genuinely "
    "insufficient — reformulate the query to target what's still missing, don't "
    "repeat similar searches. Do not write the final answer yet, that happens in a "
    "separate step.\n\n"
    "IMPORTANT: when forming search queries, focus on the factual content — company "
    "names, financial metrics, dates, and numbers. Strip out filing-type jargon "
    '(e.g. "10-K", "10-Q", "FY2023", "as of") that would bias retrieval toward '
    "document metadata instead of the actual financial data. For example, search "
    '"Western Alliance total deposits 2023" not "Western Alliance FY2023 10-K deposits".'
)

_SYNTHESIS_PROMPT = """Answer the question using ONLY the evidence below. Break your \
answer into discrete claims. For each claim, list which evidence IDs support it \
(e.g. "rag.md#0"). If a claim is not directly supported by any evidence, give it an \
empty source_chunk_ids list rather than inventing a citation.

IMPORTANT: Your default should be to ANSWER, not to abstain. Follow these rules:
1. If ANY evidence chunk contains a fact that directly answers (or partially answers) the \
question, extract it as a claim and cite it.
2. If the evidence contains the same metric but with slightly different phrasing, use it.
3. However, do NOT substitute data from a DIFFERENT time period, entity, or filing than \
what the question specifically asks about. If the question asks about June 2022 and you \
only have June 2023 data, that is NOT an answer — abstain and explain what's missing.
4. Only set "unable_to_answer" to true when the evidence genuinely does not contain the \
specific facts asked about (wrong time period, wrong entity, or completely unrelated topic).
5. When you do abstain, provide a specific reason explaining what you looked for and what \
the evidence actually contains instead.

Return JSON matching exactly this shape, nothing else:
{{"unable_to_answer": false, "reason": "", "claims": [{{"text": "...", "source_chunk_ids": ["..."]}}]}}

Evidence:
{evidence}

Question: {question}"""

_ENTAILMENT_PROMPT = """For each (claim, evidence) pair below, does the evidence text \
actually STATE the specific fact in the claim — not just relate to the same general \
topic? A chunk that's topically about the same company/date but doesn't contain the \
claimed fact should be marked false.

Pairs:
{pairs}

Return JSON matching exactly this shape, one entry per pair in order, nothing else:
{{"results": [true or false, ...]}}"""


@dataclass
class OAIClient:
    client: OpenAI
    model: str
    provider_name: str


def make_client(provider_name: str, api_key: str) -> OAIClient:
    base_url, model = PROVIDERS[provider_name]
    return OAIClient(OpenAI(api_key=api_key, base_url=base_url), model, provider_name)


def gather_evidence(
    question: str, oai: OAIClient, collection: str | None = None
) -> dict[str, Evidence]:
    retriever = Retriever(collection=collection)
    evidence: dict[str, Evidence] = {}
    messages = [
        {"role": "system", "content": _SYSTEM_INSTRUCTION},
        {"role": "user", "content": f"Question: {question}"},
    ]

    for _ in range(MAX_SEARCH_ITERATIONS):
        with timed() as t:
            response = oai.client.chat.completions.create(
                model=oai.model, messages=messages, tools=[_SEARCH_TOOL], temperature=0,
            )
        u = response.usage
        log_call(oai.provider_name, oai.model, "agent_search",
                 u.prompt_tokens if u else 0, u.completion_tokens if u else 0, t.ms)
        message = response.choices[0].message
        if not message.tool_calls:
            break

        evidence_before = set(evidence.keys())
        messages.append(message)
        for call in message.tool_calls:
            try:
                args = json.loads(call.function.arguments)
                query = args.get("query", question)
            except (json.JSONDecodeError, AttributeError):
                query = question
            results = retriever.search(query)
            for r in results:
                evidence.setdefault(
                    r["chunk_id"], Evidence(r["chunk_id"], r["source"], r["text"])
                )
            summary = "\n".join(f"[{r['chunk_id']}] {r['text'][:300]}" for r in results)
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": summary or "No results found.",
            })

        new_chunks = set(evidence.keys()) - evidence_before
        if not new_chunks:
            break

    return evidence


def _check_entailment(pairs: list[tuple[str, str]], oai: OAIClient) -> list[bool]:
    if not pairs:
        return []
    pairs_block = "\n\n".join(
        f"Pair {i+1}:\nCLAIM: {claim}\nEVIDENCE: {chunk}" for i, (claim, chunk) in enumerate(pairs)
    )
    prompt = _ENTAILMENT_PROMPT.format(pairs=pairs_block)
    with timed() as t:
        response = oai.client.chat.completions.create(
            model=oai.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0,
        )
    u = response.usage
    log_call(oai.provider_name, oai.model, "entailment",
             u.prompt_tokens if u else 0, u.completion_tokens if u else 0, t.ms)
    try:
        results = json.loads(response.choices[0].message.content)["results"]
        if len(results) == len(pairs):
            return [bool(r) for r in results]
    except (json.JSONDecodeError, KeyError, TypeError, AttributeError):
        pass
    return [False] * len(pairs)


def synthesize(question: str, evidence: dict[str, Evidence], oai: OAIClient) -> list[Claim]:
    if not evidence:
        return [Claim(text="No evidence was retrieved to answer this question.",
                       source_chunk_ids=[], supported=False, unable_to_answer=True)]

    evidence_block = "\n\n".join(f"[{e.chunk_id}] {e.text}" for e in evidence.values())
    prompt = _SYNTHESIS_PROMPT.format(evidence=evidence_block, question=question)
    with timed() as t:
        response = oai.client.chat.completions.create(
            model=oai.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0,
        )
    u = response.usage
    log_call(oai.provider_name, oai.model, "synthesis",
             u.prompt_tokens if u else 0, u.completion_tokens if u else 0, t.ms)
    parsed = json.loads(response.choices[0].message.content)
    if parsed.get("unable_to_answer"):
        reason = parsed.get("reason", "")
        return [Claim(text=reason or "The model declared it cannot answer this question "
                            "from the retrieved evidence.",
                       source_chunk_ids=[], supported=False,
                       unable_to_answer=True, reason=reason)]

    raw_claims = []
    for raw in parsed.get("claims", []):
        if not isinstance(raw, dict):
            raw_claims.append((str(raw), []))
            continue
        existing_ids = [cid for cid in raw.get("source_chunk_ids", []) if cid in evidence]
        raw_claims.append((raw["text"], existing_ids))

    check_pairs = []
    pair_index = []
    for i, (claim_text, existing_ids) in enumerate(raw_claims):
        for cid in existing_ids:
            check_pairs.append((claim_text, evidence[cid].text))
            pair_index.append((i, cid))

    entailment_results = _check_entailment(check_pairs, oai)
    entailed_by_claim: dict[int, list[str]] = {i: [] for i in range(len(raw_claims))}
    for (claim_idx, cid), entailed in zip(pair_index, entailment_results):
        if entailed:
            entailed_by_claim[claim_idx].append(cid)

    return [
        Claim(text=text, source_chunk_ids=entailed_by_claim[i], supported=bool(entailed_by_claim[i]))
        for i, (text, _) in enumerate(raw_claims)
    ]
