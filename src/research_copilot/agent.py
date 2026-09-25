from dataclasses import dataclass

import google.generativeai as genai

from research_copilot.key_rotation import get_api_key, rotate_on_quota_error
from research_copilot.query import Retriever
from research_copilot.retry import DailyQuotaExhausted, with_backoff
from research_copilot.telemetry import log_call, timed

# "gemini-flash-latest" resolves to a "thinking" model variant (~12s/call even
# for trivial prompts, confirmed via direct API timing) — far too slow for a
# multi-call agent loop. "gemini-flash-lite-latest" resolves to a non-thinking
# model, responds in <1s, and still supports function calling correctly.
MODEL_NAME = "models/gemini-flash-lite-latest"
MAX_SEARCH_ITERATIONS = 4
_REQUEST_TIMEOUT_SECONDS = 90

_SEARCH_FUNCTION = genai.protos.FunctionDeclaration(
    name="search",
    description=(
        "Search the document corpus for passages relevant to a query. Call this "
        "whenever you need evidence to answer the user's question. You may call it "
        "more than once with reformulated or follow-up queries if the first results "
        "are insufficient."
    ),
    parameters=genai.protos.Schema(
        type=genai.protos.Type.OBJECT,
        properties={"query": genai.protos.Schema(type=genai.protos.Type.STRING)},
        required=["query"],
    ),
)
_SEARCH_TOOL = genai.protos.Tool(function_declarations=[_SEARCH_FUNCTION])

_SYSTEM_INSTRUCTION = (
    "You are a research assistant with access to a `search` tool over a document "
    "corpus. Decide whether you need to search at all, and if so, what to search "
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


@dataclass
class Evidence:
    chunk_id: str
    source: str
    text: str


def _function_calls(response):
    return [
        part.function_call
        for part in response.candidates[0].content.parts
        if part.function_call
    ]


def gather_evidence(
    question: str, api_key: str | None = None, collection: str | None = None
) -> dict[str, Evidence]:
    """Hand-rolled agent loop (ADR-005): the model decides whether/what/when to
    search. We own the loop termination and iteration cap ourselves rather than
    delegating to a framework's automatic function-calling."""
    api_key = api_key or get_api_key()
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        MODEL_NAME,
        tools=[_SEARCH_TOOL],
        system_instruction=_SYSTEM_INSTRUCTION,
        # Pinned for eval reproducibility — two runs of the same question were
        # observed retrieving different evidence via different search queries,
        # making before/after eval comparisons unreliable (see WEEKLY_LOG.md).
        generation_config={"temperature": 0},
    )
    chat = model.start_chat(history=[])
    retriever = Retriever(collection=collection)
    evidence: dict[str, Evidence] = {}

    with timed() as t:
        response = with_backoff(
            chat.send_message,
            f"Question: {question}",
            request_options={"timeout": _REQUEST_TIMEOUT_SECONDS},
            label="agent_init",
        )
    u = response.usage_metadata
    log_call("gemini", MODEL_NAME, "agent_init", u.prompt_token_count, u.candidates_token_count, t.ms)

    for _ in range(MAX_SEARCH_ITERATIONS):
        calls = _function_calls(response)
        if not calls:
            break

        evidence_before = set(evidence.keys())
        function_responses = []
        for call in calls:
            query = call.args.get("query", question)
            results = retriever.search(query)
            for r in results:
                evidence.setdefault(
                    r["chunk_id"], Evidence(r["chunk_id"], r["source"], r["text"])
                )
            summary = "\n".join(f"[{r['chunk_id']}] {r['text'][:300]}" for r in results)
            function_responses.append(
                genai.protos.Part(
                    function_response=genai.protos.FunctionResponse(
                        name="search",
                        response={"result": summary or "No results found."},
                    )
                )
            )

        new_chunks = set(evidence.keys()) - evidence_before
        if not new_chunks:
            break
        with timed() as t:
            response = with_backoff(
                chat.send_message,
                genai.protos.Content(parts=function_responses),
                request_options={"timeout": _REQUEST_TIMEOUT_SECONDS},
                label="agent_search",
            )
        u = response.usage_metadata
        log_call("gemini", MODEL_NAME, "agent_search", u.prompt_token_count, u.candidates_token_count, t.ms)

    return evidence
