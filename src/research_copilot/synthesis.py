import json
from dataclasses import dataclass

import google.generativeai as genai

from research_copilot.agent import Evidence
from research_copilot.key_rotation import get_api_key
from research_copilot.retry import with_backoff
from research_copilot.telemetry import log_call, timed

# See agent.py for why "flash-lite" over the "flash-latest" thinking variant.
MODEL_NAME = "models/gemini-flash-lite-latest"
_REQUEST_TIMEOUT_SECONDS = 90

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

# A cross-encoder reranker was tried first for this check and rejected: with
# REALISTIC full-chunk text (not short clean snippets), it scored a confirmed
# bad citation (deposits claim, cited chunk about branch locations) at 0.97+,
# indistinguishable from confirmed good citations (also 0.97+) — a reranker
# measures topical relevance ("same company, same date"), not entailment
# ("does this text state this specific fact"), and those are different
# questions. An LLM-based check, tested against the same real cases, correctly
# separated them (false/true) — costs one extra Gemini call per question
# (batched across all of that question's claims, not per-claim) but actually
# works.
_ENTAILMENT_PROMPT = """For each (claim, evidence) pair below, does the evidence text \
actually STATE the specific fact in the claim — not just relate to the same general \
topic? A chunk that's topically about the same company/date but doesn't contain the \
claimed fact should be marked false.

Pairs:
{pairs}

Return JSON matching exactly this shape, one entry per pair in order, nothing else:
{{"results": [true or false, ...]}}"""


@dataclass
class Claim:
    text: str
    source_chunk_ids: list[str]
    supported: bool
    unable_to_answer: bool = False
    reason: str = ""


def synthesize(question: str, evidence: dict[str, Evidence], api_key: str | None = None) -> list[Claim]:
    """Structured-output citation synthesis (ADR-006): the model must attach
    source_chunk_ids to each claim. _verify() then checks those IDs against
    the actual retrieved evidence (existence) AND runs a separate LLM-based
    entailment check (does the cited chunk's text actually state the claimed
    fact, not just exist) — see ADR-006's known-limitation fix in
    KNOWN_TRADEOFFS.md."""
    if not evidence:
        return [Claim(text="No evidence was retrieved to answer this question.",
                       source_chunk_ids=[], supported=False, unable_to_answer=True)]

    api_key = api_key or get_api_key()
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        MODEL_NAME,
        # temperature=0 pinned for eval reproducibility (see agent.py, WEEKLY_LOG.md)
        generation_config={"response_mime_type": "application/json", "temperature": 0},
    )
    evidence_block = "\n\n".join(f"[{e.chunk_id}] {e.text}" for e in evidence.values())
    prompt = _SYNTHESIS_PROMPT.format(evidence=evidence_block, question=question)

    with timed() as t:
        response = with_backoff(
            model.generate_content,
            prompt,
            request_options={"timeout": _REQUEST_TIMEOUT_SECONDS},
            label="synthesis",
        )
    u = response.usage_metadata
    log_call("gemini", MODEL_NAME, "synthesis", u.prompt_token_count, u.candidates_token_count, t.ms)
    parsed = json.loads(response.text)
    if parsed.get("unable_to_answer"):
        reason = parsed.get("reason", "")
        return [Claim(text=reason or "The model declared it cannot answer this question "
                            "from the retrieved evidence.",
                       source_chunk_ids=[], supported=False,
                       unable_to_answer=True, reason=reason)]
    return _verify(parsed, evidence, api_key)


def _check_entailment(
    pairs: list[tuple[str, str]], api_key: str
) -> list[bool]:
    """pairs: [(claim_text, chunk_text), ...]. One batched Gemini call covering
    every citation across all of a question's claims, not one call per claim."""
    if not pairs:
        return []
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        MODEL_NAME,
        # temperature=0 pinned for eval reproducibility (see agent.py, WEEKLY_LOG.md)
        generation_config={"response_mime_type": "application/json", "temperature": 0},
    )
    pairs_block = "\n\n".join(
        f"Pair {i+1}:\nCLAIM: {claim}\nEVIDENCE: {chunk}" for i, (claim, chunk) in enumerate(pairs)
    )
    prompt = _ENTAILMENT_PROMPT.format(pairs=pairs_block)
    with timed() as t:
        response = with_backoff(
            model.generate_content, prompt, request_options={"timeout": _REQUEST_TIMEOUT_SECONDS},
            label="entailment",
        )
    u = response.usage_metadata
    log_call("gemini", MODEL_NAME, "entailment", u.prompt_token_count, u.candidates_token_count, t.ms)
    try:
        results = json.loads(response.text)["results"]
        if len(results) == len(pairs):
            return [bool(r) for r in results]
    except (json.JSONDecodeError, KeyError, TypeError):
        pass
    # Malformed entailment response: fail closed (treat as not-entailed)
    # rather than silently trusting an unparseable check.
    return [False] * len(pairs)


def _verify(parsed: dict, evidence: dict[str, Evidence], api_key: str) -> list[Claim]:
    """Two-stage check: a citation only counts if (1) it names a chunk ID we
    actually retrieved, AND (2) an LLM-based entailment check confirms that
    chunk's text actually states the claimed fact, not just exists. Claims
    with no valid citation are flagged, not dropped — silently suppressing
    them would hide exactly the failure mode we want visible."""
    raw_claims = []
    for raw in parsed.get("claims", []):
        # The model occasionally returns a claim as a bare string instead of
        # the requested {"text", "source_chunk_ids"} object (observed in
        # practice during eval runs) — treat it as an uncited claim rather
        # than crashing on the missing structure.
        if not isinstance(raw, dict):
            raw_claims.append((str(raw), []))
            continue
        existing_ids = [cid for cid in raw.get("source_chunk_ids", []) if cid in evidence]
        raw_claims.append((raw["text"], existing_ids))

    # Collect every (claim, chunk) pair needing an entailment check across
    # ALL claims, so the whole question costs one extra Gemini call.
    check_pairs = []
    pair_index = []  # (claim_idx, chunk_id) per entry in check_pairs
    for i, (claim_text, existing_ids) in enumerate(raw_claims):
        for cid in existing_ids:
            check_pairs.append((claim_text, evidence[cid].text))
            pair_index.append((i, cid))

    entailment_results = _check_entailment(check_pairs, api_key)
    entailed_by_claim: dict[int, list[str]] = {i: [] for i in range(len(raw_claims))}
    for (claim_idx, cid), entailed in zip(pair_index, entailment_results):
        if entailed:
            entailed_by_claim[claim_idx].append(cid)

    return [
        Claim(text=text, source_chunk_ids=entailed_by_claim[i], supported=bool(entailed_by_claim[i]))
        for i, (text, _) in enumerate(raw_claims)
    ]


def render(claims: list[Claim]) -> str:
    lines = []
    for claim in claims:
        if claim.supported:
            marker = " " + " ".join(f"[{cid}]" for cid in claim.source_chunk_ids)
        else:
            marker = " [UNSUPPORTED — no citation]"
        lines.append(claim.text + marker)
    return "\n".join(lines)
