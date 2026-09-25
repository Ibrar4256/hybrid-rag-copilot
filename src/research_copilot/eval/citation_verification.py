"""Citation-verification eval (Task #5): runs the full agent loop + synthesis
against real questions and checks whether citations are actually correct, and
whether adversarial/unanswerable questions correctly trigger abstention.

This is the only Gemini-dependent eval stage — retrieval-stage metrics
(precision@k) are covered entirely offline by retrieval_ablation.py. Gemini's
free tier caps at a small number of requests/day (see retry.py's
DailyQuotaExhausted). Once hit, this fails over PERMANENTLY (not per-call) to
the first configured OpenAI-compatible provider (ADR-007) — a one-time
switch, not per-question round-robin, so results stay attributable: every
result is tagged with which provider actually answered it, and a run split
across providers is never silently presented as single-model numbers.
"""

from research_copilot.agent import gather_evidence as gemini_gather_evidence
from research_copilot.config import settings
from research_copilot.eval.ground_truth import ADVERSARIAL_QUESTIONS, build_ground_truth
from research_copilot.llm.openai_compatible import PROVIDERS, gather_evidence as oai_gather_evidence
from research_copilot.llm.openai_compatible import make_client, synthesize as oai_synthesize
from research_copilot.retry import DailyQuotaExhausted
from research_copilot.synthesis import synthesize as gemini_synthesize

COLLECTION = "sec_filings_banks_chunk200"


def _first_available_fallback() -> tuple[str, str] | None:
    """(provider_name, api_key) for the first configured OpenAI-compatible
    provider, in a fixed preference order, or None if none are configured."""
    key_map = {
        "groq": settings.groq_api_key,
        "openrouter": settings.openrouter_api_key,
        "sambanova": settings.sambanova_api_key,
        "cerebras": settings.cerebras_api_key,
    }
    for name in PROVIDERS:
        if key_map.get(name):
            return name, key_map[name]
    return None


def _evaluate_answerable(qid: int, question: str, correct_chunk_ids: set[str], oai) -> dict:
    if oai is None:
        evidence = gemini_gather_evidence(question, settings.gemini_api_key, COLLECTION)
        claims = gemini_synthesize(question, evidence, settings.gemini_api_key)
    else:
        evidence = oai_gather_evidence(question, oai, COLLECTION)
        claims = oai_synthesize(question, evidence, oai)

    supported_claims = [c for c in claims if c.supported]
    cited_correctly = any(
        any(cid in correct_chunk_ids for cid in c.source_chunk_ids) for c in supported_claims
    )
    return {
        "id": qid,
        "question": question,
        "provider": "gemini" if oai is None else oai.provider_name,
        "evidence_found": bool(evidence),
        "num_claims": len(claims),
        "num_supported": len(supported_claims),
        "cited_correct_chunk": cited_correctly,
        "declared_unable_to_answer": any(c.unable_to_answer for c in claims),
    }


def _evaluate_adversarial(qid: int, question: str, oai) -> dict:
    if oai is None:
        evidence = gemini_gather_evidence(question, settings.gemini_api_key, COLLECTION)
        claims = gemini_synthesize(question, evidence, settings.gemini_api_key)
    else:
        evidence = oai_gather_evidence(question, oai, COLLECTION)
        claims = oai_synthesize(question, evidence, oai)

    supported_claims = [c for c in claims if c.supported]
    declared_unable = any(c.unable_to_answer for c in claims)
    # An explicit "I can't answer" declaration, or literally no claims at all,
    # counts as abstention. A claim the model attempted but that failed
    # entailment (num_supported == 0 but declared_unable == False) does NOT
    # count — that's a fabrication that happened to get caught, not an honest
    # hedge, and conflating the two previously inflated this metric (see
    # KNOWN_TRADEOFFS.md).
    correctly_abstained = declared_unable or len(claims) == 0
    return {
        "id": qid,
        "question": question,
        "provider": "gemini" if oai is None else oai.provider_name,
        "evidence_found": bool(evidence),
        "num_claims": len(claims),
        "num_supported": len(supported_claims),
        "declared_unable_to_answer": declared_unable,
        "correctly_abstained": correctly_abstained,
    }


def run(question_ids: list[int] | None = None, force_provider: str | None = None) -> dict:
    """question_ids: optional subset (mixing answerable + adversarial IDs) to
    run. None = all 38. Starts on Gemini; permanently fails over to the first
    configured OpenAI-compatible provider (groq/openrouter/sambanova/cerebras)
    the moment Gemini's daily quota is hit, rather than stopping.

    force_provider: skip Gemini entirely and run the WHOLE eval on one named
    OpenAI-compatible provider (e.g. "groq") — for a clean, single-model,
    directly-comparable baseline run, avoiding the "split across providers"
    caveat that a mid-run failover would otherwise introduce."""
    if force_provider is None and not settings.gemini_api_key:
        raise ValueError("GEMINI_API_KEY required — see .env (or pass force_provider=)")

    answerable = build_ground_truth(chunk_size_tokens=200, chunk_overlap_tokens=40)
    if question_ids is not None:
        answerable = [g for g in answerable if g["id"] in question_ids]
        adversarial = [q for q in ADVERSARIAL_QUESTIONS if q[0] in question_ids]
    else:
        adversarial = ADVERSARIAL_QUESTIONS

    if force_provider is not None:
        key_map = {
            "groq": settings.groq_api_key,
            "openrouter": settings.openrouter_api_key,
            "sambanova": settings.sambanova_api_key,
            "cerebras": settings.cerebras_api_key,
        }
        key = key_map.get(force_provider)
        if not key:
            raise ValueError(f"No API key configured for '{force_provider}'")
        oai = make_client(force_provider, key)
    else:
        oai = None  # None = still on Gemini

    def _fail_over(qid):
        nonlocal oai
        fallback = _first_available_fallback()
        if fallback is None:
            return False
        name, key = fallback
        oai = make_client(name, key)
        print(f"Q{qid}: Gemini daily quota exhausted — failing over to '{name}' "
              f"for the remainder of this run")
        return True

    answerable_results = []
    adversarial_results = []

    for g in answerable:
        try:
            r = _evaluate_answerable(g["id"], g["question"], set(g["chunk_ids"]), oai)
            answerable_results.append(r)
            print(f"Q{g['id']} [answerable, {r['provider']}] cited_correct={r['cited_correct_chunk']} "
                  f"supported={r['num_supported']}/{r['num_claims']}")
        except DailyQuotaExhausted:
            if _fail_over(g["id"]):
                try:
                    r = _evaluate_answerable(g["id"], g["question"], set(g["chunk_ids"]), oai)
                    answerable_results.append(r)
                    print(f"Q{g['id']} [answerable, {r['provider']}] "
                          f"cited_correct={r['cited_correct_chunk']} "
                          f"supported={r['num_supported']}/{r['num_claims']}")
                except Exception as e:
                    print(f"Q{g['id']}: ERROR on fallback ({e!r}) — skipping, continuing")
            else:
                print(f"Q{g['id']}: DAILY QUOTA EXHAUSTED, no fallback configured — stopping")
                break
        except Exception as e:
            # One malformed question/response shouldn't lose every other
            # question's already-computed results (this crashed a full run
            # once — see WEEKLY_LOG.md).
            print(f"Q{g['id']}: ERROR ({e!r}) — skipping, continuing")

    for qid, question in adversarial:
        try:
            r = _evaluate_adversarial(qid, question, oai)
            adversarial_results.append(r)
            print(f"Q{qid} [adversarial, {r['provider']}] correctly_abstained={r['correctly_abstained']}")
        except DailyQuotaExhausted:
            if _fail_over(qid):
                try:
                    r = _evaluate_adversarial(qid, question, oai)
                    adversarial_results.append(r)
                    print(f"Q{qid} [adversarial, {r['provider']}] "
                          f"correctly_abstained={r['correctly_abstained']}")
                except Exception as e:
                    print(f"Q{qid}: ERROR on fallback ({e!r}) — skipping, continuing")
            else:
                print(f"Q{qid}: DAILY QUOTA EXHAUSTED, no fallback configured — stopping")
                break
        except Exception as e:
            print(f"Q{qid}: ERROR ({e!r}) — skipping, continuing")

    all_results = answerable_results + adversarial_results
    providers_used = {r["provider"] for r in all_results}
    return {
        "answerable": answerable_results,
        "adversarial": adversarial_results,
        "mixed_providers": len(providers_used) > 1,
        "providers_used": providers_used,
    }


def print_summary(results: dict) -> None:
    a = results["answerable"]
    adv = results["adversarial"]
    print("\n--- Summary ---")
    if a:
        correct = sum(r["cited_correct_chunk"] for r in a)
        print(f"Answerable: {correct}/{len(a)} correctly cited the ground-truth chunk "
              f"({correct/len(a):.1%})")
    if adv:
        correct = sum(r["correctly_abstained"] for r in adv)
        print(f"Adversarial: {correct}/{len(adv)} correctly abstained ({correct/len(adv):.1%})")
    if results["mixed_providers"]:
        counts = {}
        for r in a + adv:
            counts[r["provider"]] = counts.get(r["provider"], 0) + 1
        print(f"NOTE: mid-run failover occurred — provider breakdown: {counts}. "
              f"Not a clean single-model comparison.")
    elif a or adv:
        provider = (a or adv)[0]["provider"]
        print(f"All questions answered by a single provider: {provider}")


if __name__ == "__main__":
    results = run()
    print_summary(results)
