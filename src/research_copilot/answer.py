import sys

from research_copilot.agent import Evidence, gather_evidence
from research_copilot.config import settings
from research_copilot.key_rotation import get_api_key
from research_copilot.retry import DailyQuotaExhausted
from research_copilot.key_rotation import rotate_on_quota_error
from research_copilot.synthesis import Claim, render, synthesize
from research_copilot.telemetry import end_query, query_stats, start_query


def _require_api_key() -> None:
    if not settings.gemini_key_pool:
        raise ValueError(
            "GEMINI_API_KEY or GEMINI_API_KEYS is required for the agent loop "
            "and citation synthesis (set it in .env — see ADR-005 and ADR-006)."
        )


def _run_pipeline(question, collection):
    while True:
        try:
            key = get_api_key()
            evidence = gather_evidence(question, key, collection)
            claims = synthesize(question, evidence, key)
            return claims, evidence
        except DailyQuotaExhausted:
            if not rotate_on_quota_error():
                raise


def answer_question(question: str, collection: str | None = None) -> str:
    _require_api_key()
    start_query()
    claims, _ = _run_pipeline(question, collection)
    end_query()
    return render(claims)


def answer_question_structured(
    question: str, collection: str | None = None
) -> tuple[list[Claim], dict[str, Evidence], dict]:
    _require_api_key()
    qid = start_query()
    claims, evidence = _run_pipeline(question, collection)
    end_query()
    stats = query_stats(qid)
    return claims, evidence, stats


if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else "What is retrieval-augmented generation?"
    coll = sys.argv[2] if len(sys.argv) > 2 else None
    print(answer_question(q, coll))
