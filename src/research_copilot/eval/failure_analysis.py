"""Failure-analysis diagnostic (Task #6): captures full detail (retrieved
evidence + raw claims with citations) for specific questions, rather than
just the pass/fail booleans citation_verification.py reports. Used to
localize WHERE a failure happens — retrieval, synthesis, or verification."""

from research_copilot.agent import gather_evidence
from research_copilot.config import settings
from research_copilot.eval.ground_truth import ADVERSARIAL_QUESTIONS, build_ground_truth
from research_copilot.synthesis import synthesize

COLLECTION = "sec_filings_banks_chunk200"


def diagnose(qid: int, question: str, correct_chunk_ids: set[str] | None = None) -> None:
    evidence = gather_evidence(question, settings.gemini_api_key, COLLECTION)
    claims = synthesize(question, evidence, settings.gemini_api_key)

    print(f"\n{'='*70}\nQ{qid}: {question}")
    if correct_chunk_ids:
        print(f"Ground truth chunk(s): {correct_chunk_ids}")

    print(f"\n--- Retrieved evidence ({len(evidence)} chunks) ---")
    for e in evidence.values():
        marker = " <-- GROUND TRUTH" if correct_chunk_ids and e.chunk_id in correct_chunk_ids else ""
        print(f"  [{e.chunk_id}]{marker}")
        # Full text, not truncated — a 250-char preview once hid the exact
        # sentence that made a "hallucination" turn out to be a correct,
        # fully-supported claim (see WEEKLY_LOG.md).
        print(f"    {e.text.strip()}")

    print(f"\n--- Synthesized claims ({len(claims)}) ---")
    for c in claims:
        if c.unable_to_answer:
            status = "DECLARED UNABLE TO ANSWER"
        elif c.supported:
            status = "SUPPORTED"
        else:
            status = "UNSUPPORTED"
        print(f"  [{status}] {c.text}")
        print(f"    cited: {c.source_chunk_ids}")


if __name__ == "__main__":
    import sys

    ids = [int(x) for x in sys.argv[1:]] if len(sys.argv) > 1 else []

    gt = {g["id"]: g for g in build_ground_truth(chunk_size_tokens=200, chunk_overlap_tokens=40)}
    adv = dict(ADVERSARIAL_QUESTIONS)

    for qid in ids:
        if qid in gt:
            diagnose(qid, gt[qid]["question"], set(gt[qid]["chunk_ids"]))
        elif qid in adv:
            diagnose(qid, adv[qid])
        else:
            print(f"Q{qid}: not found in ground truth or adversarial set")
