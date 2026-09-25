#!/usr/bin/env python3
"""Run a clean baseline eval across all 41 questions, logging structured
results + telemetry to data/eval/baseline_<timestamp>.json.

This captures the "before" snapshot for the Phase 1-3 improvement cycle.
Each question gets: bucket, expected, actual claims, round count, latency,
cost, pass/fail, abstention quality grade.

Usage:
  python scripts/run_baseline.py                    # all 41 questions
  python scripts/run_baseline.py --ids 1 2 41       # specific questions
  python scripts/run_baseline.py --tag before_loop_fix  # label the run
"""

import argparse
import json
import time
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from research_copilot.agent import gather_evidence as gemini_gather_evidence
from research_copilot.config import settings
from research_copilot.eval.ground_truth import ADVERSARIAL_QUESTIONS, build_ground_truth
from research_copilot.key_rotation import get_api_key, rotate_on_quota_error, reset as reset_keys
from research_copilot.retry import DailyQuotaExhausted
from research_copilot.synthesis import synthesize as gemini_synthesize
from research_copilot.telemetry import end_query, load_log, start_query

COLLECTION = "sec_filings_banks_chunk200"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "eval"


def _run_question(question: str) -> tuple[list, dict, str]:
    """Run one question through the full pipeline, returning (claims, evidence, query_id).
    Automatically rotates through all API keys on daily quota exhaustion."""
    qid = start_query()
    while True:
        try:
            key = get_api_key()
            evidence = gemini_gather_evidence(question, key, COLLECTION)
            claims = gemini_synthesize(question, evidence, key)
            break
        except DailyQuotaExhausted:
            new_key = rotate_on_quota_error()
            if not new_key:
                end_query()
                raise
            print(f"(rotated key) ", end="", flush=True)
    end_query()
    return claims, evidence, qid


def _extract_telemetry(query_id: str) -> dict:
    """Pull telemetry for a specific query_id."""
    calls = [r for r in load_log() if r.get("query_id") == query_id]
    if not calls:
        return {}
    search_rounds = sum(1 for c in calls if c["call_type"] == "agent_search")
    rerank_lat = sum(c["latency_ms"] for c in calls if c["call_type"] == "rerank")
    llm_lat = sum(c["latency_ms"] for c in calls if c["call_type"] in
                  ("agent_init", "agent_search", "synthesis", "entailment"))
    return {
        "query_id": query_id,
        "total_calls": len(calls),
        "search_rounds": search_rounds,
        "total_tokens_in": sum(c["tokens_in"] for c in calls),
        "total_tokens_out": sum(c["tokens_out"] for c in calls),
        "total_latency_ms": round(sum(c["latency_ms"] for c in calls), 1),
        "rerank_latency_ms": round(rerank_lat, 1),
        "llm_latency_ms": round(llm_lat, 1),
        "total_cost_usd": round(sum(c["cost_usd"] for c in calls), 8),
        "has_entailment": any(c["call_type"] == "entailment" for c in calls),
        "calls": calls,
    }


def run_baseline(question_ids: list[int] | None = None, tag: str = "baseline") -> dict:
    reset_keys()
    answerable = build_ground_truth(chunk_size_tokens=200, chunk_overlap_tokens=40)
    adversarial = ADVERSARIAL_QUESTIONS

    if question_ids is not None:
        answerable = [g for g in answerable if g["id"] in question_ids]
        adversarial = [q for q in adversarial if q[0] in question_ids]

    results = []
    total = len(answerable) + len(adversarial)
    done = 0

    for g in answerable:
        done += 1
        print(f"[{done}/{total}] Q{g['id']} (bucket {g['bucket']})...", end=" ", flush=True)
        try:
            claims, evidence, qid = _run_question(g["question"])
            tel = _extract_telemetry(qid)

            supported = [c for c in claims if c.supported]
            cited_correct = any(
                any(cid in set(g["chunk_ids"]) for cid in c.source_chunk_ids)
                for c in supported
            )
            declared_unable = any(c.unable_to_answer for c in claims)

            if cited_correct:
                verdict = "pass"
            elif declared_unable:
                verdict = "abstained_incorrectly"
            else:
                verdict = "fail"

            result = {
                "id": g["id"],
                "bucket": g["bucket"],
                "type": "answerable",
                "question": g["question"],
                "expected_chunk_ids": g["chunk_ids"],
                "actual_claims": [
                    {"text": c.text, "source_chunk_ids": c.source_chunk_ids,
                     "supported": c.supported, "unable_to_answer": c.unable_to_answer,
                     "reason": getattr(c, "reason", "")}
                    for c in claims
                ],
                "evidence_count": len(evidence),
                "cited_correct_chunk": cited_correct,
                "declared_unable": declared_unable,
                "verdict": verdict,
                "telemetry": tel,
            }
            results.append(result)
            reason_tag = ""
            if declared_unable:
                r0 = getattr(claims[0], "reason", "") if claims else ""
                reason_tag = f" | reason: {r0[:60]}" if r0 else ""
            print(f"{verdict} | {tel.get('search_rounds', '?')} rounds | "
                  f"{tel.get('total_latency_ms', 0)/1000:.1f}s | "
                  f"${tel.get('total_cost_usd', 0):.5f}{reason_tag}")

        except DailyQuotaExhausted:
            print("QUOTA EXHAUSTED — stopping answerable questions")
            break
        except Exception as e:
            print(f"ERROR: {e!r}")
            results.append({
                "id": g["id"], "bucket": g["bucket"], "type": "answerable",
                "question": g["question"], "verdict": "error", "error": str(e),
            })

    for qid_num, question in adversarial:
        done += 1
        print(f"[{done}/{total}] Q{qid_num} (adversarial)...", end=" ", flush=True)
        try:
            claims, evidence, qid = _run_question(question)
            tel = _extract_telemetry(qid)

            declared_unable = any(c.unable_to_answer for c in claims)
            correctly_abstained = declared_unable or len(claims) == 0
            supported = [c for c in claims if c.supported]

            if correctly_abstained:
                verdict = "pass"
            elif len(supported) > 0:
                verdict = "fail_confident_wrong"
            else:
                verdict = "fail_fabricated_unsupported"

            abstention_text = ""
            if claims:
                abstention_text = claims[0].text

            result = {
                "id": qid_num,
                "bucket": 3,
                "type": "adversarial",
                "question": question,
                "actual_claims": [
                    {"text": c.text, "source_chunk_ids": c.source_chunk_ids,
                     "supported": c.supported, "unable_to_answer": c.unable_to_answer,
                     "reason": getattr(c, "reason", "")}
                    for c in claims
                ],
                "evidence_count": len(evidence),
                "correctly_abstained": correctly_abstained,
                "declared_unable": declared_unable,
                "abstention_text": abstention_text,
                "verdict": verdict,
                "telemetry": tel,
            }
            results.append(result)
            print(f"{verdict} | {tel.get('search_rounds', '?')} rounds | "
                  f"{tel.get('total_latency_ms', 0)/1000:.1f}s | "
                  f"${tel.get('total_cost_usd', 0):.5f}")

        except DailyQuotaExhausted:
            print("QUOTA EXHAUSTED — stopping adversarial questions")
            break
        except Exception as e:
            print(f"ERROR: {e!r}")
            results.append({
                "id": qid_num, "bucket": 3, "type": "adversarial",
                "question": question, "verdict": "error", "error": str(e),
            })

    # Aggregate
    completed = [r for r in results if r["verdict"] != "error"]
    answerable_r = [r for r in completed if r["type"] == "answerable"]
    adversarial_r = [r for r in completed if r["type"] == "adversarial"]

    summary = {
        "tag": tag,
        "timestamp": time.time(),
        "total_questions": len(results),
        "completed": len(completed),
        "errors": len(results) - len(completed),
    }

    if answerable_r:
        summary["answerable"] = {
            "total": len(answerable_r),
            "pass": sum(1 for r in answerable_r if r["verdict"] == "pass"),
            "fail": sum(1 for r in answerable_r if r["verdict"] == "fail"),
            "abstained_incorrectly": sum(1 for r in answerable_r if r["verdict"] == "abstained_incorrectly"),
            "avg_rounds": round(sum(r["telemetry"].get("search_rounds", 0) for r in answerable_r) / len(answerable_r), 1),
            "avg_latency_ms": round(sum(r["telemetry"].get("total_latency_ms", 0) for r in answerable_r) / len(answerable_r), 0),
            "avg_cost_usd": round(sum(r["telemetry"].get("total_cost_usd", 0) for r in answerable_r) / len(answerable_r), 8),
        }

    if adversarial_r:
        summary["adversarial"] = {
            "total": len(adversarial_r),
            "pass": sum(1 for r in adversarial_r if r["verdict"] == "pass"),
            "fail_confident_wrong": sum(1 for r in adversarial_r if r["verdict"] == "fail_confident_wrong"),
            "fail_fabricated_unsupported": sum(1 for r in adversarial_r if r["verdict"] == "fail_fabricated_unsupported"),
            "avg_rounds": round(sum(r["telemetry"].get("search_rounds", 0) for r in adversarial_r) / len(adversarial_r), 1),
            "avg_latency_ms": round(sum(r["telemetry"].get("total_latency_ms", 0) for r in adversarial_r) / len(adversarial_r), 0),
            "avg_cost_usd": round(sum(r["telemetry"].get("total_cost_usd", 0) for r in adversarial_r) / len(adversarial_r), 8),
        }

    if completed:
        tels = [r["telemetry"] for r in completed if r.get("telemetry")]
        if tels:
            summary["overall"] = {
                "avg_rounds": round(sum(t.get("search_rounds", 0) for t in tels) / len(tels), 1),
                "avg_latency_ms": round(sum(t.get("total_latency_ms", 0) for t in tels) / len(tels), 0),
                "avg_rerank_pct": round(sum(t.get("rerank_latency_ms", 0) for t in tels) / sum(t.get("total_latency_ms", 1) for t in tels) * 100, 1),
                "avg_cost_usd": round(sum(t.get("total_cost_usd", 0) for t in tels) / len(tels), 8),
                "total_cost_usd": round(sum(t.get("total_cost_usd", 0) for t in tels), 6),
            }

    output = {"summary": summary, "results": results}

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"baseline_{tag}_{int(time.time())}.json"
    out_path = OUTPUT_DIR / filename
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Results saved to {out_path}")
    print(json.dumps(summary, indent=2))
    return output


def main():
    parser = argparse.ArgumentParser(description="Run baseline eval")
    parser.add_argument("--ids", nargs="+", type=int, help="Specific question IDs")
    parser.add_argument("--tag", default="before_loop_fix", help="Label for this run")
    args = parser.parse_args()
    run_baseline(question_ids=args.ids, tag=args.tag)


if __name__ == "__main__":
    main()
