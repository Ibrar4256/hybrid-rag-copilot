#!/usr/bin/env python3
"""Cost/latency analysis over telemetry logs (ADR-009).

Reads data/telemetry/calls.jsonl, groups by query_id, and prints:
  - Per-query averages (tokens, latency, cost)
  - Breakdown by call type
  - "Cost per 1,000 users/month" at various query-per-day rates

Usage:
  python scripts/cost_analysis.py                    # all data
  python scripts/cost_analysis.py --last 10          # last 10 queries only
  python scripts/cost_analysis.py --json             # machine-readable output
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

LOG_FILE = Path(__file__).parent.parent / "data" / "telemetry" / "calls.jsonl"


def load_records() -> list[dict]:
    if not LOG_FILE.exists():
        return []
    records = []
    with open(LOG_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def group_by_query(records: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        qid = r.get("query_id") or "unknown"
        groups[qid].append(r)
    return groups


def analyze(records: list[dict], last_n: int | None = None) -> dict:
    by_query = group_by_query(records)
    query_ids = sorted(by_query.keys(), key=lambda qid: min(r["ts"] for r in by_query[qid]))

    if last_n:
        query_ids = query_ids[-last_n:]

    if not query_ids:
        return {"error": "No telemetry data found."}

    query_costs = []
    query_latencies = []
    query_tokens_in = []
    query_tokens_out = []
    query_call_counts = []

    for qid in query_ids:
        calls = by_query[qid]
        query_costs.append(sum(c["cost_usd"] for c in calls))
        query_latencies.append(sum(c["latency_ms"] for c in calls))
        query_tokens_in.append(sum(c["tokens_in"] for c in calls))
        query_tokens_out.append(sum(c["tokens_out"] for c in calls))
        query_call_counts.append(len(calls))

    n = len(query_ids)
    avg_cost = sum(query_costs) / n
    avg_latency = sum(query_latencies) / n
    avg_tokens_in = sum(query_tokens_in) / n
    avg_tokens_out = sum(query_tokens_out) / n

    by_type: dict[str, dict] = defaultdict(
        lambda: {"count": 0, "tokens_in": 0, "tokens_out": 0, "latency_ms": 0.0, "cost_usd": 0.0}
    )
    for qid in query_ids:
        for r in by_query[qid]:
            t = by_type[r["call_type"]]
            t["count"] += 1
            t["tokens_in"] += r["tokens_in"]
            t["tokens_out"] += r["tokens_out"]
            t["latency_ms"] += r["latency_ms"]
            t["cost_usd"] += r["cost_usd"]

    for t in by_type.values():
        t["avg_latency_ms"] = round(t["latency_ms"] / t["count"], 1)
        t["avg_cost_usd"] = round(t["cost_usd"] / t["count"], 8)

    return {
        "queries_analyzed": n,
        "total_calls": sum(query_call_counts),
        "per_query_avg": {
            "tokens_in": round(avg_tokens_in, 1),
            "tokens_out": round(avg_tokens_out, 1),
            "latency_ms": round(avg_latency, 1),
            "cost_usd": round(avg_cost, 8),
            "llm_calls": round(sum(query_call_counts) / n, 1),
        },
        "cost_per_1000_users_monthly_usd": {
            "5_queries_per_day": round(avg_cost * 5 * 30 * 1000, 2),
            "10_queries_per_day": round(avg_cost * 10 * 30 * 1000, 2),
            "20_queries_per_day": round(avg_cost * 20 * 30 * 1000, 2),
        },
        "by_call_type": {k: dict(v) for k, v in sorted(by_type.items())},
    }


def print_report(stats: dict) -> None:
    if "error" in stats:
        print(stats["error"])
        return

    pq = stats["per_query_avg"]
    print(f"=== Cost/Latency Analysis ({stats['queries_analyzed']} queries, {stats['total_calls']} total calls) ===\n")

    print("Per-query averages:")
    print(f"  Tokens in:     {pq['tokens_in']:,.0f}")
    print(f"  Tokens out:    {pq['tokens_out']:,.0f}")
    print(f"  LLM calls:     {pq['llm_calls']:.1f}")
    print(f"  Latency:       {pq['latency_ms']:,.0f} ms ({pq['latency_ms']/1000:.1f}s)")
    print(f"  Cost:          ${pq['cost_usd']:.6f}")

    print("\nCost per 1,000 users/month (at published rates):")
    for label, cost in stats["cost_per_1000_users_monthly_usd"].items():
        print(f"  {label.replace('_', ' ')}: ${cost:,.2f}")

    print("\nBreakdown by call type:")
    print(f"  {'Type':<16} {'Count':>6} {'Avg Lat':>10} {'Tokens In':>10} {'Tokens Out':>10} {'Total Cost':>12}")
    print(f"  {'─'*16} {'─'*6} {'─'*10} {'─'*10} {'─'*10} {'─'*12}")
    for name, t in sorted(stats["by_call_type"].items()):
        print(
            f"  {name:<16} {t['count']:>6} {t['avg_latency_ms']:>8.0f}ms "
            f"{t['tokens_in']:>10,} {t['tokens_out']:>10,} ${t['cost_usd']:>10.6f}"
        )


def main():
    parser = argparse.ArgumentParser(description="Analyze telemetry cost/latency data")
    parser.add_argument("--last", type=int, help="Only analyze the last N queries")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    records = load_records()
    stats = analyze(records, last_n=args.last)

    if args.json:
        print(json.dumps(stats, indent=2))
    else:
        print_report(stats)


if __name__ == "__main__":
    main()
