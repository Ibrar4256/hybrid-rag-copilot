"""Lightweight cost/latency logger (ADR-009, Option A).

One function, one JSONL sink. Every LLM and local-model call logs here;
Project 5 can ingest the same file later without this module changing.
"""

import json
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

_LOG_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "telemetry"
_LOG_FILE = _LOG_DIR / "calls.jsonl"

# Published per-1M-token pricing (USD). Free-tier usage costs $0 in
# practice, but we log at published rates so the "cost per 1,000 users"
# calculation reflects what a paid deployment would actually cost.
_COST_PER_1M_TOKENS: dict[str, dict[str, float]] = {
    "models/gemini-flash-lite-latest": {"input": 0.075, "output": 0.30},
    "models/gemini-2.0-flash-lite": {"input": 0.075, "output": 0.30},
    "openai/gpt-oss-120b": {"input": 0.0, "output": 0.0},
    "nex-agi/nex-n2.5-pro:free": {"input": 0.0, "output": 0.0},
    "Meta-Llama-3.3-70B-Instruct": {"input": 0.0, "output": 0.0},
    "llama-3.3-70b": {"input": 0.0, "output": 0.0},
}

_current_query_id: str | None = None


def start_query() -> str:
    """Begin a new query scope — all subsequent log_call()s include this ID
    until the next start_query() or end_query()."""
    global _current_query_id
    _current_query_id = uuid.uuid4().hex[:12]
    return _current_query_id


def end_query() -> None:
    global _current_query_id
    _current_query_id = None


def _compute_cost(model: str, tokens_in: int, tokens_out: int) -> float:
    rates = _COST_PER_1M_TOKENS.get(model, {"input": 0.0, "output": 0.0})
    return (tokens_in * rates["input"] + tokens_out * rates["output"]) / 1_000_000


def log_call(
    provider: str,
    model: str,
    call_type: str,
    tokens_in: int,
    tokens_out: int,
    latency_ms: float,
    cost_usd: float | None = None,
):
    if cost_usd is None:
        cost_usd = _compute_cost(model, tokens_in, tokens_out)

    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    record = {
        "ts": time.time(),
        "query_id": _current_query_id,
        "provider": provider,
        "model": model,
        "call_type": call_type,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "latency_ms": round(latency_ms, 1),
        "cost_usd": round(cost_usd, 8),
    }
    with open(_LOG_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")


def load_log() -> list[dict]:
    if not _LOG_FILE.exists():
        return []
    records = []
    with open(_LOG_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def query_stats(query_id: str) -> dict:
    """Aggregate stats for a single query by its ID."""
    calls = [r for r in load_log() if r.get("query_id") == query_id]
    if not calls:
        return {}
    return {
        "query_id": query_id,
        "total_calls": len(calls),
        "total_tokens_in": sum(c["tokens_in"] for c in calls),
        "total_tokens_out": sum(c["tokens_out"] for c in calls),
        "total_latency_ms": round(sum(c["latency_ms"] for c in calls), 1),
        "total_cost_usd": round(sum(c["cost_usd"] for c in calls), 8),
        "calls": calls,
    }


@contextmanager
def timed():
    """Context manager yielding an object whose .ms attribute holds elapsed time."""
    class _Timer:
        ms: float = 0.0
    t = _Timer()
    t0 = time.monotonic()
    try:
        yield t
    finally:
        t.ms = (time.monotonic() - t0) * 1000
