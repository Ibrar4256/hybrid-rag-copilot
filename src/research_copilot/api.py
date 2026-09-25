"""FastAPI backend (ADR-008): wraps the existing agent loop + citation
synthesis pipeline as a JSON API, and serves the static walking-skeleton
frontend. No streaming yet — a full response is returned once synthesis
completes (see ADR-008's Consequences for what's deferred).

Run: uvicorn research_copilot.api:app --reload
"""

from collections import defaultdict
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from research_copilot.answer import answer_question_structured
from research_copilot.config import settings
from research_copilot.telemetry import load_log

app = FastAPI(title="Research Copilot API")

WEB_DIR = Path(__file__).parent.parent.parent / "web"

DEFAULT_COLLECTION = "sec_filings_banks_chunk200"


class AskRequest(BaseModel):
    question: str
    collection: str | None = None


class ClaimOut(BaseModel):
    text: str
    source_chunk_ids: list[str]
    supported: bool
    unable_to_answer: bool = False
    reason: str = ""


class EvidenceOut(BaseModel):
    chunk_id: str
    source: str
    text: str


class CallOut(BaseModel):
    provider: str
    model: str
    call_type: str
    tokens_in: int
    tokens_out: int
    latency_ms: float
    cost_usd: float


class TelemetryOut(BaseModel):
    query_id: str
    total_calls: int
    total_tokens_in: int
    total_tokens_out: int
    total_latency_ms: float
    total_cost_usd: float
    calls: list[CallOut]


class AskResponse(BaseModel):
    claims: list[ClaimOut]
    evidence: list[EvidenceOut]
    telemetry: TelemetryOut | None = None


@app.post("/api/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    if not req.question.strip():
        raise HTTPException(400, "question must not be empty")
    if not settings.gemini_key_pool:
        raise HTTPException(500, "GEMINI_API_KEY or GEMINI_API_KEYS not configured — see .env")

    try:
        claims, evidence, stats = answer_question_structured(
            req.question, req.collection or DEFAULT_COLLECTION
        )
    except Exception as e:
        raise HTTPException(502, f"pipeline error: {e!r}") from e

    telemetry = None
    if stats:
        telemetry = TelemetryOut(
            query_id=stats["query_id"],
            total_calls=stats["total_calls"],
            total_tokens_in=stats["total_tokens_in"],
            total_tokens_out=stats["total_tokens_out"],
            total_latency_ms=stats["total_latency_ms"],
            total_cost_usd=stats["total_cost_usd"],
            calls=[
                CallOut(
                    provider=c["provider"], model=c["model"],
                    call_type=c["call_type"], tokens_in=c["tokens_in"],
                    tokens_out=c["tokens_out"], latency_ms=c["latency_ms"],
                    cost_usd=c["cost_usd"],
                )
                for c in stats["calls"]
            ],
        )

    return AskResponse(
        claims=[
            ClaimOut(text=c.text, source_chunk_ids=c.source_chunk_ids,
                     supported=c.supported, unable_to_answer=c.unable_to_answer,
                     reason=getattr(c, "reason", ""))
            for c in claims
        ],
        evidence=[
            EvidenceOut(chunk_id=e.chunk_id, source=e.source, text=e.text)
            for e in evidence.values()
        ],
        telemetry=telemetry,
    )


@app.get("/api/telemetry/summary")
def telemetry_summary():
    """Aggregate telemetry across all logged queries."""
    records = load_log()
    if not records:
        return {"total_queries": 0, "total_calls": 0}

    by_query: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        qid = r.get("query_id") or "unknown"
        by_query[qid].append(r)

    query_costs = []
    query_latencies = []
    query_tokens = []
    for qid, calls in by_query.items():
        query_costs.append(sum(c["cost_usd"] for c in calls))
        query_latencies.append(sum(c["latency_ms"] for c in calls))
        query_tokens.append(sum(c["tokens_in"] + c["tokens_out"] for c in calls))

    n = len(query_costs)
    avg_cost = sum(query_costs) / n
    avg_latency = sum(query_latencies) / n
    avg_tokens = sum(query_tokens) / n

    by_type: dict[str, dict] = defaultdict(lambda: {"count": 0, "tokens_in": 0, "tokens_out": 0, "latency_ms": 0.0, "cost_usd": 0.0})
    for r in records:
        t = by_type[r["call_type"]]
        t["count"] += 1
        t["tokens_in"] += r["tokens_in"]
        t["tokens_out"] += r["tokens_out"]
        t["latency_ms"] += r["latency_ms"]
        t["cost_usd"] += r["cost_usd"]

    return {
        "total_queries": n,
        "total_calls": len(records),
        "avg_cost_per_query_usd": round(avg_cost, 8),
        "avg_latency_per_query_ms": round(avg_latency, 1),
        "avg_tokens_per_query": round(avg_tokens, 1),
        "cost_per_1000_users_monthly": {
            "at_5_queries_per_day": round(avg_cost * 5 * 30 * 1000, 2),
            "at_10_queries_per_day": round(avg_cost * 10 * 30 * 1000, 2),
            "at_20_queries_per_day": round(avg_cost * 20 * 30 * 1000, 2),
        },
        "by_call_type": dict(by_type),
    }


@app.get("/")
def index() -> FileResponse:
    return FileResponse(
        WEB_DIR / "index.html",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
