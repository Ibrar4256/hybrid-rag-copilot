from research_copilot import synthesis
from research_copilot.agent import Evidence


def _evidence(*chunk_ids):
    return {cid: Evidence(chunk_id=cid, source="doc.md", text=f"text for {cid}") for cid in chunk_ids}


def test_verify_supports_claim_when_cited_chunk_is_entailed(monkeypatch):
    monkeypatch.setattr(synthesis, "_check_entailment", lambda pairs, key: [True])
    evidence = _evidence("doc.md#0")
    parsed = {"claims": [{"text": "fact X", "source_chunk_ids": ["doc.md#0"]}]}

    claims = synthesis._verify(parsed, evidence, api_key="fake")

    assert len(claims) == 1
    assert claims[0].supported is True
    assert claims[0].source_chunk_ids == ["doc.md#0"]


def test_verify_rejects_hallucinated_chunk_id_not_in_evidence(monkeypatch):
    # a chunk ID the model invented (not actually retrieved) must be filtered
    # out before it ever reaches the entailment check
    calls = []
    monkeypatch.setattr(
        synthesis, "_check_entailment", lambda pairs, key: calls.append(pairs) or [True] * len(pairs)
    )
    evidence = _evidence("doc.md#0")
    parsed = {"claims": [{"text": "fact X", "source_chunk_ids": ["doc.md#99-does-not-exist"]}]}

    claims = synthesis._verify(parsed, evidence, api_key="fake")

    assert claims[0].supported is False
    assert claims[0].source_chunk_ids == []
    assert calls == [[]]  # no entailment call made for a nonexistent chunk


def test_verify_rejects_claim_when_entailment_check_fails(monkeypatch):
    # chunk ID is real (was retrieved) but the entailment check says the text
    # doesn't actually support the claim — must NOT count as supported
    monkeypatch.setattr(synthesis, "_check_entailment", lambda pairs, key: [False])
    evidence = _evidence("doc.md#0")
    parsed = {"claims": [{"text": "fact X", "source_chunk_ids": ["doc.md#0"]}]}

    claims = synthesis._verify(parsed, evidence, api_key="fake")

    assert claims[0].supported is False


def test_verify_handles_malformed_bare_string_claim(monkeypatch):
    # observed in practice: the model sometimes returns a claim as a plain
    # string instead of the requested {"text", "source_chunk_ids"} object
    monkeypatch.setattr(synthesis, "_check_entailment", lambda pairs, key: [])
    evidence = _evidence("doc.md#0")
    parsed = {"claims": ["just a bare string claim"]}

    claims = synthesis._verify(parsed, evidence, api_key="fake")

    assert len(claims) == 1
    assert claims[0].text == "just a bare string claim"
    assert claims[0].supported is False


def test_verify_keeps_only_entailed_chunks_among_multiple_citations(monkeypatch):
    # a claim citing two real chunks where only one is actually entailed
    # should end up supported, but only by the entailed chunk
    monkeypatch.setattr(synthesis, "_check_entailment", lambda pairs, key: [True, False])
    evidence = _evidence("doc.md#0", "doc.md#1")
    parsed = {
        "claims": [{"text": "fact X", "source_chunk_ids": ["doc.md#0", "doc.md#1"]}]
    }

    claims = synthesis._verify(parsed, evidence, api_key="fake")

    assert claims[0].supported is True
    assert claims[0].source_chunk_ids == ["doc.md#0"]


def test_verify_batches_entailment_checks_across_all_claims_in_one_call(monkeypatch):
    # _verify should collect every (claim, chunk) pair across ALL claims and
    # call _check_entailment exactly once per question, not once per claim
    call_count = {"n": 0}

    def fake_check(pairs, key):
        call_count["n"] += 1
        return [True] * len(pairs)

    monkeypatch.setattr(synthesis, "_check_entailment", fake_check)
    evidence = _evidence("doc.md#0", "doc.md#1")
    parsed = {
        "claims": [
            {"text": "fact A", "source_chunk_ids": ["doc.md#0"]},
            {"text": "fact B", "source_chunk_ids": ["doc.md#1"]},
        ]
    }

    claims = synthesis._verify(parsed, evidence, api_key="fake")

    assert call_count["n"] == 1
    assert all(c.supported for c in claims)


def test_render_marks_unsupported_claims_distinctly():
    claims = [
        synthesis.Claim(text="supported fact", source_chunk_ids=["doc.md#0"], supported=True),
        synthesis.Claim(text="unsupported fact", source_chunk_ids=[], supported=False),
    ]

    rendered = synthesis.render(claims)

    assert "[doc.md#0]" in rendered
    assert "[UNSUPPORTED — no citation]" in rendered
