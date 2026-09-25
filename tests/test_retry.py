import pytest
from google.api_core.exceptions import DeadlineExceeded, ResourceExhausted

from research_copilot.retry import DailyQuotaExhausted, with_backoff


def test_with_backoff_returns_result_on_success():
    assert with_backoff(lambda: 42) == 42


def test_with_backoff_retries_transient_rate_limit_then_succeeds(monkeypatch):
    monkeypatch.setattr("research_copilot.retry.time.sleep", lambda s: None)
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise ResourceExhausted("rate limited")
        return "ok"

    assert with_backoff(flaky) == "ok"
    assert calls["n"] == 3


def test_with_backoff_raises_after_max_retries(monkeypatch):
    monkeypatch.setattr("research_copilot.retry.time.sleep", lambda s: None)

    def always_fails():
        raise ResourceExhausted("rate limited")

    with pytest.raises(ResourceExhausted):
        with_backoff(always_fails)


def test_with_backoff_retries_deadline_exceeded(monkeypatch):
    monkeypatch.setattr("research_copilot.retry.time.sleep", lambda s: None)
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 2:
            raise DeadlineExceeded("timeout")
        return "ok"

    assert with_backoff(flaky) == "ok"


def test_with_backoff_fails_fast_on_daily_quota_marker(monkeypatch):
    # the daily quota is a hard wall — must NOT sleep/retry, must raise
    # immediately as DailyQuotaExhausted, not the generic ResourceExhausted
    monkeypatch.setattr(
        "research_copilot.retry.time.sleep",
        lambda s: pytest.fail("should not sleep on daily quota exhaustion"),
    )

    def quota_exhausted():
        raise ResourceExhausted("Quota exceeded for quota metric X PerDay")

    with pytest.raises(DailyQuotaExhausted):
        with_backoff(quota_exhausted)


def test_with_backoff_passes_args_and_kwargs_through():
    def fn(a, b, c=None):
        return (a, b, c)

    assert with_backoff(fn, 1, 2, c=3) == (1, 2, 3)
