import sys
import time

from google.api_core.exceptions import DeadlineExceeded, ResourceExhausted

_MAX_RETRIES = 5

_DAILY_QUOTA_MARKER = "PerDay"

_RETRYABLE_EXCEPTIONS = (ResourceExhausted, DeadlineExceeded)


class DailyQuotaExhausted(Exception):
    """Raised immediately (no retry) when Gemini's daily free-tier quota is hit."""


def with_backoff(fn, *args, label: str = "", **kwargs):
    """Retry on Gemini free-tier per-minute rate limits and transient 504s with
    exponential backoff; fail fast on the daily quota instead of retrying
    uselessly. Each retry is logged to stderr with timing and error details."""
    tag = f"[retry:{label}]" if label else "[retry]"
    for attempt in range(_MAX_RETRIES):
        try:
            return fn(*args, **kwargs)
        except ResourceExhausted as e:
            if _DAILY_QUOTA_MARKER in str(e):
                raise DailyQuotaExhausted(str(e)) from e
            if attempt == _MAX_RETRIES - 1:
                print(f"{tag} exhausted {_MAX_RETRIES} retries on ResourceExhausted, raising", file=sys.stderr)
                raise
            sleep_s = 2**attempt * 5  # 5s, 10s, 20s, 40s
            print(f"{tag} attempt {attempt+1}/{_MAX_RETRIES} hit ResourceExhausted (rate limit), sleeping {sleep_s}s", file=sys.stderr)
            time.sleep(sleep_s)
        except DeadlineExceeded:
            if attempt == _MAX_RETRIES - 1:
                print(f"{tag} exhausted {_MAX_RETRIES} retries on DeadlineExceeded, raising", file=sys.stderr)
                raise
            sleep_s = 2**attempt * 3  # 3s, 6s, 12s, 24s
            print(f"{tag} attempt {attempt+1}/{_MAX_RETRIES} hit DeadlineExceeded (504), sleeping {sleep_s}s", file=sys.stderr)
            time.sleep(sleep_s)
    raise RuntimeError("unreachable")
