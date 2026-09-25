"""Gemini API key rotation on daily quota exhaustion.

Maintains a pool of keys from GEMINI_API_KEYS and automatically rotates
to the next key when DailyQuotaExhausted is raised. Thread-safe via a
module-level index.
"""

import threading

from research_copilot.config import settings
from research_copilot.retry import DailyQuotaExhausted

_lock = threading.Lock()
_current_index = 0
_exhausted: set[int] = set()


def get_api_key() -> str:
    pool = settings.gemini_key_pool
    if not pool:
        raise ValueError("No Gemini API keys configured")
    with _lock:
        if len(_exhausted) >= len(pool):
            raise DailyQuotaExhausted(
                f"All {len(pool)} Gemini API keys have hit daily quota"
            )
        return pool[_current_index % len(pool)]


def rotate_on_quota_error() -> str | None:
    pool = settings.gemini_key_pool
    if not pool:
        return None
    with _lock:
        global _current_index
        _exhausted.add(_current_index % len(pool))
        remaining = len(pool) - len(_exhausted)
        if remaining == 0:
            return None
        for i in range(len(pool)):
            candidate = (_current_index + 1 + i) % len(pool)
            if candidate not in _exhausted:
                _current_index = candidate
                print(f"[key_rotation] Rotated to key {candidate + 1}/{len(pool)} "
                      f"({remaining} remaining)")
                return pool[candidate]
    return None


def reset():
    with _lock:
        global _current_index
        _current_index = 0
        _exhausted.clear()
