"""
Per-user token-bucket rate limiting.

In-process and per-worker: with `API_WORKERS>1` each worker keeps its own
buckets, so the effective ceiling is `workers x rate`. That is the intended
trade-off for a starter — it needs no Redis, and the point here is to stop one
user monopolising the single upstream model, not to meter billing.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field


@dataclass
class _Bucket:
    tokens: float
    updated: float


@dataclass
class RateLimiter:
    """Token bucket per key: `rate` requests/minute with a burst allowance."""

    rate_per_minute: float
    burst: int
    _buckets: dict[str, _Bucket] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def enabled(self) -> bool:
        return self.rate_per_minute > 0

    def check(self, key: str) -> tuple[bool, float]:
        """Consume one token for `key`.

        Returns `(allowed, retry_after_seconds)`; `retry_after` is 0 when
        allowed.
        """
        if not self.enabled:
            return True, 0.0

        per_second = self.rate_per_minute / 60.0
        capacity = float(max(1, self.burst))
        now = time.monotonic()

        with self._lock:
            bucket = self._buckets.get(key)
            if bucket is None:
                # A new key starts full, minus the request being served.
                self._buckets[key] = _Bucket(tokens=capacity - 1.0, updated=now)
                return True, 0.0

            elapsed = max(0.0, now - bucket.updated)
            bucket.tokens = min(capacity, bucket.tokens + elapsed * per_second)
            bucket.updated = now

            if bucket.tokens >= 1.0:
                bucket.tokens -= 1.0
                return True, 0.0

            missing = 1.0 - bucket.tokens
            return False, round(missing / per_second, 2) if per_second else 60.0

    def reset(self) -> None:
        with self._lock:
            self._buckets.clear()
