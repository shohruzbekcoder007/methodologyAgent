"""Token bucket behaviour — the per-user request ceiling."""

from __future__ import annotations

import pytest

from app.rate_limit import RateLimiter


@pytest.fixture
def clock(monkeypatch):
    """A monotonic clock the test advances by hand."""
    now = [1000.0]

    def fake_monotonic() -> float:
        return now[0]

    monkeypatch.setattr("app.rate_limit.time.monotonic", fake_monotonic)

    def advance(seconds: float) -> None:
        now[0] += seconds

    return advance


def test_disabled_when_rate_is_zero():
    limiter = RateLimiter(rate_per_minute=0, burst=5)
    assert not limiter.enabled
    for _ in range(100):
        allowed, retry = limiter.check("someone")
        assert allowed and retry == 0.0


def test_burst_is_allowed_then_refused(clock):
    limiter = RateLimiter(rate_per_minute=60, burst=3)
    assert [limiter.check("u")[0] for _ in range(3)] == [True, True, True]

    allowed, retry_after = limiter.check("u")
    assert not allowed
    assert retry_after > 0


def test_tokens_refill_over_time(clock):
    limiter = RateLimiter(rate_per_minute=60, burst=2)  # 1 token/second
    limiter.check("u")
    limiter.check("u")
    assert limiter.check("u")[0] is False

    clock(1.0)
    assert limiter.check("u")[0] is True, "one second should buy one token"


def test_refill_is_capped_at_burst(clock):
    limiter = RateLimiter(rate_per_minute=60, burst=2)
    limiter.check("u")
    clock(3600.0)  # idle for an hour

    assert [limiter.check("u")[0] for _ in range(2)] == [True, True]
    assert limiter.check("u")[0] is False, "bucket must not exceed burst"


def test_keys_do_not_share_a_bucket(clock):
    limiter = RateLimiter(rate_per_minute=60, burst=1)
    assert limiter.check("shohruz")[0] is True
    assert limiter.check("shohruz")[0] is False
    # One user exhausting their quota must not block anyone else.
    assert limiter.check("aziza")[0] is True


def test_retry_after_shrinks_as_the_bucket_refills(clock):
    limiter = RateLimiter(rate_per_minute=60, burst=1)
    limiter.check("u")
    _, first = limiter.check("u")
    clock(0.5)
    _, second = limiter.check("u")
    assert second < first


def test_reset_clears_every_bucket(clock):
    limiter = RateLimiter(rate_per_minute=60, burst=1)
    limiter.check("u")
    assert limiter.check("u")[0] is False
    limiter.reset()
    assert limiter.check("u")[0] is True
