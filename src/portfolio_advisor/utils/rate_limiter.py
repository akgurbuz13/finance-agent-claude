"""Token budget enforcement and API rate limiting."""

from __future__ import annotations

import asyncio
import logging
import time

logger = logging.getLogger(__name__)


class AsyncTokenBucketRateLimiter:
    """Async token-bucket rate limiter for external API providers.

    Allows up to `max_tokens` calls, refilling at `refill_rate` tokens/second.
    Callers await `acquire()` which blocks when the bucket is empty.
    """

    def __init__(self, name: str, max_tokens: int, refill_rate: float):
        self.name = name
        self.max_tokens = max_tokens
        self.refill_rate = refill_rate  # tokens per second

        self._tokens = float(max_tokens)
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self.max_tokens, self._tokens + elapsed * self.refill_rate)
        self._last_refill = now

    async def acquire(self) -> None:
        """Wait until a token is available, then consume one."""
        while True:
            async with self._lock:
                self._refill()
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                # Calculate wait time until next token
                wait = (1.0 - self._tokens) / self.refill_rate if self.refill_rate > 0 else 1.0

            await asyncio.sleep(wait)

    @property
    def available(self) -> float:
        """Approximate available tokens (may be slightly stale, safe for concurrent reads)."""
        elapsed = time.monotonic() - self._last_refill
        return min(self.max_tokens, self._tokens + elapsed * self.refill_rate)

    def status(self) -> dict:
        return {
            "name": self.name,
            "available_tokens": round(self.available, 1),
            "max_tokens": self.max_tokens,
            "refill_rate_per_sec": self.refill_rate,
        }


class TokenBudgetEnforcer:
    """Tracks token usage and enforces daily budget limits."""

    def __init__(self, daily_budget: int = 100_000):
        self.daily_budget = daily_budget
        self._used_today = 0
        self._reset_date: str = ""

    def _check_reset(self) -> None:
        today = time.strftime("%Y-%m-%d")
        if today != self._reset_date:
            self._used_today = 0
            self._reset_date = today

    def record_usage(self, tokens: int) -> None:
        self._check_reset()
        self._used_today += tokens

    def can_proceed(self) -> bool:
        self._check_reset()
        return self._used_today < self.daily_budget

    @property
    def remaining(self) -> int:
        self._check_reset()
        return max(0, self.daily_budget - self._used_today)


class WebSearchRateLimiter:
    """Rate limiter for web search API calls."""

    def __init__(self, max_daily: int = 20, min_interval_seconds: float = 2.0):
        self.max_daily = max_daily
        self.min_interval = min_interval_seconds
        self._count_today = 0
        self._reset_date: str = ""
        self._last_call: float = 0.0

    def _check_reset(self) -> None:
        today = time.strftime("%Y-%m-%d")
        if today != self._reset_date:
            self._count_today = 0
            self._reset_date = today

    async def acquire(self) -> bool:
        """Wait for rate limit and return True if within budget, False if exhausted."""
        self._check_reset()
        if self._count_today >= self.max_daily:
            logger.warning(f"Web search daily limit reached ({self.max_daily})")
            return False

        # Enforce minimum interval
        elapsed = time.monotonic() - self._last_call
        if elapsed < self.min_interval:
            await asyncio.sleep(self.min_interval - elapsed)

        self._count_today += 1
        self._last_call = time.monotonic()
        return True

    @property
    def remaining(self) -> int:
        self._check_reset()
        return max(0, self.max_daily - self._count_today)
