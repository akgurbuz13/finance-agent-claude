"""Token budget enforcement and API rate limiting."""

from __future__ import annotations

import asyncio
import logging
import time

logger = logging.getLogger(__name__)


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
