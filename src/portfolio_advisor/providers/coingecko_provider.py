"""CoinGecko API wrapper with rate limiting."""

from __future__ import annotations

import logging

from portfolio_advisor.utils.circuit_breaker import CircuitBreaker
from portfolio_advisor.utils.rate_limiter import AsyncTokenBucketRateLimiter

logger = logging.getLogger(__name__)


class CoinGeckoProvider:
    """CoinGecko API with rate limiting (10 calls/min free tier)."""

    def __init__(self):
        self.rate_limiter = AsyncTokenBucketRateLimiter(
            "coingecko", max_tokens=10, refill_rate=10.0 / 60.0
        )
        self.circuit = CircuitBreaker("coingecko", failure_threshold=5, recovery_timeout=300)

    async def acquire_rate_limit(self) -> None:
        """Acquire a rate limit token before making a CoinGecko call."""
        await self.rate_limiter.acquire()

    def status(self) -> dict:
        return {
            "provider": "coingecko",
            "rate_limiter": self.rate_limiter.status(),
            "circuit_breaker": self.circuit.status(),
        }
