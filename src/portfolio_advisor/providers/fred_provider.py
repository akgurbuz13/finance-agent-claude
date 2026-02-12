"""FRED API provider for economic data."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

import httpx

from portfolio_advisor.utils.circuit_breaker import CircuitBreaker
from portfolio_advisor.utils.rate_limiter import AsyncTokenBucketRateLimiter
from portfolio_advisor.utils.retry import retry_with_backoff

logger = logging.getLogger(__name__)


class FREDProvider:
    """FRED (Federal Reserve Economic Data) API provider.

    Rate limit: 120 requests/minute.
    """

    BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.rate_limiter = AsyncTokenBucketRateLimiter("fred", max_tokens=120, refill_rate=2.0)
        self.circuit = CircuitBreaker("fred", failure_threshold=5, recovery_timeout=300)
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=15)
        return self._client

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def fetch_series(
        self,
        series_id: str,
        start: str | None = None,
        end: str | None = None,
        limit: int = 500,
    ) -> list[dict] | None:
        """Fetch observations for a FRED series."""
        if not self.api_key:
            return None

        if end is None:
            end = datetime.now(tz=None).strftime("%Y-%m-%d")
        if start is None:
            start = (datetime.now(tz=None) - timedelta(days=365)).strftime("%Y-%m-%d")

        params = {
            "series_id": series_id,
            "api_key": self.api_key,
            "file_type": "json",
            "observation_start": start,
            "observation_end": end,
            "sort_order": "desc",
            "limit": limit,
        }

        async def _do_fetch():
            await self.rate_limiter.acquire()
            async with self.circuit:
                client = await self._get_client()
                resp = await client.get(self.BASE_URL, params=params)
                resp.raise_for_status()
                return resp.json()

        try:
            data = await retry_with_backoff(
                _do_fetch, operation_name=f"fred:{series_id}"
            )
            observations = []
            for obs in data.get("observations", []):
                val = obs.get("value", ".")
                if val != ".":
                    observations.append({"date": obs["date"], "value": float(val)})
            return observations
        except Exception as e:
            logger.warning(f"FRED fetch failed for {series_id}: {e}")
            return None

    async def fetch_treasury_yields(self) -> dict | None:
        """Fetch actual treasury yields: DGS2, DGS5, DGS10, DGS30."""
        series_map = {"2y": "DGS2", "5y": "DGS5", "10y": "DGS10", "30y": "DGS30"}
        yields = {}
        for label, series_id in series_map.items():
            obs = await self.fetch_series(series_id, limit=5)
            if obs:
                yields[label] = obs[0]["value"]
        return yields if yields else None

    async def fetch_vix(self) -> float | None:
        """Fetch CBOE VIX from FRED series VIXCLS."""
        obs = await self.fetch_series("VIXCLS", limit=5)
        if obs:
            return obs[0]["value"]
        return None

    async def fetch_credit_spread(self) -> float | None:
        """Fetch ICE BofA US HY OAS (BAMLH0A0HYM2) — actual credit spread in pct points."""
        obs = await self.fetch_series("BAMLH0A0HYM2", limit=5)
        if obs:
            return obs[0]["value"]
        return None

    def status(self) -> dict:
        return {
            "provider": "fred",
            "has_key": bool(self.api_key),
            "rate_limiter": self.rate_limiter.status(),
            "circuit_breaker": self.circuit.status(),
        }
