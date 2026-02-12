"""Alpha Vantage API provider for fundamentals and OHLCV fallback."""

from __future__ import annotations

import asyncio
import itertools
import logging
import time

import httpx

from portfolio_advisor.utils.circuit_breaker import CircuitBreaker
from portfolio_advisor.utils.rate_limiter import AsyncTokenBucketRateLimiter
from portfolio_advisor.utils.retry import retry_with_backoff

logger = logging.getLogger(__name__)

# Per-key limits (Alpha Vantage free tier)
_PER_KEY_RATE_PER_MIN = 5
_PER_KEY_DAILY_LIMIT = 25


class AlphaVantageProvider:
    """Alpha Vantage REST API provider with round-robin key rotation.

    Rate limit: 5 calls/min and 25 calls/day per key.
    With N keys: N × 5 calls/min and N × 25 calls/day.
    Used primarily as fallback for fundamentals.
    """

    BASE_URL = "https://www.alphavantage.co/query"

    def __init__(self, api_keys: list[str]):
        if not api_keys:
            raise ValueError("AlphaVantageProvider requires at least one API key")

        self._api_keys = api_keys
        self._key_cycle = itertools.cycle(range(len(api_keys)))
        self._total_keys = len(api_keys)

        # Per-key rate limiters
        self._rate_limiters = [
            AsyncTokenBucketRateLimiter(
                f"alpha_vantage_key_{i}",
                max_tokens=_PER_KEY_RATE_PER_MIN,
                refill_rate=_PER_KEY_RATE_PER_MIN / 60.0,
            )
            for i in range(len(api_keys))
        ]

        # Per-key daily call counters with date-based reset
        self._daily_calls = [0] * len(api_keys)
        self._reset_date: str = time.strftime("%Y-%m-%d")

        # Shared circuit breaker
        self.circuit = CircuitBreaker("alpha_vantage", failure_threshold=5, recovery_timeout=300)

        # Lock for round-robin cycle + daily counter access (not async-safe without it)
        self._state_lock = asyncio.Lock()

        # One HTTP client (shared — key passed as query param, not header)
        self._client: httpx.AsyncClient | None = None
        self._client_lock = asyncio.Lock()

        logger.info(
            f"Alpha Vantage provider initialized with {self._total_keys} keys "
            f"({self._total_keys * _PER_KEY_DAILY_LIMIT} calls/day effective)"
        )

    def _check_daily_reset_unlocked(self) -> None:
        """Reset daily counters at midnight. Caller must hold _state_lock."""
        today = time.strftime("%Y-%m-%d")
        if today != self._reset_date:
            self._daily_calls = [0] * self._total_keys
            self._reset_date = today
            logger.info("Alpha Vantage daily call counters reset")

    async def _next_key_index(self) -> int | None:
        """Round-robin to next available key (async-safe, skip exhausted keys)."""
        async with self._state_lock:
            self._check_daily_reset_unlocked()
            # Try each key once to find one that isn't exhausted
            for _ in range(self._total_keys):
                idx = next(self._key_cycle)
                if self._daily_calls[idx] < _PER_KEY_DAILY_LIMIT:
                    return idx
            return None  # All keys exhausted

    async def _get_client(self) -> httpx.AsyncClient:
        async with self._client_lock:
            if self._client is None or self._client.is_closed:
                self._client = httpx.AsyncClient(timeout=15)
            return self._client

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def _request(self, params: dict) -> dict | None:
        """Make a rate-limited request using round-robin key rotation."""
        key_idx = await self._next_key_index()
        if key_idx is None:
            logger.warning("Alpha Vantage: all keys exhausted for today")
            return None

        # Atomically increment daily counter
        async with self._state_lock:
            self._daily_calls[key_idx] += 1

        params["apikey"] = self._api_keys[key_idx]

        async def _do():
            await self._rate_limiters[key_idx].acquire()
            async with self.circuit:
                client = await self._get_client()
                resp = await client.get(self.BASE_URL, params=params)
                resp.raise_for_status()
                data = resp.json()
                # Alpha Vantage returns error messages in JSON
                if "Error Message" in data or "Note" in data:
                    logger.warning(
                        f"Alpha Vantage error: "
                        f"{data.get('Error Message', data.get('Note'))}"
                    )
                    return None
                return data

        try:
            return await retry_with_backoff(
                _do, operation_name="alpha_vantage"
            )
        except Exception as e:
            logger.warning(f"Alpha Vantage request failed: {e}")
            return None

    async def fetch_fundamentals(self, ticker: str) -> dict | None:
        """Fetch company overview (fundamentals) for a ticker."""
        data = await self._request({"function": "OVERVIEW", "symbol": ticker})
        if not data or "Symbol" not in data:
            return None
        return {
            "ticker": ticker,
            "pe_ratio": _safe_float(data.get("PERatio")),
            "forward_pe": _safe_float(data.get("ForwardPE")),
            "pb_ratio": _safe_float(data.get("PriceToBookRatio")),
            "ps_ratio": _safe_float(data.get("PriceToSalesRatioTTM")),
            "ev_ebitda": _safe_float(data.get("EVToEBITDA")),
            "roe": _safe_float(data.get("ReturnOnEquityTTM")),
            "roa": _safe_float(data.get("ReturnOnAssetsTTM")),
            "profit_margin": _safe_float(data.get("ProfitMargin")),
            "operating_margin": _safe_float(data.get("OperatingMarginTTM")),
            "debt_to_equity": _safe_float(data.get("DebtToEquity")),
            "current_ratio": _safe_float(data.get("CurrentRatio")),
            "revenue_growth_yoy": _safe_float(data.get("QuarterlyRevenueGrowthYOY")),
            "earnings_growth_yoy": _safe_float(data.get("QuarterlyEarningsGrowthYOY")),
            "dividend_yield": _safe_float(data.get("DividendYield")),
            "market_cap": _safe_float(data.get("MarketCapitalization")),
            "sector": data.get("Sector", ""),
            "industry": data.get("Industry", ""),
            "analyst_target_price": _safe_float(data.get("AnalystTargetPrice")),
            "analyst_rating": data.get("AnalystRatingStrongBuy", ""),
            "source": "alpha_vantage",
        }

    async def fetch_analyst_ratings(self, ticker: str) -> list[dict]:
        """Alpha Vantage doesn't provide granular analyst ratings. Returns empty list."""
        return []

    def status(self) -> dict:
        # Read stale counters without lock (safe for status reporting)
        total_used = sum(self._daily_calls)
        total_limit = self._total_keys * _PER_KEY_DAILY_LIMIT
        return {
            "provider": "alpha_vantage",
            "total_keys": self._total_keys,
            "daily_calls_total": total_used,
            "daily_limit_total": total_limit,
            "per_key_calls": list(self._daily_calls),
            "rate_limiters": [rl.status() for rl in self._rate_limiters],
            "circuit_breaker": self.circuit.status(),
        }


def _safe_float(val) -> float | None:
    if val is None or val == "None" or val == "-":
        return None
    try:
        f = float(val)
        return f if not (f != f) else None
    except (ValueError, TypeError):
        return None
