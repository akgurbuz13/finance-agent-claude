"""Massive API provider for market data, earnings, news, and fundamentals."""

from __future__ import annotations

import asyncio
import itertools
import logging

import httpx

from portfolio_advisor.utils.circuit_breaker import CircuitBreaker
from portfolio_advisor.utils.rate_limiter import AsyncTokenBucketRateLimiter
from portfolio_advisor.utils.retry import retry_with_backoff

logger = logging.getLogger(__name__)

# Per-key rate limit (Massive free tier: 5 calls/min per key)
_PER_KEY_RATE_LIMIT = 5


class MassiveProvider:
    """Massive REST API provider with round-robin key rotation.

    Covers: treasury yields, earnings (Benzinga), news with sentiment,
    fundamentals (ratios), short interest, dividends, analyst ratings,
    market movers.

    Rate limit: 5 calls/min per key. With N keys, effective rate = N × 5 calls/min.
    """

    BASE_URL = "https://api.massive.com"

    def __init__(self, api_keys: list[str], per_key_rate_limit: int = _PER_KEY_RATE_LIMIT):
        if not api_keys:
            raise ValueError("MassiveProvider requires at least one API key")

        self._api_keys = api_keys
        self._key_cycle = itertools.cycle(range(len(api_keys)))

        # Each key gets its own rate limiter to enforce per-key limits independently
        self._rate_limiters = [
            AsyncTokenBucketRateLimiter(
                f"massive_key_{i}", max_tokens=per_key_rate_limit,
                refill_rate=per_key_rate_limit / 60.0,
            )
            for i in range(len(api_keys))
        ]

        # Single shared circuit breaker (if the API itself is down, all keys fail)
        self.circuit = CircuitBreaker("massive", failure_threshold=5, recovery_timeout=300)

        # One HTTP client per key (each has its own Authorization header)
        self._clients: list[httpx.AsyncClient | None] = [None] * len(api_keys)
        self._client_locks = [asyncio.Lock() for _ in range(len(api_keys))]

        # Lock for round-robin cycle (next() is not async-safe)
        self._cycle_lock = asyncio.Lock()

        self._total_keys = len(api_keys)
        logger.info(
            f"Massive provider initialized with {self._total_keys} keys "
            f"({self._total_keys * per_key_rate_limit} calls/min effective)"
        )

    async def _next_key_index(self) -> int:
        """Round-robin to next key index (async-safe)."""
        async with self._cycle_lock:
            return next(self._key_cycle)

    async def _get_client(self, key_idx: int) -> httpx.AsyncClient:
        async with self._client_locks[key_idx]:
            if self._clients[key_idx] is None or self._clients[key_idx].is_closed:
                self._clients[key_idx] = httpx.AsyncClient(
                    timeout=20,
                    headers={"Authorization": f"Bearer {self._api_keys[key_idx]}"},
                )
            return self._clients[key_idx]

    async def close(self) -> None:
        for client in self._clients:
            if client and not client.is_closed:
                await client.aclose()

    async def _request(self, path: str, params: dict | None = None) -> dict | None:
        """Make a rate-limited, circuit-broken request using round-robin key rotation."""
        key_idx = await self._next_key_index()

        async def _do():
            await self._rate_limiters[key_idx].acquire()
            async with self.circuit:
                client = await self._get_client(key_idx)
                resp = await client.get(f"{self.BASE_URL}{path}", params=params or {})
                # 403 = subscription tier limitation, not a transient error
                if resp.status_code == 403:
                    logger.info(f"Massive API {path}: 403 (subscription tier)")
                    return None
                resp.raise_for_status()
                return resp.json()

        try:
            return await retry_with_backoff(_do, operation_name=f"massive:{path}")
        except Exception as e:
            logger.warning(f"Massive API request failed for {path}: {e}")
            return None

    # ── Economic Data ────────────────────────────────────────────────────────

    async def fetch_treasury_yields(self) -> dict | None:
        """Fetch treasury yields from /fed/v1/treasury-yields."""
        data = await self._request(
            "/fed/v1/treasury-yields",
            {"limit": 1, "sort": "date.desc"},
        )
        if not data:
            return None
        results = data.get("results", data.get("data", []))
        if not results:
            return None
        latest = results[0] if isinstance(results, list) else results
        yields = {}
        for field, label in [
            ("yield_2_year", "2y"), ("yield_5_year", "5y"),
            ("yield_10_year", "10y"), ("yield_30_year", "30y"),
        ]:
            val = latest.get(field)
            if val is not None:
                yields[label] = float(val)
        return yields if yields else None

    # ── Earnings ─────────────────────────────────────────────────────────────

    async def fetch_earnings(self, tickers: list[str]) -> list[dict]:
        """Fetch earnings from Benzinga via /benzinga/v1/earnings."""
        tickers_str = ",".join(tickers)
        data = await self._request(
            "/benzinga/v1/earnings",
            {"ticker.any_of": tickers_str, "limit": 50},
        )
        if not data:
            return []
        results = data.get("results", data.get("data", []))
        if not isinstance(results, list):
            return []
        entries = []
        for r in results:
            entry = {
                "ticker": r.get("ticker", r.get("symbol", "")),
                "earnings_date": r.get("date", r.get("earnings_date", "")),
                "earnings_time": r.get("time", r.get("earnings_time", "unknown")),
                "eps_estimate": _safe_float(r.get("eps_estimate")),
                "eps_actual": _safe_float(r.get("eps_actual")),
                "revenue_estimate": _safe_float(r.get("revenue_estimate")),
                "revenue_actual": _safe_float(r.get("revenue_actual")),
                "source": "massive_benzinga",
            }
            # Compute surprise
            if entry["eps_actual"] is not None and entry["eps_estimate"] is not None:
                est = entry["eps_estimate"]
                if est != 0:
                    entry["eps_surprise_pct"] = round(
                        ((entry["eps_actual"] - est) / abs(est)) * 100, 1
                    )
                entry["status"] = "reported"
            else:
                entry["status"] = "upcoming"
            entries.append(entry)
        return entries

    # ── News ─────────────────────────────────────────────────────────────────

    async def fetch_news(self, tickers: list[str], limit: int = 10) -> list[dict]:
        """Fetch news with sentiment from /v2/reference/news."""
        tickers_str = ",".join(tickers)
        # Single ticker: use `ticker`; multiple: use `ticker.any_of`
        ticker_param = "ticker" if len(tickers) == 1 else "ticker.any_of"
        data = await self._request(
            "/v2/reference/news",
            {ticker_param: tickers_str, "limit": limit, "order": "desc"},
        )
        if not data:
            return []
        results = data.get("results", data.get("data", []))
        if not isinstance(results, list):
            return []
        articles = []
        for r in results:
            article = {
                "title": r.get("title", ""),
                "published_utc": r.get("published_utc", r.get("published_at", "")),
                "source": r.get("publisher", {}).get("name", r.get("source", "")),
                "url": r.get("article_url", r.get("url", "")),
                "tickers": r.get("tickers", []),
                "description": r.get("description", "")[:200],
            }
            # Sentiment (Massive provides per-ticker sentiment via insights)
            insights = r.get("insights", [])
            if insights:
                sentiments = {}
                for ins in insights:
                    t = ins.get("ticker", "")
                    sentiments[t] = {
                        "sentiment": ins.get("sentiment", "neutral"),
                        "reasoning": ins.get("sentiment_reasoning", ""),
                    }
                article["sentiments"] = sentiments
            articles.append(article)
        return articles

    # ── Fundamentals ─────────────────────────────────────────────────────────

    async def fetch_fundamentals(self, ticker: str) -> dict | None:
        """Fetch fundamental ratios from /stocks/financials/v1/ratios."""
        data = await self._request(
            "/stocks/financials/v1/ratios",
            {"ticker": ticker, "limit": 1},
        )
        if not data:
            return None
        results = data.get("results", data.get("data", []))
        if not results:
            return None
        r = results[0] if isinstance(results, list) else results
        return {
            "ticker": ticker,
            "pe_ratio": _safe_float(r.get("pe_ratio", r.get("price_earnings_ratio"))),
            "forward_pe": _safe_float(r.get("forward_pe")),
            "pb_ratio": _safe_float(r.get("pb_ratio", r.get("price_book_ratio"))),
            "ps_ratio": _safe_float(r.get("ps_ratio", r.get("price_sales_ratio"))),
            "ev_ebitda": _safe_float(r.get("ev_ebitda", r.get("enterprise_value_ebitda"))),
            "roe": _safe_float(r.get("roe", r.get("return_on_equity"))),
            "roa": _safe_float(r.get("roa", r.get("return_on_assets"))),
            "profit_margin": _safe_float(r.get("profit_margin", r.get("net_profit_margin"))),
            "operating_margin": _safe_float(r.get("operating_margin")),
            "debt_to_equity": _safe_float(r.get("debt_to_equity")),
            "current_ratio": _safe_float(r.get("current_ratio")),
            "revenue_growth_yoy": _safe_float(r.get("revenue_growth")),
            "earnings_growth_yoy": _safe_float(r.get("earnings_growth")),
            "dividend_yield": _safe_float(r.get("dividend_yield")),
            "market_cap": _safe_float(r.get("market_cap", r.get("market_capitalization"))),
            "sector": r.get("sector", ""),
            "industry": r.get("industry", ""),
            "source": "massive",
        }

    # ── Short Interest ───────────────────────────────────────────────────────

    async def fetch_short_interest(self, tickers: list[str]) -> list[dict]:
        """Fetch short interest data from /stocks/v1/short-interest."""
        results = []
        for ticker in tickers:
            data = await self._request(
                "/stocks/v1/short-interest",
                {"ticker": ticker, "limit": 1, "order": "desc"},
            )
            if not data:
                continue
            entries = data.get("results", data.get("data", []))
            if entries:
                r = entries[0] if isinstance(entries, list) else entries
                results.append({
                    "ticker": ticker,
                    "short_interest": _safe_float(r.get("short_interest", r.get("shares_short"))),
                    "short_pct_float": _safe_float(
                        r.get("short_pct_float", r.get("short_percent_of_float"))
                    ),
                    "days_to_cover": _safe_float(r.get("days_to_cover")),
                    "settlement_date": r.get("settlement_date", r.get("date", "")),
                    "source": "massive",
                })
        return results

    # ── Dividends ────────────────────────────────────────────────────────────

    async def fetch_dividends(self, ticker: str) -> list[dict]:
        """Fetch dividend history from /v3/reference/dividends."""
        data = await self._request(
            "/v3/reference/dividends",
            {"ticker": ticker, "limit": 20, "order": "desc"},
        )
        if not data:
            return []
        results = data.get("results", data.get("data", []))
        if not isinstance(results, list):
            return []
        return [
            {
                "ticker": ticker,
                "ex_date": r.get("ex_dividend_date", r.get("ex_date", "")),
                "pay_date": r.get("pay_date", ""),
                "amount": _safe_float(r.get("cash_amount", r.get("amount"))),
                "frequency": r.get("frequency", ""),
                "type": r.get("dividend_type", "cash"),
                "source": "massive",
            }
            for r in results
        ]

    # ── Analyst Ratings ──────────────────────────────────────────────────────

    async def fetch_analyst_ratings(self, ticker: str) -> list[dict]:
        """Fetch analyst ratings from /benzinga/v1/ratings."""
        data = await self._request(
            "/benzinga/v1/ratings",
            {"ticker": ticker, "limit": 10},
        )
        if not data:
            return []
        results = data.get("results", data.get("data", []))
        if not isinstance(results, list):
            return []
        return [
            {
                "ticker": ticker,
                "date": r.get("date", ""),
                "firm": r.get("analyst", r.get("firm", "")),
                "action": r.get("action_type", r.get("action", "")),
                "rating_current": r.get("rating_current", r.get("rating", "")),
                "price_target": _safe_float(r.get("pt_current", r.get("price_target"))),
                "source": "massive_benzinga",
            }
            for r in results
        ]

    # ── Market Movers ────────────────────────────────────────────────────────

    async def fetch_market_movers(self) -> dict | None:
        """Fetch top market movers (gainers + losers) from snapshot API."""
        gainers_data = await self._request(
            "/v2/snapshot/locale/us/markets/stocks/gainers"
        )
        losers_data = await self._request(
            "/v2/snapshot/locale/us/markets/stocks/losers"
        )
        if not gainers_data and not losers_data:
            return None
        return {
            "gainers": (gainers_data or {}).get("tickers", [])[:10],
            "losers": (losers_data or {}).get("tickers", [])[:10],
        }

    # ── Splits ───────────────────────────────────────────────────────────────

    async def fetch_splits(self, ticker: str) -> list[dict]:
        """Fetch stock splits from /stocks/v1/splits."""
        data = await self._request(
            "/stocks/v1/splits",
            {"ticker": ticker, "limit": 10},
        )
        if not data:
            return []
        results = data.get("results", data.get("data", []))
        if not isinstance(results, list):
            return []
        return [
            {
                "ticker": ticker,
                "execution_date": r.get("execution_date", r.get("date", "")),
                "split_from": r.get("split_from"),
                "split_to": r.get("split_to"),
                "split_ratio": f"{int(r.get('split_to', 0))}:{int(r.get('split_from', 0))}"
                if r.get("split_to") and r.get("split_from")
                else r.get("split_ratio", ""),
                "source": "massive",
            }
            for r in results
        ]

    def status(self) -> dict:
        return {
            "provider": "massive",
            "total_keys": self._total_keys,
            "rate_limiters": [rl.status() for rl in self._rate_limiters],
            "circuit_breaker": self.circuit.status(),
        }


def _safe_float(val) -> float | None:
    """Safely convert a value to float, returning None on failure."""
    if val is None:
        return None
    try:
        f = float(val)
        return f if not (f != f) else None  # NaN check
    except (ValueError, TypeError):
        return None
