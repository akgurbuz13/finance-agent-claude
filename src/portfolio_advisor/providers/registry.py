"""Provider registry with fallback chains per data type."""

from __future__ import annotations

import logging

from portfolio_advisor.providers.alpha_vantage_provider import AlphaVantageProvider
from portfolio_advisor.providers.coingecko_provider import CoinGeckoProvider
from portfolio_advisor.providers.fred_provider import FREDProvider
from portfolio_advisor.providers.massive_provider import MassiveProvider
from portfolio_advisor.providers.yfinance_provider import YFinanceProvider

logger = logging.getLogger(__name__)


class ProviderRegistry:
    """Manages all data providers and implements fallback chains.

    Fallback chains per data type:
    - Treasury Yields: Massive -> FRED -> yfinance
    - VIX: FRED -> yfinance
    - Credit Spread: FRED -> (ETF proxy in caller)
    - Earnings: Massive -> (yfinance in caller)
    - News: Massive -> (WebSearchTool in caller)
    - Fundamentals: Massive -> Alpha Vantage
    - Short Interest: Massive
    - Dividends: Massive
    - Analyst Ratings: Massive
    """

    def __init__(
        self,
        fred: FREDProvider | None = None,
        massive: MassiveProvider | None = None,
        alpha_vantage: AlphaVantageProvider | None = None,
        yfinance: YFinanceProvider | None = None,
        coingecko: CoinGeckoProvider | None = None,
    ):
        self.fred = fred
        self.massive = massive
        self.alpha_vantage = alpha_vantage
        self.yfinance = yfinance or YFinanceProvider()
        self.coingecko = coingecko or CoinGeckoProvider()

    async def fetch_treasury_yields(self) -> tuple[dict | None, str]:
        """Fetch treasury yields with fallback chain. Returns (data, source)."""
        # Try Massive first (single API call for all maturities)
        if self.massive:
            data = await self.massive.fetch_treasury_yields()
            if data:
                return data, "massive"

        # Try FRED (4 API calls, one per maturity)
        if self.fred:
            data = await self.fred.fetch_treasury_yields()
            if data:
                return data, "fred"

        # Fallback to yfinance
        data = await self.yfinance.fetch_treasury_yields()
        if data:
            return data, "yfinance"

        return None, "unavailable"

    async def fetch_vix(self) -> tuple[float | None, str]:
        """Fetch VIX with fallback chain. Returns (value, source)."""
        if self.fred:
            val = await self.fred.fetch_vix()
            if val is not None:
                return val, "fred"

        val = await self.yfinance.fetch_vix()
        if val is not None:
            return val, "yfinance"

        return None, "unavailable"

    async def fetch_credit_spread(self) -> tuple[float | None, str]:
        """Fetch HY OAS credit spread. Returns (value_pct, source)."""
        if self.fred:
            val = await self.fred.fetch_credit_spread()
            if val is not None:
                return val, "fred"

        return None, "unavailable"

    async def fetch_earnings(self, tickers: list[str]) -> tuple[list[dict], str]:
        """Fetch earnings with fallback. Returns (entries, source)."""
        if self.massive:
            entries = await self.massive.fetch_earnings(tickers)
            if entries:
                return entries, "massive_benzinga"
        return [], "unavailable"

    async def fetch_news(self, tickers: list[str], limit: int = 10) -> list[dict]:
        """Fetch news from Massive. Returns list of articles."""
        if self.massive:
            return await self.massive.fetch_news(tickers, limit)
        return []

    async def fetch_fundamentals(self, ticker: str) -> tuple[dict | None, str]:
        """Fetch fundamentals with fallback. Returns (data, source)."""
        if self.massive:
            data = await self.massive.fetch_fundamentals(ticker)
            if data:
                return data, "massive"

        if self.alpha_vantage:
            data = await self.alpha_vantage.fetch_fundamentals(ticker)
            if data:
                return data, "alpha_vantage"

        return None, "unavailable"

    async def fetch_short_interest(self, tickers: list[str]) -> list[dict]:
        """Fetch short interest data. Returns list of entries."""
        if self.massive:
            return await self.massive.fetch_short_interest(tickers)
        return []

    async def fetch_dividends(self, ticker: str) -> list[dict]:
        """Fetch dividend history."""
        if self.massive:
            return await self.massive.fetch_dividends(ticker)
        return []

    async def fetch_splits(self, ticker: str) -> list[dict]:
        """Fetch stock split history."""
        if self.massive:
            return await self.massive.fetch_splits(ticker)
        return []

    async def fetch_analyst_ratings(self, ticker: str) -> list[dict]:
        """Fetch analyst ratings."""
        if self.massive:
            return await self.massive.fetch_analyst_ratings(ticker)
        return []

    async def fetch_market_movers(self) -> dict | None:
        """Fetch market movers (top gainers/losers)."""
        if self.massive:
            return await self.massive.fetch_market_movers()
        return None

    async def close(self) -> None:
        """Close all provider HTTP clients."""
        if self.fred:
            await self.fred.close()
        if self.massive:
            await self.massive.close()
        if self.alpha_vantage:
            await self.alpha_vantage.close()

    def provider_status(self) -> dict:
        """Return status of all providers (for health check)."""
        status = {}
        if self.fred:
            status["fred"] = self.fred.status()
        if self.massive:
            status["massive"] = self.massive.status()
        if self.alpha_vantage:
            status["alpha_vantage"] = self.alpha_vantage.status()
        status["yfinance"] = self.yfinance.status()
        status["coingecko"] = self.coingecko.status()
        return status


_singleton_registry: ProviderRegistry | None = None


def create_registry(settings) -> ProviderRegistry:
    """Create a ProviderRegistry from application settings."""
    fred = None
    if settings.fred_api_key:
        fred = FREDProvider(settings.fred_api_key)
        logger.info("FRED provider enabled")

    massive = None
    massive_keys = settings.get_massive_keys()
    if massive_keys:
        massive = MassiveProvider(massive_keys)
        logger.info(
            f"Massive provider enabled with {len(massive_keys)} keys "
            f"({len(massive_keys) * 5} calls/min effective)"
        )

    alpha_vantage = None
    av_keys = settings.get_alpha_vantage_keys()
    if av_keys:
        alpha_vantage = AlphaVantageProvider(av_keys)
        logger.info(
            f"Alpha Vantage provider enabled with {len(av_keys)} keys "
            f"({len(av_keys) * 25} calls/day effective)"
        )

    return ProviderRegistry(
        fred=fred,
        massive=massive,
        alpha_vantage=alpha_vantage,
    )


def set_global_registry(registry: ProviderRegistry) -> None:
    """Set the singleton registry (called once at startup)."""
    global _singleton_registry
    _singleton_registry = registry


def get_global_registry() -> ProviderRegistry | None:
    """Get the singleton registry. Returns None if not initialized."""
    return _singleton_registry
