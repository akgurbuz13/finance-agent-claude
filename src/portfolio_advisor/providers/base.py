"""Protocol classes for data providers."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class EconomicDataProvider(Protocol):
    """Provider for economic/macro data (treasury yields, VIX, credit spreads)."""

    async def fetch_treasury_yields(self) -> dict | None:
        """Fetch current treasury yields. Returns {2y, 5y, 10y, 30y} or None."""
        ...

    async def fetch_vix(self) -> float | None:
        """Fetch current CBOE VIX level."""
        ...

    async def fetch_credit_spread(self) -> float | None:
        """Fetch HY OAS credit spread in percentage points."""
        ...


@runtime_checkable
class EarningsProvider(Protocol):
    """Provider for earnings calendar data."""

    async def fetch_earnings(self, tickers: list[str]) -> list[dict]:
        """Fetch earnings calendar for given tickers."""
        ...


@runtime_checkable
class NewsProvider(Protocol):
    """Provider for news and sentiment data."""

    async def fetch_news(self, tickers: list[str], limit: int = 10) -> list[dict]:
        """Fetch recent news for given tickers."""
        ...


@runtime_checkable
class FundamentalsProvider(Protocol):
    """Provider for fundamental financial data."""

    async def fetch_fundamentals(self, ticker: str) -> dict | None:
        """Fetch fundamental ratios for a ticker."""
        ...

    async def fetch_analyst_ratings(self, ticker: str) -> list[dict]:
        """Fetch analyst ratings/price targets for a ticker."""
        ...


@runtime_checkable
class SentimentProvider(Protocol):
    """Provider for short interest and sentiment data."""

    async def fetch_short_interest(self, tickers: list[str]) -> list[dict]:
        """Fetch short interest data for given tickers."""
        ...


@runtime_checkable
class CorporateActionsProvider(Protocol):
    """Provider for dividends and stock splits."""

    async def fetch_dividends(self, ticker: str) -> list[dict]:
        """Fetch dividend history for a ticker."""
        ...

    async def fetch_splits(self, ticker: str) -> list[dict]:
        """Fetch stock split history for a ticker."""
        ...
