"""Live integration tests for ProviderRegistry fallback chains.

Tests verify that fallback chains work correctly with real providers
when upstream providers are unavailable (set to None) or return errors.

Run with: pytest -m live tests/test_providers/test_fallback_chains.py -v
"""

from __future__ import annotations

import os

import pytest

from portfolio_advisor.providers.registry import ProviderRegistry
from portfolio_advisor.providers.fred_provider import FREDProvider
from portfolio_advisor.providers.massive_provider import MassiveProvider
from portfolio_advisor.providers.alpha_vantage_provider import AlphaVantageProvider
from portfolio_advisor.providers.yfinance_provider import YFinanceProvider

# ── Helper: collect keys from environment ─────────────────────────────────────


def _collect_massive_keys() -> list[str]:
    combined = os.environ.get("PA_MASSIVE_API_KEYS", "")
    if combined:
        return [k.strip() for k in combined.split(",") if k.strip()]
    keys = []
    for i in range(1, 5):
        k = os.environ.get(f"PA_MASSIVE_API_KEY_{i}", "")
        if k:
            keys.append(k)
    legacy = os.environ.get("PA_MASSIVE_API_KEY", "")
    if not keys and legacy:
        keys = [legacy]
    return keys


def _collect_av_keys() -> list[str]:
    combined = os.environ.get("PA_ALPHA_VANTAGE_API_KEYS", "")
    if combined:
        return [k.strip() for k in combined.split(",") if k.strip()]
    keys = []
    for i in range(1, 3):
        k = os.environ.get(f"PA_ALPHA_VANTAGE_API_KEY_{i}", "")
        if k:
            keys.append(k)
    legacy = os.environ.get("PA_ALPHA_VANTAGE_API_KEY", "")
    if not keys and legacy:
        keys = [legacy]
    return keys


MASSIVE_KEYS = _collect_massive_keys()
AV_KEYS = _collect_av_keys()
FRED_KEY = os.environ.get("PA_FRED_API_KEY", "")

has_massive = bool(MASSIVE_KEYS)
has_av = bool(AV_KEYS)
has_fred = bool(FRED_KEY)


# ══════════════════════════════════════════════════════════════════════════════
# Treasury Yields Fallback
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.live
@pytest.mark.skipif(not has_fred, reason="PA_FRED_API_KEY not set")
async def test_yields_massive_to_fred_fallback():
    """With massive=None and fred=real, yields come from FRED with source='fred'."""
    fred = FREDProvider(FRED_KEY)
    registry = ProviderRegistry(massive=None, fred=fred, alpha_vantage=None)
    try:
        data, source = await registry.fetch_treasury_yields()
        assert source == "fred", f"Expected source='fred', got '{source}'"
        assert data is not None
        assert isinstance(data, dict)
        assert "10y" in data
        assert isinstance(data["10y"], float)
    finally:
        await registry.close()


@pytest.mark.live
async def test_yields_all_down_to_yfinance():
    """With massive=None, fred=None, yfinance provides yields."""
    registry = ProviderRegistry(massive=None, fred=None, alpha_vantage=None)
    try:
        data, source = await registry.fetch_treasury_yields()
        assert source == "yfinance", f"Expected source='yfinance', got '{source}'"
        assert data is not None
        assert isinstance(data, dict)
        # yfinance provides at least 10y
        assert "10y" in data
    finally:
        await registry.close()


# ══════════════════════════════════════════════════════════════════════════════
# VIX Fallback
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.live
async def test_vix_fred_to_yfinance_fallback():
    """With fred=None, yfinance VIX is returned."""
    registry = ProviderRegistry(massive=None, fred=None, alpha_vantage=None)
    try:
        val, source = await registry.fetch_vix()
        assert source == "yfinance", f"Expected source='yfinance', got '{source}'"
        assert val is not None
        assert isinstance(val, float)
        assert 5 <= val <= 80, f"VIX out of range: {val}"
    finally:
        await registry.close()


# ══════════════════════════════════════════════════════════════════════════════
# Fundamentals Fallback
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.live
@pytest.mark.skipif(
    not has_massive or not has_av,
    reason="Need both PA_MASSIVE_API_KEYS and PA_ALPHA_VANTAGE_API_KEYS",
)
async def test_fundamentals_massive_to_av_fallback():
    """Massive returns None (403 on free tier), AV returns data with source='alpha_vantage'."""
    massive = MassiveProvider(MASSIVE_KEYS)
    av = AlphaVantageProvider(AV_KEYS)
    registry = ProviderRegistry(massive=massive, fred=None, alpha_vantage=av)
    try:
        data, source = await registry.fetch_fundamentals("AAPL")
        # Massive free tier returns None for fundamentals (403)
        # so fallback to AV is expected; but if Massive works, that's also fine
        assert data is not None, "Expected fundamentals from at least one provider"
        assert source in ("massive", "alpha_vantage"), f"Unexpected source: {source}"
        assert "pe_ratio" in data
    finally:
        await registry.close()


# ══════════════════════════════════════════════════════════════════════════════
# Credit Spread (FRED only)
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.live
async def test_credit_spread_fred_only():
    """With fred=None, credit spread returns (None, 'unavailable')."""
    registry = ProviderRegistry(massive=None, fred=None, alpha_vantage=None)
    try:
        val, source = await registry.fetch_credit_spread()
        assert val is None
        assert source == "unavailable"
    finally:
        await registry.close()


# ══════════════════════════════════════════════════════════════════════════════
# All Providers Down
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.live
async def test_all_providers_down_graceful():
    """With all providers=None, every fetch returns None/empty/unavailable."""
    # Create a yfinance provider that always fails to simulate total outage
    class BrokenYFinance(YFinanceProvider):
        async def fetch_treasury_yields(self):
            return None

        async def fetch_vix(self):
            return None

    registry = ProviderRegistry(
        massive=None,
        fred=None,
        alpha_vantage=None,
        yfinance=BrokenYFinance(),
    )
    try:
        # Treasury yields
        data, source = await registry.fetch_treasury_yields()
        assert data is None
        assert source == "unavailable"

        # VIX
        val, source = await registry.fetch_vix()
        assert val is None
        assert source == "unavailable"

        # Credit spread
        val, source = await registry.fetch_credit_spread()
        assert val is None
        assert source == "unavailable"

        # Earnings
        entries, source = await registry.fetch_earnings(["AAPL"])
        assert entries == []
        assert source == "unavailable"

        # News
        articles = await registry.fetch_news(["AAPL"])
        assert articles == []

        # Fundamentals
        data, source = await registry.fetch_fundamentals("AAPL")
        assert data is None
        assert source == "unavailable"

        # Short interest
        result = await registry.fetch_short_interest(["AAPL"])
        assert result == []

        # Dividends
        result = await registry.fetch_dividends("AAPL")
        assert result == []

        # Splits
        result = await registry.fetch_splits("AAPL")
        assert result == []

        # Analyst ratings
        result = await registry.fetch_analyst_ratings("AAPL")
        assert result == []

        # Market movers
        result = await registry.fetch_market_movers()
        assert result is None
    finally:
        await registry.close()
