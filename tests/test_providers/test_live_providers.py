"""Live API integration tests for individual data providers.

All tests require real API keys and network access.
Run with: pytest -m live tests/test_providers/test_live_providers.py -v
"""

from __future__ import annotations

import os
import time

import pytest

from portfolio_advisor.providers.massive_provider import MassiveProvider
from portfolio_advisor.providers.fred_provider import FREDProvider
from portfolio_advisor.providers.alpha_vantage_provider import AlphaVantageProvider
from portfolio_advisor.providers.yfinance_provider import YFinanceProvider
from portfolio_advisor.providers.coingecko_provider import CoinGeckoProvider

# ── Helper: collect keys from environment ─────────────────────────────────────


def _collect_massive_keys() -> list[str]:
    """Collect Massive API keys from PA_MASSIVE_API_KEYS or PA_MASSIVE_API_KEY_1.._4."""
    combined = os.environ.get("PA_MASSIVE_API_KEYS", "")
    if combined:
        return [k.strip() for k in combined.split(",") if k.strip()]
    keys = []
    for i in range(1, 5):
        k = os.environ.get(f"PA_MASSIVE_API_KEY_{i}", "")
        if k:
            keys.append(k)
    # Also try single legacy key
    legacy = os.environ.get("PA_MASSIVE_API_KEY", "")
    if not keys and legacy:
        keys = [legacy]
    return keys


def _collect_av_keys() -> list[str]:
    """Collect Alpha Vantage API keys from PA_ALPHA_VANTAGE_API_KEYS or _1/_2."""
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


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def massive_provider() -> MassiveProvider:
    return MassiveProvider(MASSIVE_KEYS)


@pytest.fixture
def fred_provider() -> FREDProvider:
    return FREDProvider(FRED_KEY)


@pytest.fixture
def av_provider() -> AlphaVantageProvider:
    return AlphaVantageProvider(AV_KEYS)


@pytest.fixture
def yfinance_provider() -> YFinanceProvider:
    return YFinanceProvider()


@pytest.fixture
def coingecko_provider() -> CoinGeckoProvider:
    return CoinGeckoProvider()


# ══════════════════════════════════════════════════════════════════════════════
# Massive Provider
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.live
@pytest.mark.skipif(not has_massive, reason="PA_MASSIVE_API_KEYS not set")
async def test_massive_treasury_yields_live(massive_provider: MassiveProvider):
    """fetch_treasury_yields() returns dict with 2y/5y/10y/30y as positive floats."""
    result = await massive_provider.fetch_treasury_yields()
    assert result is not None, "fetch_treasury_yields returned None"
    assert isinstance(result, dict)
    for key in ("2y", "5y", "10y", "30y"):
        assert key in result, f"Missing yield key: {key}"
        assert isinstance(result[key], float), f"{key} is not float: {type(result[key])}"
        assert result[key] > 0, f"{key} yield should be positive, got {result[key]}"


@pytest.mark.live
@pytest.mark.skipif(not has_massive, reason="PA_MASSIVE_API_KEYS not set")
async def test_massive_news_single_ticker(massive_provider: MassiveProvider):
    """fetch_news(['AAPL']) returns list of articles with title, url, published_utc."""
    articles = await massive_provider.fetch_news(["AAPL"])
    assert isinstance(articles, list)
    assert len(articles) > 0, "Expected at least one news article for AAPL"
    for article in articles:
        assert "title" in article, "Article missing 'title'"
        assert "url" in article, "Article missing 'url'"
        assert "published_utc" in article, "Article missing 'published_utc'"
        assert isinstance(article["title"], str)
        assert len(article["title"]) > 0


@pytest.mark.live
@pytest.mark.skipif(not has_massive, reason="PA_MASSIVE_API_KEYS not set")
async def test_massive_news_multi_ticker(massive_provider: MassiveProvider):
    """fetch_news(['AAPL', 'MSFT']) returns articles that may span multiple tickers."""
    articles = await massive_provider.fetch_news(["AAPL", "MSFT"])
    assert isinstance(articles, list)
    assert len(articles) > 0, "Expected at least one article for AAPL+MSFT"
    # At least verify we got articles back; tickers field may reference either ticker
    all_tickers_seen = set()
    for article in articles:
        for t in article.get("tickers", []):
            all_tickers_seen.add(t)
    # We expect at least one of the requested tickers to appear
    assert all_tickers_seen & {"AAPL", "MSFT"}, (
        f"Expected AAPL or MSFT in article tickers, got {all_tickers_seen}"
    )


@pytest.mark.live
@pytest.mark.skipif(not has_massive, reason="PA_MASSIVE_API_KEYS not set")
async def test_massive_short_interest(massive_provider: MassiveProvider):
    """fetch_short_interest(['AAPL']) returns list with short_interest, short_pct_float."""
    result = await massive_provider.fetch_short_interest(["AAPL"])
    assert isinstance(result, list)
    assert len(result) > 0, "Expected short interest data for AAPL"
    entry = result[0]
    assert entry["ticker"] == "AAPL"
    # short_interest should be present (may be None on some tiers)
    assert "short_interest" in entry
    assert "short_pct_float" in entry
    assert "days_to_cover" in entry


@pytest.mark.live
@pytest.mark.skipif(not has_massive, reason="PA_MASSIVE_API_KEYS not set")
async def test_massive_dividends(massive_provider: MassiveProvider):
    """fetch_dividends('AAPL') returns list with ex_date and amount > 0."""
    result = await massive_provider.fetch_dividends("AAPL")
    assert isinstance(result, list)
    assert len(result) > 0, "Expected dividend data for AAPL"
    entry = result[0]
    assert "ex_date" in entry
    assert entry["ex_date"], "ex_date should be non-empty"
    assert "amount" in entry
    assert entry["amount"] is not None and entry["amount"] > 0, (
        f"Dividend amount should be positive, got {entry['amount']}"
    )


@pytest.mark.live
@pytest.mark.skipif(not has_massive, reason="PA_MASSIVE_API_KEYS not set")
async def test_massive_splits(massive_provider: MassiveProvider):
    """fetch_splits('AAPL') returns list (may be empty if no recent splits)."""
    result = await massive_provider.fetch_splits("AAPL")
    assert isinstance(result, list)
    # AAPL had a 4:1 split in 2020, so there should be data
    # But the API may return empty depending on the time window, so just check type
    if result:
        entry = result[0]
        assert entry["ticker"] == "AAPL"
        assert "execution_date" in entry
        assert "split_ratio" in entry


@pytest.mark.live
@pytest.mark.skipif(not has_massive, reason="PA_MASSIVE_API_KEYS not set")
async def test_massive_403_endpoints(massive_provider: MassiveProvider):
    """Endpoints requiring paid tier return None or empty on free tier."""
    # fetch_earnings returns empty list on 403
    earnings = await massive_provider.fetch_earnings(["AAPL"])
    assert earnings is None or isinstance(earnings, list)

    # fetch_fundamentals returns None on 403
    fundamentals = await massive_provider.fetch_fundamentals("AAPL")
    assert fundamentals is None or isinstance(fundamentals, dict)

    # fetch_analyst_ratings returns empty list on 403
    ratings = await massive_provider.fetch_analyst_ratings("AAPL")
    assert ratings is None or isinstance(ratings, list)

    # fetch_market_movers returns None on 403
    movers = await massive_provider.fetch_market_movers()
    assert movers is None or isinstance(movers, dict)


@pytest.mark.live
@pytest.mark.skipif(not has_massive, reason="PA_MASSIVE_API_KEYS not set")
async def test_massive_rate_limiter_enforced():
    """Rate limiter with 1 key (5 calls/min) should throttle 6 rapid calls > 1 second."""
    # Use a single key with a strict rate limit to force throttling
    provider = MassiveProvider([MASSIVE_KEYS[0]], per_key_rate_limit=5)
    start = time.monotonic()

    # Make 6 rapid calls — the 6th should be delayed by the rate limiter
    for _ in range(6):
        await provider.fetch_treasury_yields()

    elapsed = time.monotonic() - start
    await provider.close()

    # With 5 tokens max and refill_rate=5/60=0.083/s, the 6th call must wait ~12s
    # for a full token refill. But the bucket starts full (5 tokens), so calls 1-5
    # are instant and the 6th must wait. Total should be > 1 second.
    assert elapsed > 1.0, (
        f"Expected rate limiter to throttle 6 calls with 1 key (5/min), "
        f"but completed in {elapsed:.2f}s"
    )


# ══════════════════════════════════════════════════════════════════════════════
# FRED Provider
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.live
@pytest.mark.skipif(not has_fred, reason="PA_FRED_API_KEY not set")
async def test_fred_treasury_yields_live(fred_provider: FREDProvider):
    """fetch_treasury_yields() returns dict with 2y/5y/10y/30y in 0-15% range."""
    result = await fred_provider.fetch_treasury_yields()
    assert result is not None, "FRED fetch_treasury_yields returned None"
    assert isinstance(result, dict)
    for key in ("2y", "5y", "10y", "30y"):
        assert key in result, f"Missing yield key: {key}"
        val = result[key]
        assert isinstance(val, float), f"{key} is not float: {type(val)}"
        assert 0 < val < 15, f"{key} yield out of range: {val}"


@pytest.mark.live
@pytest.mark.skipif(not has_fred, reason="PA_FRED_API_KEY not set")
async def test_fred_vix_live(fred_provider: FREDProvider):
    """fetch_vix() returns a float between 5 and 80."""
    result = await fred_provider.fetch_vix()
    assert result is not None, "FRED fetch_vix returned None"
    assert isinstance(result, float)
    assert 5 <= result <= 80, f"VIX out of expected range: {result}"


@pytest.mark.live
@pytest.mark.skipif(not has_fred, reason="PA_FRED_API_KEY not set")
async def test_fred_credit_spread_live(fred_provider: FREDProvider):
    """fetch_credit_spread() returns a float in 0.5-10 percentage point range."""
    result = await fred_provider.fetch_credit_spread()
    assert result is not None, "FRED fetch_credit_spread returned None"
    assert isinstance(result, float)
    assert 0.5 <= result <= 10, f"Credit spread out of expected range: {result}"


@pytest.mark.live
@pytest.mark.skipif(not has_fred, reason="PA_FRED_API_KEY not set")
async def test_fred_generic_series(fred_provider: FREDProvider):
    """fetch_series('UNRATE') returns recent unemployment rate in 2-15% range."""
    result = await fred_provider.fetch_series("UNRATE", limit=5)
    assert result is not None, "FRED fetch_series('UNRATE') returned None"
    assert isinstance(result, list)
    assert len(result) > 0, "Expected at least one observation"
    latest = result[0]
    assert "date" in latest
    assert "value" in latest
    assert isinstance(latest["value"], float)
    assert 2 <= latest["value"] <= 15, f"Unemployment rate out of range: {latest['value']}"


# ══════════════════════════════════════════════════════════════════════════════
# Alpha Vantage Provider
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.live
@pytest.mark.skipif(not has_av, reason="PA_ALPHA_VANTAGE_API_KEYS not set")
async def test_alpha_vantage_fundamentals_live(av_provider: AlphaVantageProvider):
    """fetch_fundamentals('AAPL') returns dict with pe_ratio, roe, sector, market_cap."""
    result = await av_provider.fetch_fundamentals("AAPL")
    assert result is not None, "Alpha Vantage fetch_fundamentals returned None"
    assert isinstance(result, dict)
    assert result["ticker"] == "AAPL"
    assert result["source"] == "alpha_vantage"
    # Key fields should be present
    for field in ("pe_ratio", "roe", "sector", "market_cap"):
        assert field in result, f"Missing field: {field}"
    # Sector should be non-empty
    assert result["sector"], "sector should be non-empty"
    # market_cap should be a large number
    assert result["market_cap"] is not None
    assert result["market_cap"] > 1_000_000_000, (
        f"AAPL market cap seems too small: {result['market_cap']}"
    )


@pytest.mark.live
@pytest.mark.skipif(not has_av, reason="PA_ALPHA_VANTAGE_API_KEYS not set")
async def test_alpha_vantage_daily_limit_tracking(av_provider: AlphaVantageProvider):
    """After one call, _daily_calls[used_key] should increment."""
    # Record initial state
    initial_total = sum(av_provider._daily_calls)

    await av_provider.fetch_fundamentals("MSFT")

    new_total = sum(av_provider._daily_calls)
    assert new_total == initial_total + 1, (
        f"Expected daily calls to increment by 1, got {initial_total} -> {new_total}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# YFinance Provider
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.live
async def test_yfinance_treasury_yields_live(yfinance_provider: YFinanceProvider):
    """YFinanceProvider().fetch_treasury_yields() returns dict with at least '10y' key."""
    result = await yfinance_provider.fetch_treasury_yields()
    assert result is not None, "yfinance fetch_treasury_yields returned None"
    assert isinstance(result, dict)
    assert "10y" in result, f"Expected '10y' key, got keys: {list(result.keys())}"
    assert isinstance(result["10y"], float)
    assert result["10y"] > 0


@pytest.mark.live
async def test_yfinance_vix_live(yfinance_provider: YFinanceProvider):
    """YFinanceProvider().fetch_vix() returns a float between 5 and 80."""
    result = await yfinance_provider.fetch_vix()
    assert result is not None, "yfinance fetch_vix returned None"
    assert isinstance(result, float)
    assert 5 <= result <= 80, f"VIX out of expected range: {result}"


# ══════════════════════════════════════════════════════════════════════════════
# CoinGecko Provider (rate limiter only — no fetch_ohlcv on provider)
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.live
async def test_coingecko_ohlcv_live():
    """CoinGecko OHLCV for BTC via _fetch_crypto_ohlcv returns data with close values."""
    from portfolio_advisor.tools.market_data import _fetch_crypto_ohlcv

    bars = await _fetch_crypto_ohlcv("bitcoin", days=30)
    assert isinstance(bars, list)
    assert len(bars) > 0, "Expected at least one OHLCV bar for bitcoin"
    for bar in bars:
        assert "close" in bar, f"Bar missing 'close': {bar}"
        assert isinstance(bar["close"], float)
        assert bar["close"] > 0
