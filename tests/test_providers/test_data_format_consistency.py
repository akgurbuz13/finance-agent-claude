"""Tests that verify output shapes are consistent across providers.

These tests fetch data from real providers and check that the returned
data structures conform to expected schemas, making cross-provider
substitution safe.

Run with: pytest -m live tests/test_providers/test_data_format_consistency.py -v
"""

from __future__ import annotations

import os

import pytest

from portfolio_advisor.providers.massive_provider import MassiveProvider
from portfolio_advisor.providers.fred_provider import FREDProvider
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

YIELD_KEYS = {"2y", "5y", "10y", "30y"}


# ══════════════════════════════════════════════════════════════════════════════
# Treasury Yields Shape
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.live
@pytest.mark.skipif(
    not has_massive or not has_fred,
    reason="Need both PA_MASSIVE_API_KEYS and PA_FRED_API_KEY",
)
async def test_treasury_yields_shape_massive_vs_fred():
    """Both Massive and FRED return dicts with the same keys {2y, 5y, 10y, 30y}, all float."""
    massive = MassiveProvider(MASSIVE_KEYS)
    fred = FREDProvider(FRED_KEY)
    try:
        massive_result = await massive.fetch_treasury_yields()
        fred_result = await fred.fetch_treasury_yields()

        # Both should return non-None dicts
        assert massive_result is not None, "Massive yields returned None"
        assert fred_result is not None, "FRED yields returned None"

        # Both should have the same key set
        assert set(massive_result.keys()) == YIELD_KEYS, (
            f"Massive keys mismatch: {set(massive_result.keys())} != {YIELD_KEYS}"
        )
        assert set(fred_result.keys()) == YIELD_KEYS, (
            f"FRED keys mismatch: {set(fred_result.keys())} != {YIELD_KEYS}"
        )

        # All values should be floats
        for key in YIELD_KEYS:
            assert isinstance(massive_result[key], float), (
                f"Massive {key} is {type(massive_result[key])}"
            )
            assert isinstance(fred_result[key], float), (
                f"FRED {key} is {type(fred_result[key])}"
            )
    finally:
        await massive.close()
        await fred.close()


@pytest.mark.live
async def test_treasury_yields_shape_yfinance():
    """yfinance returns dict with at least {10y} present, all float."""
    yf_provider = YFinanceProvider()
    result = await yf_provider.fetch_treasury_yields()
    assert result is not None, "yfinance yields returned None"
    assert isinstance(result, dict)
    assert "10y" in result, f"Expected '10y' in keys, got {list(result.keys())}"
    for key, val in result.items():
        assert isinstance(val, float), f"yfinance {key} is {type(val)}, expected float"


# ══════════════════════════════════════════════════════════════════════════════
# VIX Shape
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.live
@pytest.mark.skipif(not has_fred, reason="PA_FRED_API_KEY not set")
async def test_vix_shape_fred_vs_yfinance():
    """Both FRED and yfinance return a single float for VIX in range 5-80."""
    fred = FREDProvider(FRED_KEY)
    yf_provider = YFinanceProvider()
    try:
        fred_vix = await fred.fetch_vix()
        yf_vix = await yf_provider.fetch_vix()

        assert fred_vix is not None, "FRED VIX returned None"
        assert yf_vix is not None, "yfinance VIX returned None"

        assert isinstance(fred_vix, float), f"FRED VIX is {type(fred_vix)}"
        assert isinstance(yf_vix, float), f"yfinance VIX is {type(yf_vix)}"

        assert 5 <= fred_vix <= 80, f"FRED VIX out of range: {fred_vix}"
        assert 5 <= yf_vix <= 80, f"yfinance VIX out of range: {yf_vix}"
    finally:
        await fred.close()


# ══════════════════════════════════════════════════════════════════════════════
# Fundamentals Shape
# ══════════════════════════════════════════════════════════════════════════════

# The 18 standard keys returned by Alpha Vantage fundamentals
AV_FUNDAMENTALS_KEYS = {
    "ticker", "pe_ratio", "forward_pe", "pb_ratio", "ps_ratio", "ev_ebitda",
    "roe", "roa", "profit_margin", "operating_margin", "debt_to_equity",
    "current_ratio", "revenue_growth_yoy", "earnings_growth_yoy",
    "dividend_yield", "market_cap", "sector", "industry",
}


@pytest.mark.live
@pytest.mark.skipif(not has_av, reason="PA_ALPHA_VANTAGE_API_KEYS not set")
async def test_fundamentals_shape_av():
    """AV fundamentals has all 18 standard keys plus extra AV-specific ones."""
    av = AlphaVantageProvider(AV_KEYS)
    try:
        result = await av.fetch_fundamentals("AAPL")
        assert result is not None, "AV fundamentals returned None"

        # Check all 18 standard keys are present
        missing = AV_FUNDAMENTALS_KEYS - set(result.keys())
        assert not missing, f"Missing standard fundamentals keys: {missing}"

        # AV also adds source, analyst_target_price, analyst_rating
        assert result["source"] == "alpha_vantage"
    finally:
        await av.close()


# ══════════════════════════════════════════════════════════════════════════════
# Credit Spread Shape
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.live
@pytest.mark.skipif(not has_fred, reason="PA_FRED_API_KEY not set")
async def test_credit_spread_units_fred():
    """FRED credit spread returns float in 0.5-10 range (pct points, not bps)."""
    fred = FREDProvider(FRED_KEY)
    try:
        result = await fred.fetch_credit_spread()
        assert result is not None, "FRED credit spread returned None"
        assert isinstance(result, float), f"Credit spread is {type(result)}, expected float"
        # OAS is in percentage points (e.g. 3.5 means 350 bps)
        assert 0.5 <= result <= 10, (
            f"Credit spread {result} out of expected range 0.5-10 pct points"
        )
    finally:
        await fred.close()


# ══════════════════════════════════════════════════════════════════════════════
# News Article Shape
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.live
@pytest.mark.skipif(not has_massive, reason="PA_MASSIVE_API_KEYS not set")
async def test_news_article_shape():
    """Massive news articles have title(str), published_utc(str), url(str)."""
    massive = MassiveProvider(MASSIVE_KEYS)
    try:
        articles = await massive.fetch_news(["AAPL"], limit=5)
        assert isinstance(articles, list)
        assert len(articles) > 0, "Expected at least one article"
        for article in articles:
            assert isinstance(article.get("title"), str), (
                f"title should be str, got {type(article.get('title'))}"
            )
            assert len(article["title"]) > 0, "title should be non-empty"

            assert isinstance(article.get("published_utc"), str), (
                f"published_utc should be str, got {type(article.get('published_utc'))}"
            )

            assert isinstance(article.get("url"), str), (
                f"url should be str, got {type(article.get('url'))}"
            )
    finally:
        await massive.close()


# ══════════════════════════════════════════════════════════════════════════════
# Short Interest Shape
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.live
@pytest.mark.skipif(not has_massive, reason="PA_MASSIVE_API_KEYS not set")
async def test_short_interest_shape():
    """Massive short interest entries have ticker(str), short_pct_float(float|None)."""
    massive = MassiveProvider(MASSIVE_KEYS)
    try:
        result = await massive.fetch_short_interest(["AAPL"])
        assert isinstance(result, list)
        assert len(result) > 0, "Expected short interest data for AAPL"
        entry = result[0]

        assert isinstance(entry.get("ticker"), str), (
            f"ticker should be str, got {type(entry.get('ticker'))}"
        )
        assert entry["ticker"] == "AAPL"

        # short_pct_float can be float or None
        spf = entry.get("short_pct_float")
        assert spf is None or isinstance(spf, float), (
            f"short_pct_float should be float|None, got {type(spf)}"
        )
    finally:
        await massive.close()
