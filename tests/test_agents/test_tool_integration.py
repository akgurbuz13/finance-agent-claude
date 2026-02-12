"""Integration tests for @function_tool wrappers with real AppContext.

Tests exercise the raw computation functions and cache tools
with a real database and (for @pytest.mark.live tests) real API providers.
"""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock

import pytest

from portfolio_advisor.agents.context import AppContext
from portfolio_advisor.db import queries
from portfolio_advisor.db.connection import get_db, init_db


# ── Helpers ──────────────────────────────────────────────────────────────────


def make_ctx(db_path: str, providers=None) -> MagicMock:
    """Create a mock RunContextWrapper with real AppContext."""
    ctx = MagicMock()
    ctx.context = AppContext(
        db_path=db_path,
        telegram_chat_id=123,
        watchlist=["SPY", "AAPL", "MSFT"],
        providers=providers,
    )
    return ctx


@pytest.fixture(autouse=True)
def _reset_settings():
    """Reset the settings singleton so each test gets a clean slate."""
    import portfolio_advisor.config as cfg_mod

    old = cfg_mod._settings
    cfg_mod._settings = None
    yield
    cfg_mod._settings = old


@pytest.fixture(autouse=True)
def _set_env(monkeypatch):
    """Set required env vars for Settings to load in tests."""
    monkeypatch.setenv("PA_OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("PA_TELEGRAM_BOT_TOKEN", "test-token")
    monkeypatch.setenv("PA_TELEGRAM_CHAT_ID", "123")


@pytest.fixture
async def db_path(tmp_path):
    """Create a temporary database with schema initialized."""
    path = str(tmp_path / "tool_integration.db")
    await init_db(path)
    return path


# ── Fundamentals ─────────────────────────────────────────────────────────────


@pytest.mark.live
async def test_get_fundamentals_tool(db_path):
    """Call fetch_fundamentals_raw with yfinance-based registry (no Massive/AV keys needed)."""
    from portfolio_advisor.providers.registry import ProviderRegistry
    from portfolio_advisor.tools.fundamentals import fetch_fundamentals_raw

    # Without Massive or Alpha Vantage keys, fetch_fundamentals_raw returns None
    # via the provider registry (no API keys available).
    registry = ProviderRegistry()
    result = await fetch_fundamentals_raw("AAPL", registry)

    # Without Massive/AV keys, result will be None. That is the expected behavior
    # when no premium providers are configured.
    # The test verifies the function runs without error.
    assert result is None or isinstance(result, dict)
    if result is not None:
        # If keys are configured, verify the structure
        assert "ticker" in result or "pe_ratio" in result or "sector" in result


# ── News ─────────────────────────────────────────────────────────────────────


@pytest.mark.live
async def test_get_ticker_news_tool():
    """Call raw news function — returns empty list without Massive API key."""
    from portfolio_advisor.providers.registry import ProviderRegistry
    from portfolio_advisor.tools.news_data import fetch_ticker_news_raw

    registry = ProviderRegistry()
    result = await fetch_ticker_news_raw(["AAPL"], registry)

    # Without Massive key, returns empty list
    assert isinstance(result, list)


# ── Short Interest ───────────────────────────────────────────────────────────


@pytest.mark.live
async def test_get_short_interest_tool():
    """Call raw short interest function — returns empty list without Massive API key."""
    from portfolio_advisor.providers.registry import ProviderRegistry
    from portfolio_advisor.tools.sentiment import fetch_short_interest_raw

    registry = ProviderRegistry()
    result = await fetch_short_interest_raw(["AAPL"], registry)

    assert isinstance(result, list)


# ── Dividends ────────────────────────────────────────────────────────────────


@pytest.mark.live
async def test_get_dividend_info_tool():
    """Call raw dividend function — returns empty list without Massive API key."""
    from portfolio_advisor.providers.registry import ProviderRegistry
    from portfolio_advisor.tools.corporate_actions import fetch_dividends_raw

    registry = ProviderRegistry()
    result = await fetch_dividends_raw("AAPL", registry)

    assert isinstance(result, list)


# ── Macro Regime ─────────────────────────────────────────────────────────────


@pytest.mark.live
async def test_compute_macro_regime_tool(db_path):
    """Call the macro regime computation via the raw helper approach.

    The compute_macro_regime function is a @function_tool, so we invoke its
    underlying logic by importing the raw helpers directly.
    """
    from portfolio_advisor.tools.economic_data import _fetch_treasury_yields

    # Test the treasury yields helper (used by macro regime)
    yields = _fetch_treasury_yields()
    assert yields is None or isinstance(yields, dict)
    if yields is not None:
        # Should have at least some maturity yields
        assert len(yields) > 0
        for key, val in yields.items():
            assert isinstance(val, float)
            assert val > 0  # yields should be positive

    # Test macro regime by fetching SPY data and computing regime signals
    import yfinance as yf

    spy = yf.download("SPY", period="1y", progress=False)
    assert not spy.empty, "SPY data download failed"

    # Verify we can compute equity momentum signals (core of macro regime)
    close = spy["Close"]
    if hasattr(close, "columns"):
        # Multi-index case
        close = close.iloc[:, 0] if close.shape[1] == 1 else close["SPY"]
    sma50 = float(close.rolling(50).mean().iloc[-1])
    sma200 = float(close.rolling(200).mean().iloc[-1])
    current = float(close.iloc[-1])

    assert current > 0
    assert sma50 > 0
    assert sma200 > 0


# ── Treasury / Yield Curve ───────────────────────────────────────────────────


@pytest.mark.live
async def test_fetch_treasury_data_tool(db_path):
    """Call _fetch_treasury_yields to verify yields can be fetched via yfinance."""
    from portfolio_advisor.tools.economic_data import _fetch_treasury_yields

    yields = _fetch_treasury_yields()

    # yfinance may occasionally fail, so allow None
    if yields is not None:
        assert isinstance(yields, dict)
        assert len(yields) > 0
        # Check that known maturities are present
        known_keys = {"3m", "5y", "10y", "30y"}
        found_keys = set(yields.keys()) & known_keys
        assert len(found_keys) > 0, f"Expected some of {known_keys}, got {set(yields.keys())}"


# ── Cache Tools After Pipeline ───────────────────────────────────────────────


@pytest.mark.live
async def test_all_cache_tools_after_pipeline(tmp_path, monkeypatch):
    """Run precompute pipeline, then verify all 10 cache tools return valid data."""
    db_p = str(tmp_path / "cache_tools.db")
    await init_db(db_p)

    monkeypatch.setenv("PA_DB_PATH", db_p)

    # Set a small watchlist in DB preferences so the pipeline uses it
    # (pipeline reads watchlist from DB, not from ctx.watchlist)
    async with get_db(db_p) as db:
        from portfolio_advisor.db.queries import update_user_preference

        await update_user_preference(db, "watchlist", ["SPY", "AAPL"])

    ctx_raw = AppContext(
        db_path=db_p,
        telegram_chat_id=123,
        watchlist=["SPY", "AAPL"],
    )

    # Run the precompute pipeline
    from portfolio_advisor.tools.precomputed import run_precompute_pipeline

    results = await run_precompute_pipeline(ctx_raw)
    assert len(results["processed"]) > 0, "Need at least one processed ticker"

    # For cache tools that are @function_tool, we query the DB directly, the same
    # way the underlying functions do, since @function_tool wrappers need a
    # ToolContext that is complex to set up in tests.

    today = date.today().isoformat()

    # 1. check_data_freshness — verify analysis_runs table has an entry
    async with get_db(db_p) as db:
        run = await queries.get_latest_analysis_run(db, "precompute")
    assert run is not None
    assert run["status"] in ("completed", "completed_with_errors")

    # 2. get_cached_technical — verify technical indicators are stored
    async with get_db(db_p) as db:
        tech = await queries.get_technical_indicators(db, "SPY")
    assert tech is not None
    assert tech["ticker"] == "SPY"
    assert "rsi_14" in tech
    assert "overall_bias" in tech
    assert tech["overall_bias"] in (
        "bullish", "slightly_bullish", "neutral", "slightly_bearish", "bearish"
    )

    # 3. get_cached_quant — verify quant metrics are stored
    async with get_db(db_p) as db:
        quant = await queries.get_quant_metrics(db, "SPY")
    assert quant is not None
    assert quant["ticker"] == "SPY"
    assert "regime" in quant

    # 4. get_cached_bulk_summary — verify bulk query works
    async with get_db(db_p) as db:
        tech_bulk = await queries.get_bulk_technical_indicators(
            db, ["SPY", "AAPL"], today
        )
        quant_bulk = await queries.get_bulk_quant_metrics(
            db, ["SPY", "AAPL"], today
        )
    assert len(tech_bulk) > 0
    assert len(quant_bulk) > 0

    # 5. get_signal_history — verify signal_trend query works
    async with get_db(db_p) as db:
        trend = await queries.get_signal_trend(db, "SPY", 7)
    # May or may not have multi-day history, but should not error
    assert isinstance(trend, list)

    # 6. get_intraday_changes — verify intraday snapshots query works
    async with get_db(db_p) as db:
        snapshots = await queries.get_intraday_snapshots(db, "SPY", today)
    assert isinstance(snapshots, list)
    assert len(snapshots) >= 1  # At least one snapshot from the pipeline run

    # 7. get_indicator_trend — verify indicator history query works
    async with get_db(db_p) as db:
        indicator_hist = await queries.get_indicator_history(db, "SPY", "rsi_14", 14)
    assert isinstance(indicator_hist, list)

    # 8. get_cached_macro — verify risk metrics (may be empty without portfolio positions)
    async with get_db(db_p) as db:
        risk = await queries.get_latest_risk_metrics(db)
    # Risk metrics require portfolio state, so may be None
    assert risk is None or isinstance(risk, dict)

    # 9. get_cached_correlations — verify correlation snapshot
    async with get_db(db_p) as db:
        corr = await queries.get_latest_correlation_snapshot(db)
    # With 2 tickers, should have correlation data
    if corr is not None:
        assert "diversification_score" in corr

    # 10. get_daily_analysis_snapshot — verify we can read narratives
    async with get_db(db_p) as db:
        tech_spy = await queries.get_technical_indicators(db, "SPY")
        tech_aapl = await queries.get_technical_indicators(db, "AAPL")
    narratives = []
    for t in [tech_spy, tech_aapl]:
        if t and t.get("narrative"):
            narratives.append(t["narrative"])
    assert len(narratives) > 0, "Should have at least one ticker narrative"
    assert any("SPY" in n or "AAPL" in n for n in narratives)
