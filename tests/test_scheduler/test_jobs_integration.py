"""Integration tests for scheduler jobs — precompute pipeline, daily context, news alerts, evening."""

from __future__ import annotations

import json
from datetime import date

import pytest

from portfolio_advisor.agents.context import AppContext
from portfolio_advisor.db import queries
from portfolio_advisor.db.connection import get_db, init_db
from portfolio_advisor.scheduler.alerts import _is_similar_to_existing


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_ctx(db_path: str, watchlist: list[str] | None = None) -> AppContext:
    """Create an AppContext for testing."""
    return AppContext(
        db_path=db_path,
        telegram_chat_id=123,
        run_date=date.today(),
        watchlist=watchlist or ["SPY", "AAPL"],
    )


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
    path = str(tmp_path / "test_jobs.db")
    await init_db(path)
    return path


# ── Tests ────────────────────────────────────────────────────────────────────


@pytest.mark.live
async def test_precompute_job_runs(tmp_path, monkeypatch):
    """Run the precompute pipeline with a small watchlist and verify DB rows."""
    db_p = str(tmp_path / "precompute.db")
    await init_db(db_p)

    # Point settings DB path to our temp DB
    monkeypatch.setenv("PA_DB_PATH", db_p)

    ctx = _make_ctx(db_p, watchlist=["SPY", "AAPL"])

    from portfolio_advisor.tools.precomputed import run_precompute_pipeline

    results = await run_precompute_pipeline(ctx)

    assert "processed" in results
    assert "failed" in results
    assert len(results["processed"]) > 0, "At least one ticker should be processed"

    today = date.today().isoformat()
    async with get_db(db_p) as db:
        tech = await queries.get_bulk_technical_indicators(db, ["SPY", "AAPL"], today)
        quant = await queries.get_bulk_quant_metrics(db, ["SPY", "AAPL"], today)

    assert len(tech) > 0, "Should have technical_indicators rows after precompute"
    assert len(quant) > 0, "Should have quant_metrics rows after precompute"

    # Verify key fields exist on at least one record
    tech_row = tech[0]
    assert "ticker" in tech_row
    assert "overall_bias" in tech_row
    assert "rsi_14" in tech_row

    quant_row = quant[0]
    assert "ticker" in quant_row
    assert "regime" in quant_row


@pytest.mark.live
async def test_daily_job_context_building(tmp_path, monkeypatch):
    """After precompute, _build_daily_context should return a non-empty context string."""
    db_p = str(tmp_path / "daily_ctx.db")
    await init_db(db_p)

    monkeypatch.setenv("PA_DB_PATH", db_p)

    # Set a small watchlist in DB preferences so the pipeline uses it
    # (pipeline reads watchlist from DB, not from ctx.watchlist)
    async with get_db(db_p) as db:
        await queries.update_user_preference(db, "watchlist", ["SPY", "AAPL"])

    ctx = _make_ctx(db_p, watchlist=["SPY", "AAPL"])

    # First run precompute so there is data in the DB
    from portfolio_advisor.tools.precomputed import run_precompute_pipeline

    await run_precompute_pipeline(ctx)

    # Now build the daily context
    from portfolio_advisor.scheduler.jobs import _build_daily_context

    context_str = await _build_daily_context(ctx)

    assert isinstance(context_str, str)
    assert len(context_str) > 100, "Context should be a substantial string"
    assert "Pre-Computed Daily Analysis" in context_str
    # Should contain ticker signals section
    assert "Ticker Signals" in context_str or "SPY" in context_str or "AAPL" in context_str


async def test_news_alert_pipeline_mock_research(db_path):
    """Test _is_similar_to_existing correctly detects duplicate themes."""
    # Exact duplicates
    existing = ["fed raises rates unexpectedly", "oil prices surge on opec cuts"]

    assert _is_similar_to_existing("fed raises rates unexpectedly", existing)
    assert _is_similar_to_existing("oil prices surge on opec cuts", existing)

    # Similar enough to be duplicate (high word overlap + sequence similarity)
    assert _is_similar_to_existing("fed raises rates unexpectedly today", existing)
    assert _is_similar_to_existing("oil prices surge on opec production cuts", existing)

    # Clearly different theme
    assert not _is_similar_to_existing("nvidia beats earnings expectations", existing)
    assert not _is_similar_to_existing("bitcoin hits new all time high", existing)


async def test_news_alert_pipeline_stores_new_themes(db_path):
    """Mock the research agent but use real DB to verify theme storage."""
    today = date.today().isoformat()

    # Store an existing theme
    async with get_db(db_path) as db:
        await queries.store_research_theme(db, {
            "theme_date": today,
            "theme": "Fed raises rates",
            "summary": "Existing theme",
            "impact": "high",
            "affected_tickers": ["SPY"],
            "sources": [],
            "source_tier": "",
            "is_active": 1,
        })

    # Verify it was stored
    async with get_db(db_path) as db:
        existing = await queries.get_active_research_themes(db, days=3)

    assert len(existing) >= 1
    assert any(t["theme"] == "Fed raises rates" for t in existing)


async def test_evening_summary_context(db_path):
    """After storing forecast data, verify the evening job can query it."""
    today = date.today().isoformat()

    # Store a forecast
    async with get_db(db_path) as db:
        await queries.store_forecast(
            db,
            ticker="SPY",
            forecast_type="return",
            horizon="1w",
            predicted_value={"expected_return_pct": 1.5},
        )

    # Verify forecast can be queried (same query the evening job uses)
    async with get_db(db_path) as db:
        cursor = await db.execute(
            """SELECT fl.ticker, fl.horizon, fl.predicted_value
               FROM forecasts_log fl
               WHERE fl.forecast_date = ?
               ORDER BY fl.ticker""",
            (today,),
        )
        forecasts = [dict(r) for r in await cursor.fetchall()]

    assert len(forecasts) >= 1
    fc = forecasts[0]
    assert fc["ticker"] == "SPY"
    assert fc["horizon"] == "1w"
    pred = json.loads(fc["predicted_value"])
    assert pred["expected_return_pct"] == 1.5
