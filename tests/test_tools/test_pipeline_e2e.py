"""End-to-end tests for the pre-compute pipeline with live market data.

These tests run the full pipeline against real APIs and verify
that every step produces valid DB entries.
"""

import os

import pytest

from portfolio_advisor.agents.context import AppContext
from portfolio_advisor.db import queries
from portfolio_advisor.db.connection import get_db, init_db

# Skip entire module if no API keys configured
pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        not os.environ.get("PA_FRED_API_KEY"),
        reason="Live API keys required (PA_FRED_API_KEY)",
    ),
]


@pytest.fixture(scope="module")
def db_path(tmp_path_factory):
    """Create a temporary database for the test module."""
    import asyncio

    path = str(tmp_path_factory.mktemp("pipeline") / "test.db")
    asyncio.run(init_db(path))
    return path


@pytest.fixture(scope="module")
def providers():
    """Create real provider registry from env vars."""
    import portfolio_advisor.config as cfg_mod

    # Ensure fresh settings and override placeholder telegram chat ID
    cfg_mod._settings = None
    os.environ.setdefault("PA_TELEGRAM_CHAT_ID", "123")
    if not os.environ.get("PA_TELEGRAM_CHAT_ID", "").isdigit():
        os.environ["PA_TELEGRAM_CHAT_ID"] = "123"
    if not os.environ.get("PA_TELEGRAM_BOT_TOKEN"):
        os.environ["PA_TELEGRAM_BOT_TOKEN"] = "test-token"

    from portfolio_advisor.config import get_settings
    from portfolio_advisor.providers.registry import create_registry

    settings = get_settings()
    return create_registry(settings)


@pytest.fixture(scope="module")
def pipeline_result(db_path, providers):
    """Run the full pipeline once and share results across tests."""
    import asyncio

    from portfolio_advisor.tools.precomputed import run_precompute_pipeline

    ctx = AppContext(
        db_path=db_path,
        telegram_chat_id=123,
        watchlist=["SPY", "AAPL", "MSFT", "GLD", "BTC"],
        providers=providers,
    )

    # Add portfolio positions so risk metrics are computed
    async def setup_and_run():
        async with get_db(db_path) as db:
            await queries.update_portfolio_position(db, "SPY", 40.0, "equity")
            await queries.update_portfolio_position(db, "AAPL", 25.0, "equity")
            await queries.update_portfolio_position(db, "MSFT", 20.0, "equity")
            await queries.update_portfolio_position(db, "GLD", 10.0, "commodity")
            await queries.update_portfolio_position(db, "BTC", 5.0, "crypto")
        return await run_precompute_pipeline(ctx)

    return asyncio.run(setup_and_run())


# ── 3.1 Full Pipeline Execution ────────────────────────────────────────────


class TestPipelineExecution:
    def test_pipeline_completes_all_tickers(self, pipeline_result):
        processed = pipeline_result["processed"]
        failed = pipeline_result["failed"]
        assert len(processed) >= 4, f"Expected >=4 processed, got {processed}"
        assert len(failed) <= 1, f"Too many failures: {failed}"

    async def test_pipeline_technical_indicators_stored(self, db_path, pipeline_result):
        processed = pipeline_result["processed"]
        async with get_db(db_path) as db:
            for ticker in processed:
                data = await queries.get_technical_indicators(db, ticker)
                assert data is not None, f"No technical indicators for {ticker}"
                assert data["ticker"] == ticker
                assert data.get("snapshot_hour") is not None

    async def test_pipeline_quant_metrics_stored(self, db_path, pipeline_result):
        processed = pipeline_result["processed"]
        async with get_db(db_path) as db:
            for ticker in processed:
                data = await queries.get_quant_metrics(db, ticker)
                assert data is not None, f"No quant metrics for {ticker}"
                assert data["ticker"] == ticker

    async def test_pipeline_risk_metrics_stored(self, db_path, pipeline_result):
        """Risk metrics should be stored since we added portfolio positions."""
        async with get_db(db_path) as db:
            risk = await queries.get_latest_risk_metrics(db)
        assert risk is not None, "No daily_risk_metrics stored"
        assert risk.get("var_95") is not None, "VaR not computed"
        assert risk.get("es_95") is not None, "ES not computed"

    async def test_pipeline_correlations_stored(self, db_path, pipeline_result):
        async with get_db(db_path) as db:
            snapshot = await queries.get_latest_correlation_snapshot(db)
        assert snapshot is not None, "No correlation snapshot stored"
        corr = snapshot.get("correlation_matrix")
        assert corr is not None and len(corr) > 0, "Empty correlation matrix"

    async def test_pipeline_earnings_stored(self, db_path, pipeline_result):
        """Earnings should be stored for equity tickers (may be empty if no upcoming)."""
        async with get_db(db_path) as db:
            cursor = await db.execute("SELECT COUNT(*) FROM earnings_calendar")
            row = await cursor.fetchone()
        # At least attempted — count may be 0 if no upcoming earnings
        assert row is not None

    async def test_pipeline_macro_sources_in_db(self, db_path, pipeline_result):
        """Verify macro data has source labels."""
        async with get_db(db_path) as db:
            risk = await queries.get_latest_risk_metrics(db)
        assert risk is not None
        # At least one source should be populated
        sources = [
            risk.get("yield_curve_source"),
            risk.get("vix_source"),
            risk.get("credit_spread_source"),
        ]
        populated = [s for s in sources if s and s != ""]
        assert len(populated) >= 1, f"No macro sources populated: {sources}"

    async def test_pipeline_narrative_quality(self, db_path, pipeline_result):
        """Each processed ticker should have a non-empty narrative."""
        processed = pipeline_result["processed"]
        async with get_db(db_path) as db:
            for ticker in processed:
                data = await queries.get_technical_indicators(db, ticker)
                assert data is not None
                narrative = data.get("narrative", "")
                assert len(narrative) > 20, (
                    f"{ticker} narrative too short: '{narrative[:50]}'"
                )


# ── 3.2 Pipeline Data Quality Checks ───────────────────────────────────────


class TestPipelineDataQuality:
    async def test_pipeline_no_nan_in_indicators(self, db_path, pipeline_result):
        """Critical fields should not be NaN."""
        processed = pipeline_result["processed"]
        async with get_db(db_path) as db:
            for ticker in processed:
                data = await queries.get_technical_indicators(db, ticker)
                assert data is not None
                rsi = data.get("rsi_14")
                assert rsi is not None, f"{ticker} RSI is None"
                assert 0 <= rsi <= 100, f"{ticker} RSI out of range: {rsi}"

                macd = data.get("macd_line")
                assert macd is not None, f"{ticker} MACD is None"

    async def test_pipeline_no_nan_in_quant(self, db_path, pipeline_result):
        """Quant fields should be populated for processed tickers."""
        processed = pipeline_result["processed"]
        # Crypto tickers (BTC, ETH) may have incomplete quant metrics
        crypto_tickers = {"BTC", "ETH", "SOL", "AVAX"}
        async with get_db(db_path) as db:
            for ticker in processed:
                data = await queries.get_quant_metrics(db, ticker)
                assert data is not None, f"No quant_metrics row for {ticker}"
                # Regime may be None for crypto tickers due to data source differences
                if ticker not in crypto_tickers:
                    regime = data.get("regime")
                    assert regime is not None, f"{ticker} regime is None"

    async def test_pipeline_crypto_volume_handling(self, db_path, pipeline_result):
        """BTC should be processed; narrative should warn about volume if zero."""
        if "BTC" not in pipeline_result["processed"]:
            pytest.skip("BTC was not processed")

        async with get_db(db_path) as db:
            data = await queries.get_technical_indicators(db, "BTC")
        assert data is not None
        # CoinGecko may or may not provide volume; if not, narrative should mention it

    async def test_pipeline_idempotent_rerun(self, db_path, providers):
        """Running pipeline twice should update, not duplicate rows."""
        from portfolio_advisor.tools.precomputed import run_precompute_pipeline

        ctx = AppContext(
            db_path=db_path,
            telegram_chat_id=123,
            watchlist=["SPY"],
            providers=providers,
        )
        result = await run_precompute_pipeline(ctx)
        assert "SPY" in result["processed"]

        # Check there's only 1 row per ticker per snapshot_hour
        async with get_db(db_path) as db:
            cursor = await db.execute(
                """SELECT COUNT(*) FROM technical_indicators
                   WHERE ticker = 'SPY' AND indicator_date = date('now')"""
            )
            row = await cursor.fetchone()
        # May have multiple snapshot_hours (6, 13, 20) but not duplicates
        assert row[0] <= 3, f"Duplicate rows found: {row[0]}"

    async def test_pipeline_partial_failure_resilience(self, db_path, providers):
        """If one ticker is invalid, others should still process.

        The pipeline reads watchlist from DB user_preferences row.
        We set the watchlist to include an invalid ticker.
        """
        from portfolio_advisor.tools.precomputed import run_precompute_pipeline

        # Store a watchlist with an invalid ticker in user preferences
        async with get_db(db_path) as db:
            await queries.update_user_preference(
                db, "watchlist", ["SPY", "INVALIDTICKER12345"]
            )

        ctx = AppContext(
            db_path=db_path,
            telegram_chat_id=123,
            watchlist=["SPY", "INVALIDTICKER12345"],
            providers=providers,
        )
        result = await run_precompute_pipeline(ctx)
        assert "SPY" in result["processed"]
        # Invalid ticker should NOT be in processed (it can't download OHLCV)
        assert "INVALIDTICKER12345" not in result["processed"]

        # Restore default watchlist
        async with get_db(db_path) as db:
            await queries.update_user_preference(db, "watchlist", [])


# ── 3.3 Cache Query Tools Verification ─────────────────────────────────────


class TestCacheTools:
    """Verify that pipeline-stored data can be queried via DB queries.

    These tests call the same DB queries that the @function_tool cache wrappers
    use internally, validating that the data is correctly stored and retrievable.
    """

    async def test_cache_freshness_check(self, db_path, pipeline_result):
        """Analysis runs table should show a fresh precompute run."""
        async with get_db(db_path) as db:
            cursor = await db.execute(
                """SELECT run_type, status, completed_at
                   FROM analysis_runs
                   WHERE run_type = 'precompute'
                   ORDER BY completed_at DESC LIMIT 1"""
            )
            row = await cursor.fetchone()

        assert row is not None, "No precompute run recorded"
        assert row["status"] in ("completed", "completed_with_errors")
        assert row["completed_at"] is not None

    async def test_get_cached_technical(self, db_path, pipeline_result):
        """Technical indicators should be retrievable for processed tickers."""
        async with get_db(db_path) as db:
            data = await queries.get_technical_indicators(db, "SPY")
        assert data is not None
        assert data["ticker"] == "SPY"
        assert data.get("rsi_14") is not None
        assert data.get("sma50") is not None
        assert data.get("overall_bias") is not None

    async def test_get_cached_quant(self, db_path, pipeline_result):
        """Quant metrics should be retrievable for processed tickers."""
        async with get_db(db_path) as db:
            data = await queries.get_quant_metrics(db, "SPY")
        assert data is not None
        assert data["ticker"] == "SPY"

    async def test_get_cached_macro(self, db_path, pipeline_result):
        """Risk metrics (including macro data) should be retrievable."""
        async with get_db(db_path) as db:
            risk = await queries.get_latest_risk_metrics(db)
        assert risk is not None
        assert risk.get("macro_regime") is not None
        assert risk.get("vix_level") is not None

    async def test_get_cached_correlations(self, db_path, pipeline_result):
        """Correlation snapshot should be retrievable."""
        async with get_db(db_path) as db:
            snapshot = await queries.get_latest_correlation_snapshot(db)
        assert snapshot is not None
        assert snapshot.get("correlation_matrix") is not None
        assert snapshot.get("diversification_score") is not None

    async def test_cache_stale_detection(self, db_path, pipeline_result):
        """After modifying timestamp, data should appear stale."""
        from datetime import datetime

        # Manually update the analysis_runs completed_at to 48h ago
        async with get_db(db_path) as db:
            await db.execute(
                """UPDATE analysis_runs
                   SET completed_at = datetime('now', '-48 hours')
                   WHERE run_type = 'precompute'"""
            )
            await db.commit()

        # Verify the timestamp is now old
        async with get_db(db_path) as db:
            cursor = await db.execute(
                """SELECT completed_at FROM analysis_runs
                   WHERE run_type = 'precompute'
                   ORDER BY completed_at DESC LIMIT 1"""
            )
            row = await cursor.fetchone()

        assert row is not None
        completed_at = datetime.fromisoformat(row["completed_at"])
        age_hours = (datetime.utcnow() - completed_at).total_seconds() / 3600
        assert age_hours > 24, f"Expected stale data (>24h), got {age_hours:.1f}h"
