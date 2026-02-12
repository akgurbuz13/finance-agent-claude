"""Tests for fundamentals DB operations and schema migration."""

import aiosqlite
import pytest

from portfolio_advisor.db import queries
from portfolio_advisor.db.connection import init_db, get_db


@pytest.fixture
async def db_path(tmp_path):
    """Create a temporary database with schema."""
    path = str(tmp_path / "test.db")
    await init_db(path)
    return path


class TestFundamentalsDB:
    """Tests for fundamentals table CRUD."""

    async def test_store_and_get_fundamentals(self, db_path):
        data = {
            "ticker": "AAPL",
            "fetch_date": "2024-01-15",
            "pe_ratio": 28.5,
            "forward_pe": 26.0,
            "pb_ratio": 45.2,
            "ps_ratio": 7.8,
            "ev_ebitda": 22.1,
            "roe": 1.71,
            "roa": 0.28,
            "profit_margin": 0.265,
            "operating_margin": 0.30,
            "debt_to_equity": 1.87,
            "current_ratio": 0.99,
            "revenue_growth_yoy": 0.02,
            "earnings_growth_yoy": -0.01,
            "dividend_yield": 0.005,
            "market_cap": 2.9e12,
            "sector": "Technology",
            "industry": "Consumer Electronics",
            "analyst_rating": "Strong Buy",
            "analyst_target_price": 220.0,
            "source": "massive",
        }

        async with get_db(db_path) as db:
            await queries.store_fundamentals(db, data)
            result = await queries.get_fundamentals(db, "AAPL")

        assert result is not None
        assert result["ticker"] == "AAPL"
        assert result["pe_ratio"] == 28.5
        assert result["sector"] == "Technology"
        assert result["analyst_rating"] == "Strong Buy"

    async def test_get_fundamentals_returns_none_for_unknown(self, db_path):
        async with get_db(db_path) as db:
            result = await queries.get_fundamentals(db, "UNKNOWN")
        assert result is None

    async def test_upsert_updates_existing(self, db_path):
        data1 = {
            "ticker": "MSFT",
            "fetch_date": "2024-01-15",
            "pe_ratio": 35.0,
            "source": "massive",
        }
        data2 = {
            "ticker": "MSFT",
            "fetch_date": "2024-01-15",
            "pe_ratio": 36.0,  # updated
            "source": "massive",
        }

        async with get_db(db_path) as db:
            await queries.store_fundamentals(db, data1)
            await queries.store_fundamentals(db, data2)
            result = await queries.get_fundamentals(db, "MSFT")

        assert result["pe_ratio"] == 36.0

    async def test_get_fundamentals_comparison(self, db_path):
        for ticker, pe in [("AAPL", 28.5), ("MSFT", 35.0), ("GOOGL", 22.0)]:
            async with get_db(db_path) as db:
                await queries.store_fundamentals(db, {
                    "ticker": ticker,
                    "fetch_date": "2024-01-15",
                    "pe_ratio": pe,
                    "source": "test",
                })

        async with get_db(db_path) as db:
            results = await queries.get_fundamentals_comparison(
                db, ["AAPL", "MSFT", "GOOGL"]
            )

        assert len(results) == 3
        tickers = [r["ticker"] for r in results]
        assert "AAPL" in tickers
        assert "MSFT" in tickers

    async def test_get_latest_fundamentals_across_dates(self, db_path):
        """get_fundamentals should return the most recent entry."""
        async with get_db(db_path) as db:
            await queries.store_fundamentals(db, {
                "ticker": "AAPL",
                "fetch_date": "2024-01-10",
                "pe_ratio": 27.0,
                "source": "test",
            })
            await queries.store_fundamentals(db, {
                "ticker": "AAPL",
                "fetch_date": "2024-01-15",
                "pe_ratio": 29.0,
                "source": "test",
            })
            result = await queries.get_fundamentals(db, "AAPL")

        assert result["pe_ratio"] == 29.0
        assert result["fetch_date"] == "2024-01-15"


class TestSchemaV4Migration:
    """Tests for v4 schema migration (WAL mode, new columns)."""

    async def test_wal_mode_enabled(self, db_path):
        async with aiosqlite.connect(db_path) as db:
            cursor = await db.execute("PRAGMA journal_mode")
            row = await cursor.fetchone()
        assert row[0] == "wal"

    async def test_busy_timeout_set_via_get_db(self, db_path):
        """busy_timeout is set by get_db() context manager, not raw connect."""
        async with get_db(db_path) as db:
            cursor = await db.execute("PRAGMA busy_timeout")
            row = await cursor.fetchone()
        assert row[0] == 30000

    async def test_fundamentals_table_exists(self, db_path):
        async with aiosqlite.connect(db_path) as db:
            cursor = await db.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='fundamentals'"
            )
            row = await cursor.fetchone()
        assert row is not None

    async def test_sentiment_metrics_table_exists(self, db_path):
        async with aiosqlite.connect(db_path) as db:
            cursor = await db.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='sentiment_metrics'"
            )
            row = await cursor.fetchone()
        assert row is not None

    async def test_sentiment_metrics_crud(self, db_path):
        data = {
            "ticker": "GME",
            "fetch_date": "2024-01-15",
            "short_interest": 50_000_000,
            "short_pct_float": 22.5,
            "days_to_cover": 6.3,
            "short_volume_ratio": 0.45,
            "source": "massive",
        }
        async with get_db(db_path) as db:
            await queries.store_sentiment_metrics(db, data)
            result = await queries.get_sentiment_metrics(db, "GME")

        assert result is not None
        assert result["ticker"] == "GME"
        assert result["short_pct_float"] == 22.5
        assert result["days_to_cover"] == 6.3

    async def test_sentiment_metrics_returns_none_for_unknown(self, db_path):
        async with get_db(db_path) as db:
            result = await queries.get_sentiment_metrics(db, "UNKNOWN")
        assert result is None

    async def test_bulk_sentiment_metrics(self, db_path):
        async with get_db(db_path) as db:
            for ticker, pct in [("GME", 22.5), ("AMC", 15.0)]:
                await queries.store_sentiment_metrics(db, {
                    "ticker": ticker,
                    "fetch_date": "2024-01-15",
                    "short_pct_float": pct,
                    "source": "test",
                })
            results = await queries.get_bulk_sentiment_metrics(db, ["GME", "AMC", "TSLA"])

        assert len(results) == 2
        tickers = [r["ticker"] for r in results]
        assert "GME" in tickers
        assert "AMC" in tickers

    async def test_sentiment_metrics_upsert(self, db_path):
        async with get_db(db_path) as db:
            await queries.store_sentiment_metrics(db, {
                "ticker": "GME",
                "fetch_date": "2024-01-15",
                "short_pct_float": 22.5,
                "source": "test",
            })
            await queries.store_sentiment_metrics(db, {
                "ticker": "GME",
                "fetch_date": "2024-01-15",
                "short_pct_float": 25.0,  # updated
                "source": "test",
            })
            result = await queries.get_sentiment_metrics(db, "GME")

        assert result["short_pct_float"] == 25.0

    async def test_v4_columns_exist_on_daily_risk_metrics(self, db_path):
        """v4 migration adds yield and source columns."""
        async with aiosqlite.connect(db_path) as db:
            cursor = await db.execute("PRAGMA table_info(daily_risk_metrics)")
            columns = {row[1] for row in await cursor.fetchall()}

        expected = {"yield_2y", "yield_5y", "yield_10y", "yield_30y",
                    "yield_curve_source", "vix_source", "credit_spread_source"}
        assert expected.issubset(columns), f"Missing columns: {expected - columns}"
