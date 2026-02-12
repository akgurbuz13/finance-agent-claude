"""Async SQLite connection manager."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

import aiosqlite

from portfolio_advisor.db.schema import SCHEMA_SQL

logger = logging.getLogger(__name__)

# New columns to add to user_preferences for v2 migration.
# Each entry: (column_name, column_type_and_default)
_V2_PREFERENCE_COLUMNS = [
    ("investment_style", "TEXT NOT NULL DEFAULT 'passive'"),
    ("rebalance_frequency", "TEXT NOT NULL DEFAULT 'monthly'"),
    ("max_crypto_pct", "REAL NOT NULL DEFAULT 15.0"),
    ("min_bond_pct", "REAL NOT NULL DEFAULT 5.0"),
    ("max_single_sector_pct", "REAL NOT NULL DEFAULT 40.0"),
    ("preferred_sectors", "TEXT NOT NULL DEFAULT '[]'"),
    ("esg_filter", "INTEGER NOT NULL DEFAULT 0"),
    ("dividend_preference", "TEXT NOT NULL DEFAULT 'neutral'"),
    ("tax_aware", "INTEGER NOT NULL DEFAULT 0"),
    ("notification_level", "TEXT NOT NULL DEFAULT 'medium'"),
    ("analysis_depth", "TEXT NOT NULL DEFAULT 'detailed'"),
    ("benchmark", "TEXT NOT NULL DEFAULT 'SPY'"),
    ("notes", "TEXT NOT NULL DEFAULT ''"),
]


async def _get_table_columns(db: aiosqlite.Connection, table: str) -> set[str]:
    """Return the set of column names for the given table."""
    cursor = await db.execute(f"PRAGMA table_info({table})")
    rows = await cursor.fetchall()
    return {row[1] for row in rows}


async def _migrate_preferences(db: aiosqlite.Connection) -> None:
    """Add v2 preference columns if they don't exist yet."""
    existing = await _get_table_columns(db, "user_preferences")
    for col_name, col_def in _V2_PREFERENCE_COLUMNS:
        if col_name not in existing:
            sql = f"ALTER TABLE user_preferences ADD COLUMN {col_name} {col_def}"
            await db.execute(sql)
            logger.info(f"Migrated user_preferences: added column {col_name}")
    await db.commit()


async def _migrate_v4(db: aiosqlite.Connection) -> None:
    """Add v4 columns to daily_risk_metrics for actual macro data sources."""
    v4_columns = [
        ("yield_2y", "REAL"),
        ("yield_5y", "REAL"),
        ("yield_10y", "REAL"),
        ("yield_30y", "REAL"),
        ("yield_curve_source", "TEXT DEFAULT 'etf_proxy'"),
        ("vix_source", "TEXT DEFAULT 'spy_proxy'"),
        ("credit_spread_source", "TEXT DEFAULT 'etf_proxy'"),
    ]
    try:
        existing = await _get_table_columns(db, "daily_risk_metrics")
        for col_name, col_def in v4_columns:
            if col_name not in existing:
                sql = f"ALTER TABLE daily_risk_metrics ADD COLUMN {col_name} {col_def}"
                await db.execute(sql)
                logger.info(f"v4 migration: added column {col_name} to daily_risk_metrics")
        await db.commit()
    except Exception:
        pass  # Table may not exist yet


async def _migrate_v3_snapshot_hour(db: aiosqlite.Connection) -> None:
    """Add snapshot_hour column and recreate UNIQUE constraints for v3.

    SQLite can't ALTER UNIQUE constraints, so we recreate affected tables
    when the snapshot_hour column is missing. Pre-computed data is ephemeral
    and will be repopulated on the next pipeline run.
    """
    tables_to_migrate = [
        (
            "technical_indicators",
            "snapshot_hour",
        ),
        (
            "quant_metrics",
            "snapshot_hour",
        ),
        (
            "daily_risk_metrics",
            "snapshot_hour",
        ),
    ]
    for table, col in tables_to_migrate:
        existing = await _get_table_columns(db, table)
        if col not in existing:
            # Drop and let SCHEMA_SQL recreate with new constraint
            await db.execute(f"DROP TABLE IF EXISTS {table}")
            logger.info(f"Migrated {table}: recreated with {col} column")

    await db.commit()


async def init_db(db_path: str) -> None:
    """Create tables if they don't exist and seed defaults."""
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    async with aiosqlite.connect(db_path) as db:
        # Enable WAL mode and busy timeout for concurrent access
        await db.execute("PRAGMA journal_mode=WAL")
        await db.execute("PRAGMA busy_timeout=30000")
        wal_mode = await db.execute("PRAGMA journal_mode")
        row = await wal_mode.fetchone()
        if row and row[0] == "wal":
            logger.info("SQLite WAL mode enabled")
        else:
            logger.warning(f"SQLite WAL mode not active (got: {row})")

        # v3 migration must run BEFORE schema creation so tables are recreated
        try:
            await _migrate_v3_snapshot_hour(db)
        except Exception:
            pass  # Tables may not exist yet on fresh install

        await db.executescript(SCHEMA_SQL)

        # Seed default user_preferences row if missing
        cursor = await db.execute("SELECT COUNT(*) FROM user_preferences")
        (count,) = await cursor.fetchone()
        if count == 0:
            await db.execute("INSERT INTO user_preferences (id) VALUES (1)")
        await db.commit()

        # Run v2 migrations (safe to call repeatedly — idempotent)
        await _migrate_preferences(db)

        # Run v4 migrations (add macro source columns + fundamentals table)
        await _migrate_v4(db)


@asynccontextmanager
async def get_db(db_path: str) -> AsyncIterator[aiosqlite.Connection]:
    """Yield an aiosqlite connection with row_factory enabled, WAL mode, and busy timeout."""
    db = await aiosqlite.connect(db_path)
    db.row_factory = aiosqlite.Row
    await db.execute("PRAGMA journal_mode=WAL")
    await db.execute("PRAGMA busy_timeout=30000")
    try:
        yield db
    finally:
        await db.close()
