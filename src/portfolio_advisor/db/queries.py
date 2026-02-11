"""Typed CRUD functions for all tables."""

from __future__ import annotations

import json
from datetime import datetime

import aiosqlite


# ── User Preferences ──────────────────────────────────────────────────────────

async def get_user_preferences(db: aiosqlite.Connection) -> dict:
    cursor = await db.execute("SELECT * FROM user_preferences WHERE id = 1")
    row = await cursor.fetchone()
    if row is None:
        return {}
    d = dict(row)
    for key in ("excluded_assets", "allowed_regions", "watchlist"):
        if isinstance(d.get(key), str):
            d[key] = json.loads(d[key])
    return d


async def update_user_preference(db: aiosqlite.Connection, key: str, value) -> None:
    if key in ("excluded_assets", "allowed_regions", "watchlist"):
        value = json.dumps(value)
    await db.execute(
        f"UPDATE user_preferences SET {key} = ?, updated_at = ? WHERE id = 1",
        (value, datetime.utcnow().isoformat()),
    )
    await db.commit()


# ── Portfolio State ───────────────────────────────────────────────────────────

async def get_portfolio_state(db: aiosqlite.Connection) -> list[dict]:
    cursor = await db.execute("SELECT * FROM portfolio_state ORDER BY weight_pct DESC")
    rows = await cursor.fetchall()
    return [dict(r) for r in rows]


async def update_portfolio_position(
    db: aiosqlite.Connection, ticker: str, weight_pct: float, asset_class: str = "equity"
) -> None:
    now = datetime.utcnow().isoformat()
    await db.execute(
        """INSERT INTO portfolio_state (ticker, weight_pct, asset_class, updated_at)
           VALUES (?, ?, ?, ?)
           ON CONFLICT(ticker) DO UPDATE SET
               weight_pct = excluded.weight_pct,
               asset_class = excluded.asset_class,
               updated_at = excluded.updated_at""",
        (ticker, weight_pct, asset_class, now),
    )
    await db.commit()


async def remove_portfolio_position(db: aiosqlite.Connection, ticker: str) -> None:
    await db.execute("DELETE FROM portfolio_state WHERE ticker = ?", (ticker,))
    await db.commit()


async def snapshot_portfolio(
    db: aiosqlite.Connection, trigger: str = "manual"
) -> None:
    positions = await get_portfolio_state(db)
    await db.execute(
        "INSERT INTO portfolio_history (snapshot_date, state_json, trigger) VALUES (?, ?, ?)",
        (datetime.utcnow().strftime("%Y-%m-%d"), json.dumps(positions), trigger),
    )
    await db.commit()


async def get_portfolio_history(db: aiosqlite.Connection, days: int = 30) -> list[dict]:
    cursor = await db.execute(
        """SELECT * FROM portfolio_history
           WHERE snapshot_date >= date('now', ?)
           ORDER BY snapshot_date DESC""",
        (f"-{days} days",),
    )
    rows = await cursor.fetchall()
    return [dict(r) for r in rows]


# ── Daily Briefs ──────────────────────────────────────────────────────────────

async def store_daily_brief(db: aiosqlite.Connection, brief: dict) -> None:
    brief_date = brief["brief_date"]
    content_json = json.dumps(brief)

    await db.execute(
        """INSERT INTO daily_briefs (brief_date, content_json, market_summary, telegram_summary)
           VALUES (?, ?, ?, ?)
           ON CONFLICT(brief_date) DO UPDATE SET
               content_json = excluded.content_json,
               market_summary = excluded.market_summary,
               telegram_summary = excluded.telegram_summary""",
        (brief_date, content_json, brief.get("market_summary"), brief.get("telegram_summary")),
    )

    # Denormalize instrument briefs
    for inst in brief.get("instruments", []):
        await db.execute(
            """INSERT INTO instrument_briefs
               (brief_date, ticker, signal, confidence, what_happened,
                why_it_matters, technical_json, quant_json, sources)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(brief_date, ticker) DO UPDATE SET
                   signal = excluded.signal,
                   confidence = excluded.confidence,
                   what_happened = excluded.what_happened,
                   why_it_matters = excluded.why_it_matters,
                   technical_json = excluded.technical_json,
                   quant_json = excluded.quant_json,
                   sources = excluded.sources""",
            (
                brief_date,
                inst["ticker"],
                inst.get("signal"),
                inst.get("confidence"),
                inst.get("what_happened"),
                inst.get("why_it_matters"),
                json.dumps(inst.get("technical_json", {})),
                json.dumps(inst.get("quant_json", {})),
                json.dumps(inst.get("sources", [])),
            ),
        )
    await db.commit()


async def retrieve_daily_briefs(
    db: aiosqlite.Connection,
    start_date: str,
    end_date: str,
    ticker: str | None = None,
) -> list[dict]:
    if ticker:
        cursor = await db.execute(
            """SELECT * FROM instrument_briefs
               WHERE brief_date BETWEEN ? AND ? AND ticker = ?
               ORDER BY brief_date DESC""",
            (start_date, end_date, ticker),
        )
    else:
        cursor = await db.execute(
            """SELECT * FROM daily_briefs
               WHERE brief_date BETWEEN ? AND ?
               ORDER BY brief_date DESC""",
            (start_date, end_date),
        )
    rows = await cursor.fetchall()
    return [dict(r) for r in rows]


# ── Weekly Reports ────────────────────────────────────────────────────────────

async def store_weekly_report(db: aiosqlite.Connection, report: dict) -> None:
    week_ending = report["week_ending"]
    await db.execute(
        """INSERT INTO weekly_reports
           (week_ending, content_json, executive_summary, allocations_json)
           VALUES (?, ?, ?, ?)
           ON CONFLICT(week_ending) DO UPDATE SET
               content_json = excluded.content_json,
               executive_summary = excluded.executive_summary,
               allocations_json = excluded.allocations_json""",
        (
            week_ending,
            json.dumps(report),
            report.get("executive_summary"),
            json.dumps(report.get("allocations", [])),
        ),
    )
    await db.commit()


async def retrieve_weekly_reports(db: aiosqlite.Connection, count: int = 4) -> list[dict]:
    cursor = await db.execute(
        "SELECT * FROM weekly_reports ORDER BY week_ending DESC LIMIT ?",
        (count,),
    )
    rows = await cursor.fetchall()
    return [dict(r) for r in rows]


# ── Price Cache ───────────────────────────────────────────────────────────────

async def cache_prices(db: aiosqlite.Connection, ticker: str, bars: list[dict], source: str = "yfinance") -> None:
    for bar in bars:
        await db.execute(
            """INSERT INTO price_cache (ticker, bar_date, open, high, low, close, volume, source)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(ticker, bar_date) DO NOTHING""",
            (ticker, bar["date"], bar["open"], bar["high"], bar["low"], bar["close"], bar["volume"], source),
        )
    await db.commit()


async def get_cached_prices(
    db: aiosqlite.Connection, ticker: str, start_date: str, end_date: str
) -> list[dict]:
    cursor = await db.execute(
        """SELECT * FROM price_cache
           WHERE ticker = ? AND bar_date BETWEEN ? AND ?
           ORDER BY bar_date""",
        (ticker, start_date, end_date),
    )
    rows = await cursor.fetchall()
    return [dict(r) for r in rows]


# ── Forecasts Log ────────────────────────────────────────────────────────────

async def store_forecast(
    db: aiosqlite.Connection,
    ticker: str,
    forecast_type: str,
    horizon: str,
    predicted_value: dict,
) -> None:
    await db.execute(
        """INSERT INTO forecasts_log (forecast_date, ticker, forecast_type, horizon, predicted_value)
           VALUES (date('now'), ?, ?, ?, ?)""",
        (ticker, forecast_type, horizon, json.dumps(predicted_value)),
    )
    await db.commit()


async def query_forecasts_log(
    db: aiosqlite.Connection,
    ticker: str | None = None,
    forecast_type: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> list[dict]:
    query = "SELECT * FROM forecasts_log WHERE 1=1"
    params: list = []
    if ticker:
        query += " AND ticker = ?"
        params.append(ticker)
    if forecast_type:
        query += " AND forecast_type = ?"
        params.append(forecast_type)
    if start_date:
        query += " AND forecast_date >= ?"
        params.append(start_date)
    if end_date:
        query += " AND forecast_date <= ?"
        params.append(end_date)
    query += " ORDER BY forecast_date DESC"

    cursor = await db.execute(query, params)
    rows = await cursor.fetchall()
    return [dict(r) for r in rows]


# ── Token Usage ───────────────────────────────────────────────────────────────

async def log_token_usage(
    db: aiosqlite.Connection,
    model: str,
    run_type: str,
    input_tokens: int,
    output_tokens: int,
    estimated_cost_usd: float,
) -> None:
    await db.execute(
        """INSERT INTO token_usage (usage_date, model, run_type, input_tokens, output_tokens, estimated_cost_usd)
           VALUES (date('now'), ?, ?, ?, ?, ?)""",
        (model, run_type, input_tokens, output_tokens, estimated_cost_usd),
    )
    await db.commit()


async def get_token_usage_summary(
    db: aiosqlite.Connection, days: int = 30
) -> dict:
    cursor = await db.execute(
        """SELECT
               SUM(input_tokens) as total_input,
               SUM(output_tokens) as total_output,
               SUM(estimated_cost_usd) as total_cost,
               COUNT(*) as call_count
           FROM token_usage
           WHERE usage_date >= date('now', ?)""",
        (f"-{days} days",),
    )
    row = await cursor.fetchone()
    return dict(row) if row else {}


async def get_daily_token_usage(db: aiosqlite.Connection) -> int:
    cursor = await db.execute(
        "SELECT COALESCE(SUM(input_tokens + output_tokens), 0) FROM token_usage WHERE usage_date = date('now')"
    )
    (total,) = await cursor.fetchone()
    return total
