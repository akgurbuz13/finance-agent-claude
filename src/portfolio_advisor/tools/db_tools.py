"""Database tools for storing and retrieving briefs, reports, forecasts."""

from __future__ import annotations

import json

from agents import RunContextWrapper, function_tool

from portfolio_advisor.agents.context import AppContext
from portfolio_advisor.db.connection import get_db
from portfolio_advisor.db import queries


@function_tool
async def store_daily_brief(
    ctx: RunContextWrapper[AppContext],
    brief_json: str,
) -> str:
    """Store a DailyBrief to the database (daily_briefs + instrument_briefs tables)."""
    brief = json.loads(brief_json)
    async with get_db(ctx.context.db_path) as db:
        await queries.store_daily_brief(db, brief)
    return json.dumps({"status": "ok", "brief_date": brief.get("brief_date")})


@function_tool
async def retrieve_daily_briefs(
    ctx: RunContextWrapper[AppContext],
    start_date: str,
    end_date: str,
    ticker: str = "",
) -> str:
    """Retrieve daily briefs by date range, optionally filtered by ticker."""
    async with get_db(ctx.context.db_path) as db:
        rows = await queries.retrieve_daily_briefs(
            db, start_date, end_date, ticker=ticker or None
        )
    return json.dumps(rows, default=str)


@function_tool
async def store_weekly_report(
    ctx: RunContextWrapper[AppContext],
    report_json: str,
) -> str:
    """Store a WeeklyReport to the database."""
    report = json.loads(report_json)
    async with get_db(ctx.context.db_path) as db:
        await queries.store_weekly_report(db, report)
    return json.dumps({"status": "ok", "week_ending": report.get("week_ending")})


@function_tool
async def retrieve_weekly_reports(
    ctx: RunContextWrapper[AppContext],
    count: int = 4,
) -> str:
    """Retrieve the N most recent weekly reports."""
    async with get_db(ctx.context.db_path) as db:
        rows = await queries.retrieve_weekly_reports(db, count)
    return json.dumps(rows, default=str)


@function_tool
async def store_forecast(
    ctx: RunContextWrapper[AppContext],
    ticker: str,
    forecast_type: str,
    horizon: str,
    predicted_value_json: str,
) -> str:
    """Log a prediction for later evaluation."""
    predicted = json.loads(predicted_value_json)
    async with get_db(ctx.context.db_path) as db:
        await queries.store_forecast(db, ticker, forecast_type, horizon, predicted)
    return json.dumps({"status": "ok"})


@function_tool
async def query_forecasts_log(
    ctx: RunContextWrapper[AppContext],
    ticker: str = "",
    forecast_type: str = "",
    start_date: str = "",
    end_date: str = "",
) -> str:
    """Query historical forecasts from the log."""
    async with get_db(ctx.context.db_path) as db:
        rows = await queries.query_forecasts_log(
            db,
            ticker=ticker or None,
            forecast_type=forecast_type or None,
            start_date=start_date or None,
            end_date=end_date or None,
        )
    return json.dumps(rows, default=str)
