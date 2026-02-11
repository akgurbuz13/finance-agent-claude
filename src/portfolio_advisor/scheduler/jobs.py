"""Scheduled job implementations — daily and weekly pipelines."""

from __future__ import annotations

import json
import logging
from datetime import date, timedelta

from agents import Runner

from portfolio_advisor.agents.context import AppContext
from portfolio_advisor.agents.orchestrator import daily_orchestrator, weekly_orchestrator
from portfolio_advisor.config import get_settings
from portfolio_advisor.db.connection import get_db
from portfolio_advisor.db import queries
from portfolio_advisor.telegram_bot.bot import send_message

logger = logging.getLogger(__name__)


async def _build_context() -> AppContext:
    """Build the shared AppContext for a run."""
    settings = get_settings()
    async with get_db(settings.db_path) as db:
        prefs = await queries.get_user_preferences(db)
    watchlist = prefs.get("watchlist", settings.default_watchlist)

    return AppContext(
        db_path=settings.db_path,
        telegram_chat_id=settings.telegram_chat_id,
        run_date=date.today(),
        watchlist=watchlist,
        token_budget_remaining=settings.daily_token_budget,
        max_web_search_calls_daily=settings.max_web_searches_daily,
    )


async def daily_job() -> None:
    """Run the daily monitoring pipeline."""
    logger.info("Starting daily monitoring pipeline")
    ctx = await _build_context()

    try:
        today = date.today().isoformat()
        prompt = (
            f"Run the daily analysis pipeline for {today}.\n"
            f"Watchlist: {', '.join(ctx.watchlist)}\n"
            f"Analyze all tickers, synthesize a daily brief, store it, "
            f"and produce a Telegram-formatted summary."
        )

        result = await Runner.run(
            starting_agent=daily_orchestrator,
            input=prompt,
            context=ctx,
        )

        # Send the summary to Telegram
        output = result.final_output or "Daily analysis completed but no summary was generated."

        # Try to extract telegram_summary from stored brief
        settings = get_settings()
        async with get_db(settings.db_path) as db:
            briefs = await queries.retrieve_daily_briefs(db, today, today)
            if briefs:
                brief_data = briefs[0]
                telegram_text = brief_data.get("telegram_summary") or output
            else:
                telegram_text = output

        await send_message(telegram_text)
        logger.info("Daily pipeline completed successfully")

    except Exception as e:
        logger.exception(f"Daily pipeline failed: {e}")
        await send_message(f"Daily analysis failed: {str(e)[:500]}")


async def weekly_job() -> None:
    """Run the weekly portfolio recommendation pipeline."""
    logger.info("Starting weekly portfolio recommendation pipeline")
    ctx = await _build_context()
    ctx.token_budget_remaining = get_settings().weekly_token_budget

    try:
        today = date.today()
        week_start = (today - timedelta(days=7)).isoformat()
        week_end = today.isoformat()

        prompt = (
            f"Run the weekly portfolio recommendation pipeline for week ending {week_end}.\n"
            f"Review daily briefs from {week_start} to {week_end}.\n"
            f"Watchlist: {', '.join(ctx.watchlist)}\n"
            f"Produce a comprehensive investment committee memo with allocation "
            f"recommendations, risk assessment, and outlook. Store the report "
            f"and produce a Telegram-formatted summary."
        )

        result = await Runner.run(
            starting_agent=weekly_orchestrator,
            input=prompt,
            context=ctx,
        )

        output = result.final_output or "Weekly analysis completed but no summary was generated."

        # Try to extract telegram_summary from stored report
        settings = get_settings()
        async with get_db(settings.db_path) as db:
            reports = await queries.retrieve_weekly_reports(db, count=1)
            if reports:
                report_data = reports[0]
                try:
                    content = json.loads(report_data.get("content_json", "{}"))
                    telegram_text = content.get("telegram_summary") or output
                except (json.JSONDecodeError, TypeError):
                    telegram_text = output
            else:
                telegram_text = output

        await send_message(telegram_text)
        logger.info("Weekly pipeline completed successfully")

    except Exception as e:
        logger.exception(f"Weekly pipeline failed: {e}")
        await send_message(f"Weekly analysis failed: {str(e)[:500]}")
