"""Telegram bot command handlers."""

from __future__ import annotations

import json
import logging

from telegram import Update
from telegram.ext import ContextTypes

from portfolio_advisor.config import get_settings
from portfolio_advisor.db.connection import get_db
from portfolio_advisor.db import queries
from portfolio_advisor.telegram_bot.formatters import (
    format_portfolio_table,
    format_preferences,
    format_status,
    format_usage,
    format_watchlist,
    truncate_for_telegram,
)

logger = logging.getLogger(__name__)


def _auth(func):
    """Decorator to check that the message is from the authorized chat."""
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE):
        settings = get_settings()
        if update.effective_chat.id != settings.telegram_chat_id:
            await update.message.reply_text("Unauthorized.")
            return
        return await func(update, context)
    return wrapper


@_auth
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Welcome message, initialize defaults, and start onboarding if needed."""
    settings = get_settings()
    from portfolio_advisor.db.connection import init_db
    await init_db(settings.db_path)

    # Set default watchlist
    async with get_db(settings.db_path) as db:
        prefs = await queries.get_user_preferences(db)
        if not prefs.get("watchlist"):
            await queries.update_user_preference(db, "watchlist", settings.default_watchlist)

        # Check onboarding state
        onboarding = await queries.get_onboarding_state(db)

    if settings.onboarding_enabled and (
        onboarding is None or onboarding.get("current_step") != "done"
    ):
        step = onboarding.get("current_step", "welcome") if onboarding else "welcome"
        if step == "welcome":
            await update.message.reply_text(
                "Welcome to Portfolio Advisor!\n\n"
                "I'm your 24/7 autonomous portfolio advisory system. "
                "Let me help you set up your investment profile — "
                "it takes about 2 minutes.\n\n"
                "Just type anything to begin the guided setup, "
                "or /help to see all commands.",
                parse_mode="Markdown",
            )
        else:
            await update.message.reply_text(
                f"Welcome back! Your setup is in progress (step: {step}).\n\n"
                "Just type anything to continue where you left off.",
                parse_mode="Markdown",
            )
        return

    await update.message.reply_text(
        "Welcome back to Portfolio Advisor!\n\n"
        "I'm your 24/7 autonomous portfolio advisory system. Here's what I do:\n\n"
        "**Daily** (7:00 UTC): Technical + quant analysis + news monitoring\n"
        "**Weekly** (Sun 18:00 UTC): Full portfolio recommendation report\n\n"
        "You can also chat with me anytime for:\n"
        "- Live market analysis\n"
        "- Portfolio questions\n"
        "- Research on any ticker\n\n"
        "Type /help for all commands.",
        parse_mode="Markdown",
    )


@_auth
async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """List all commands."""
    help_text = (
        "*Available Commands*\n\n"
        "/start - Welcome & initialize\n"
        "/status - System status\n"
        "/portfolio - Current allocations\n"
        "/prefs - Show preferences\n"
        "/set <key> <value> - Update preference\n"
        "/watchlist - Show watchlist\n"
        "/addticker TSLA GOOG - Add tickers\n"
        "/removeticker IWM - Remove tickers\n"
        "/confirm <ticker> <weight> - Confirm trade\n"
        "/brief - Latest daily brief\n"
        "/report - Latest weekly report\n"
        "/earnings - Upcoming & recent earnings\n"
        "/news - On-demand news research\n"
        "/usage - Token usage & cost\n"
        "/rundaily - Force daily run\n"
        "/runweekly - Force weekly run\n"
        "/help - This message\n\n"
        "_Or just type any question for live analysis!_"
    )
    await update.message.reply_text(help_text, parse_mode="Markdown")


@_auth
async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """System status."""
    settings = get_settings()
    async with get_db(settings.db_path) as db:
        # Last daily
        cursor = await db.execute(
            "SELECT brief_date FROM daily_briefs ORDER BY brief_date DESC LIMIT 1"
        )
        row = await cursor.fetchone()
        last_daily = dict(row)["brief_date"] if row else None

        # Last weekly
        cursor = await db.execute(
            "SELECT week_ending FROM weekly_reports ORDER BY week_ending DESC LIMIT 1"
        )
        row = await cursor.fetchone()
        last_weekly = dict(row)["week_ending"] if row else None

    status_text = format_status(last_daily, last_weekly, f"{settings.daily_run_hour}:00 UTC daily", f"{settings.weekly_run_day} {settings.weekly_run_hour}:00 UTC")
    try:
        await update.message.reply_text(status_text, parse_mode="Markdown")
    except Exception:
        await update.message.reply_text(status_text)


@_auth
async def cmd_portfolio(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show current portfolio."""
    settings = get_settings()
    async with get_db(settings.db_path) as db:
        positions = await queries.get_portfolio_state(db)
    total = sum(p["weight_pct"] for p in positions)
    cash = round(100.0 - total, 2)
    text = format_portfolio_table(positions, cash)
    try:
        await update.message.reply_text(text, parse_mode="Markdown")
    except Exception:
        await update.message.reply_text(text)


@_auth
async def cmd_prefs(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show user preferences."""
    settings = get_settings()
    async with get_db(settings.db_path) as db:
        prefs = await queries.get_user_preferences(db)
    text = format_preferences(prefs)
    try:
        await update.message.reply_text(text, parse_mode="Markdown")
    except Exception:
        await update.message.reply_text(text)


@_auth
async def cmd_set(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Update a preference. Usage: /set risk_tolerance aggressive"""
    args = context.args
    if not args or len(args) < 2:
        await update.message.reply_text("Usage: /set <key> <value>\nExample: /set risk_tolerance aggressive")
        return

    key = args[0]
    value = " ".join(args[1:])
    valid_keys = {
        "risk_tolerance", "time_horizon", "excluded_assets",
        "allowed_regions", "cash_target_pct", "max_position_pct",
        # v2 fields
        "investment_style", "rebalance_frequency",
        "max_crypto_pct", "min_bond_pct", "max_single_sector_pct",
        "preferred_sectors", "esg_filter", "dividend_preference",
        "tax_aware", "notification_level", "analysis_depth",
        "benchmark", "notes",
    }
    if key not in valid_keys:
        await update.message.reply_text(f"Invalid key. Valid keys: {', '.join(sorted(valid_keys))}")
        return

    settings = get_settings()
    async with get_db(settings.db_path) as db:
        parsed = value
        if key in ("cash_target_pct", "max_position_pct", "max_crypto_pct",
                    "min_bond_pct", "max_single_sector_pct"):
            parsed = float(value)
        elif key in ("excluded_assets", "allowed_regions", "preferred_sectors"):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                parsed = [v.strip() for v in value.split(",")]
        elif key in ("esg_filter", "tax_aware"):
            parsed = value.lower() in ("true", "1", "yes", "on")
        await queries.update_user_preference(db, key, parsed)

    await update.message.reply_text(f"Updated {key} = {value}")


@_auth
async def cmd_watchlist(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show watchlist."""
    settings = get_settings()
    async with get_db(settings.db_path) as db:
        prefs = await queries.get_user_preferences(db)
    watchlist = prefs.get("watchlist", [])
    text = format_watchlist(watchlist)
    try:
        await update.message.reply_text(text, parse_mode="Markdown")
    except Exception:
        await update.message.reply_text(text)


@_auth
async def cmd_addticker(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Add tickers to watchlist. Usage: /addticker TSLA GOOG"""
    if not context.args:
        await update.message.reply_text("Usage: /addticker TSLA GOOG")
        return

    tickers = [t.upper() for t in context.args]
    settings = get_settings()
    async with get_db(settings.db_path) as db:
        prefs = await queries.get_user_preferences(db)
        current = prefs.get("watchlist", [])
        updated = list(set(current + tickers))
        await queries.update_user_preference(db, "watchlist", updated)

    added = [t for t in tickers if t not in current]
    if added:
        await update.message.reply_text(f"Added: {', '.join(added)}\nWatchlist: {len(updated)} tickers")
    else:
        await update.message.reply_text("All tickers already in watchlist.")


@_auth
async def cmd_removeticker(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Remove tickers from watchlist. Usage: /removeticker IWM"""
    if not context.args:
        await update.message.reply_text("Usage: /removeticker IWM")
        return

    tickers = [t.upper() for t in context.args]
    settings = get_settings()
    async with get_db(settings.db_path) as db:
        prefs = await queries.get_user_preferences(db)
        current = prefs.get("watchlist", [])
        updated = [t for t in current if t not in tickers]
        await queries.update_user_preference(db, "watchlist", updated)

    removed = [t for t in tickers if t in current]
    if removed:
        await update.message.reply_text(f"Removed: {', '.join(removed)}\nWatchlist: {len(updated)} tickers")
    else:
        await update.message.reply_text("None of those tickers were in the watchlist.")


@_auth
async def cmd_confirm(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Confirm a trade. Usage: /confirm AAPL 10.5"""
    if not context.args or len(context.args) < 2:
        await update.message.reply_text("Usage: /confirm <ticker> <weight_pct>\nExample: /confirm AAPL 10.5")
        return

    ticker = context.args[0].upper()
    try:
        weight = float(context.args[1])
    except ValueError:
        await update.message.reply_text("Weight must be a number (e.g., 10.5)")
        return

    settings = get_settings()
    async with get_db(settings.db_path) as db:
        if weight <= 0:
            await queries.remove_portfolio_position(db, ticker)
            action = "Removed"
        else:
            asset_class = _classify(ticker)
            await queries.update_portfolio_position(db, ticker, weight, asset_class)
            action = "Updated"

            # Auto-sync: ensure portfolio tickers are on the watchlist
            prefs = await queries.get_user_preferences(db)
            watchlist = prefs.get("watchlist", [])
            if ticker not in [t.upper() for t in watchlist]:
                watchlist.append(ticker)
                await queries.update_user_preference(db, "watchlist", watchlist)

        await queries.snapshot_portfolio(db, trigger="user_confirmed")

    await update.message.reply_text(f"{action} {ticker} → {weight}%")


@_auth
async def cmd_brief(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show latest daily brief."""
    settings = get_settings()
    async with get_db(settings.db_path) as db:
        cursor = await db.execute(
            "SELECT telegram_summary FROM daily_briefs ORDER BY brief_date DESC LIMIT 1"
        )
        row = await cursor.fetchone()

    if row:
        text = dict(row)["telegram_summary"] or "No summary available."
        text = truncate_for_telegram(text)
        try:
            await update.message.reply_text(text, parse_mode="Markdown")
        except Exception:
            await update.message.reply_text(text)
    else:
        await update.message.reply_text("No daily briefs yet. Run /rundaily to generate one.")


@_auth
async def cmd_report(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show latest weekly report."""
    settings = get_settings()
    async with get_db(settings.db_path) as db:
        rows = await queries.retrieve_weekly_reports(db, count=1)

    if rows:
        content = rows[0]
        try:
            report_data = json.loads(content.get("content_json", "{}"))
            text = report_data.get("telegram_summary", content.get("executive_summary", "No summary."))
        except (json.JSONDecodeError, TypeError):
            text = content.get("executive_summary", "No summary available.")
        text = truncate_for_telegram(text)
        try:
            await update.message.reply_text(text, parse_mode="Markdown")
        except Exception:
            await update.message.reply_text(text)
    else:
        await update.message.reply_text("No weekly reports yet. Run /runweekly to generate one.")


@_auth
async def cmd_usage(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show token usage."""
    settings = get_settings()
    async with get_db(settings.db_path) as db:
        summary = await queries.get_token_usage_summary(db, days=30)
        daily_used = await queries.get_daily_token_usage(db)

    summary["today_tokens_used"] = daily_used
    summary["today_budget_remaining"] = max(0, settings.daily_token_budget - daily_used)
    summary["period_days"] = 30
    text = format_usage(summary)
    try:
        await update.message.reply_text(text, parse_mode="Markdown")
    except Exception:
        await update.message.reply_text(text)


@_auth
async def cmd_rundaily(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Force a daily run now."""
    await update.message.reply_text("Starting daily analysis run... This may take a few minutes.")
    from portfolio_advisor.scheduler.jobs import daily_job
    try:
        await daily_job()
        await update.message.reply_text("Daily run completed! Use /brief to see the results.")
    except Exception as e:
        logger.exception(f"Manual daily run failed: {e}")
        await update.message.reply_text(f"Daily run failed: {str(e)[:300]}")


@_auth
async def cmd_runweekly(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Force a weekly run now."""
    await update.message.reply_text("Starting weekly analysis run... This may take several minutes.")
    from portfolio_advisor.scheduler.jobs import weekly_job
    try:
        await weekly_job()
        await update.message.reply_text("Weekly run completed! Use /report to see the results.")
    except Exception as e:
        logger.exception(f"Manual weekly run failed: {e}")
        await update.message.reply_text(f"Weekly run failed: {str(e)[:300]}")


@_auth
async def cmd_news(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Trigger on-demand news research for the watchlist."""
    await update.message.reply_text("Searching for latest news... This may take a minute.")

    try:
        from portfolio_advisor.scheduler.alerts import run_news_alert_pipeline
        from portfolio_advisor.scheduler.jobs import _build_context

        ctx = await _build_context()
        result = await run_news_alert_pipeline(ctx)

        new_count = result.get("new_themes", 0)
        total = result.get("total_themes", 0)
        alerts = result.get("alerts_sent", 0)

        summary_parts = [f"Found {total} themes ({new_count} new)"]
        if alerts:
            summary_parts.append(f"{alerts} high-impact alerts sent")
        if result.get("error"):
            summary_parts.append(f"Error: {result['error'][:200]}")

        await update.message.reply_text(
            f"**News Research Complete**\n\n{'. '.join(summary_parts)}.",
            parse_mode="Markdown",
        )
    except Exception as e:
        logger.exception(f"On-demand news check failed: {e}")
        await update.message.reply_text(f"News check failed: {str(e)[:300]}")


@_auth
async def cmd_earnings(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show upcoming and recent earnings for watchlist tickers."""
    settings = get_settings()
    async with get_db(settings.db_path) as db:
        upcoming = await queries.get_upcoming_earnings(db, days=14)
        recent = await queries.get_recent_earnings(db, days=7)

    lines: list[str] = []

    if upcoming:
        lines.append("**Upcoming Earnings (next 14 days)**\n")
        for e in upcoming:
            time_str = f" ({e['earnings_time']})" if e.get("earnings_time", "unknown") != "unknown" else ""
            eps_str = f" — Est EPS: ${e['eps_estimate']:.2f}" if e.get("eps_estimate") else ""
            rev_str = ""
            if e.get("revenue_estimate") and e["revenue_estimate"] > 1e6:
                rev_str = f", Est Rev: ${e['revenue_estimate'] / 1e9:.1f}B"
            lines.append(f"  {e['ticker']}: {e['earnings_date']}{time_str}{eps_str}{rev_str}")
    else:
        lines.append("No upcoming earnings in the next 14 days.")

    if recent:
        lines.append("\n**Recent Reports (last 7 days)**\n")
        for e in recent:
            surprise = e.get("eps_surprise_pct", 0)
            if surprise > 0:
                verdict = f"beat by {surprise:.1f}%"
            elif surprise < 0:
                verdict = f"miss by {abs(surprise):.1f}%"
            else:
                verdict = "in-line"
            actual_str = f"${e['eps_actual']:.2f}" if e.get("eps_actual") is not None else "?"
            est_str = f"${e['eps_estimate']:.2f}" if e.get("eps_estimate") is not None else "?"
            lines.append(f"  {e['ticker']}: EPS {actual_str} vs {est_str} est ({verdict})")

    text = "\n".join(lines) if lines else "No earnings data available yet."
    try:
        await update.message.reply_text(text, parse_mode="Markdown")
    except Exception:
        await update.message.reply_text(text)


def _classify(ticker: str) -> str:
    ticker = ticker.upper()
    if ticker in {"BTC", "ETH", "SOL", "AVAX"}:
        return "crypto"
    if ticker in {"TLT", "IEF", "HYG", "AGG", "BND", "LQD"}:
        return "bond"
    if ticker in {"GLD", "SLV", "USO", "DBA"}:
        return "commodity"
    return "equity"
