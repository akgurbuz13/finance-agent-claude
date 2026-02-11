"""Telegram Application builder and handler registration."""

from __future__ import annotations

import logging

from telegram.ext import Application, CommandHandler, MessageHandler, filters

from portfolio_advisor.config import get_settings
from portfolio_advisor.telegram_bot.chat_handler import handle_chat_message
from portfolio_advisor.telegram_bot.commands import (
    cmd_addticker,
    cmd_brief,
    cmd_confirm,
    cmd_help,
    cmd_portfolio,
    cmd_prefs,
    cmd_removeticker,
    cmd_report,
    cmd_rundaily,
    cmd_runweekly,
    cmd_set,
    cmd_start,
    cmd_status,
    cmd_usage,
    cmd_watchlist,
)

logger = logging.getLogger(__name__)


def build_application() -> Application:
    """Build and configure the Telegram bot application."""
    settings = get_settings()
    app = Application.builder().token(settings.telegram_bot_token).build()

    # Register command handlers
    commands = {
        "start": cmd_start,
        "help": cmd_help,
        "status": cmd_status,
        "portfolio": cmd_portfolio,
        "prefs": cmd_prefs,
        "set": cmd_set,
        "watchlist": cmd_watchlist,
        "addticker": cmd_addticker,
        "removeticker": cmd_removeticker,
        "confirm": cmd_confirm,
        "brief": cmd_brief,
        "report": cmd_report,
        "usage": cmd_usage,
        "rundaily": cmd_rundaily,
        "runweekly": cmd_runweekly,
    }

    for name, handler in commands.items():
        app.add_handler(CommandHandler(name, handler))

    # Free-text → Chat Agent (must be last so commands are matched first)
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_chat_message))

    logger.info("Telegram bot handlers registered")
    return app


async def send_message(text: str, parse_mode: str = "Markdown") -> None:
    """Send a message to the configured chat (for scheduled jobs)."""
    settings = get_settings()
    from telegram import Bot

    bot = Bot(token=settings.telegram_bot_token)
    try:
        await bot.send_message(
            chat_id=settings.telegram_chat_id,
            text=text[:4096],
            parse_mode=parse_mode,
        )
    except Exception:
        # Fallback without parse_mode if markdown fails
        await bot.send_message(
            chat_id=settings.telegram_chat_id,
            text=text[:4096],
        )
