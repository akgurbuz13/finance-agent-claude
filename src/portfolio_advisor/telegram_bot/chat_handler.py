"""Free-text message handler — routes to the Chat Agent."""

from __future__ import annotations

import json
import logging
from datetime import date

from agents import Runner
from telegram import Update
from telegram.ext import ContextTypes

from portfolio_advisor.agents.chat import chat_agent
from portfolio_advisor.agents.context import AppContext
from portfolio_advisor.config import get_settings
from portfolio_advisor.telegram_bot.formatters import truncate_for_telegram

logger = logging.getLogger(__name__)


async def handle_chat_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Route free-text messages to the Chat Agent."""
    if not update.message or not update.message.text:
        return

    settings = get_settings()

    # Auth check
    if update.effective_chat.id != settings.telegram_chat_id:
        await update.message.reply_text("Unauthorized.")
        return

    user_text = update.message.text
    logger.info(f"Chat message from {update.effective_user.id}: {user_text[:100]}")

    # Send typing indicator
    await update.message.chat.send_action("typing")

    app_context = AppContext(
        db_path=settings.db_path,
        telegram_chat_id=settings.telegram_chat_id,
        run_date=date.today(),
        watchlist=settings.default_watchlist,
        token_budget_remaining=settings.daily_token_budget,
        max_web_search_calls_daily=settings.max_web_searches_daily,
    )

    try:
        # Build conversation history from context if available
        chat_history = context.user_data.get("chat_history", [])

        # Add user message to history
        messages = list(chat_history)
        messages.append({"role": "user", "content": user_text})

        result = await Runner.run(
            starting_agent=chat_agent,
            input=messages,
            context=app_context,
        )

        response_text = result.final_output or "I couldn't generate a response. Please try again."

        # Store conversation history (keep last 20 messages)
        chat_history.append({"role": "user", "content": user_text})
        chat_history.append({"role": "assistant", "content": response_text})
        context.user_data["chat_history"] = chat_history[-20:]

        # Send response, handling Telegram's message length limit
        response_text = truncate_for_telegram(response_text)

        # Try MarkdownV2 first, fall back to plain text
        try:
            await update.message.reply_text(response_text, parse_mode="MarkdownV2")
        except Exception:
            # Fallback: strip markdown and send plain
            await update.message.reply_text(response_text)

    except Exception as e:
        logger.exception(f"Chat agent error: {e}")
        await update.message.reply_text(
            f"Sorry, I encountered an error processing your request. Please try again.\n\nError: {str(e)[:200]}"
        )
