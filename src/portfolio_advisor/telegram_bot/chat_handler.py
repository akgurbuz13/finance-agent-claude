"""Free-text message handler — routes to the Chat Agent or Onboarding Agent."""

from __future__ import annotations

import logging
from datetime import date

from agents import Runner
from telegram import Update
from telegram.ext import ContextTypes

from portfolio_advisor.agents.context import AppContext
from portfolio_advisor.config import get_settings
from portfolio_advisor.db import queries
from portfolio_advisor.db.connection import get_db
from portfolio_advisor.telegram_bot.formatters import truncate_for_telegram

logger = logging.getLogger(__name__)


async def handle_chat_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Route free-text messages to the Chat Agent or Onboarding Agent."""
    if not update.message or not update.message.text:
        return

    settings = get_settings()

    # Auth check
    chat_id = update.effective_chat.id
    if chat_id != settings.telegram_chat_id:
        await update.message.reply_text("Unauthorized.")
        return

    user_text = update.message.text
    logger.info(f"Chat message from {update.effective_user.id}: {user_text[:100]}")

    # Send typing indicator
    await update.message.chat.send_action("typing")

    # Check onboarding state — route to onboarding if incomplete
    if settings.onboarding_enabled:
        async with get_db(settings.db_path) as db:
            onboarding = await queries.get_onboarding_state(db)
        if onboarding is None or onboarding.get("current_step") != "done":
            await _handle_onboarding(update, user_text, chat_id, onboarding)
            return

    # Normal chat flow
    await _handle_chat(update, user_text, chat_id)


def _get_providers():
    """Get the singleton provider registry (set at startup in main.py)."""
    try:
        from portfolio_advisor.providers.registry import get_global_registry
        return get_global_registry()
    except Exception:
        return None


async def _handle_onboarding(
    update: Update,
    user_text: str,
    chat_id: int,
    onboarding: dict | None,
) -> None:
    """Route message to the onboarding agent."""
    settings = get_settings()

    async with get_db(settings.db_path) as db:
        prefs = await queries.get_user_preferences(db)
    watchlist = prefs.get("watchlist", settings.default_watchlist)

    app_context = AppContext(
        db_path=settings.db_path,
        telegram_chat_id=settings.telegram_chat_id,
        run_date=date.today(),
        watchlist=watchlist,
        token_budget_remaining=settings.daily_token_budget,
        max_web_search_calls_daily=settings.max_web_searches_daily,
        providers=_get_providers(),
    )

    try:
        from portfolio_advisor.agents.onboarding import get_onboarding_agent

        # Load conversation history (onboarding uses same chat_history table)
        async with get_db(settings.db_path) as db:
            history = await queries.get_chat_history(db, chat_id, limit=20)

        # Build messages with onboarding context
        step = onboarding.get("current_step", "welcome") if onboarding else "welcome"
        system_context = f"[Current onboarding step: {step}]"

        messages = [{"role": "user", "content": system_context}]
        messages.extend({"role": h["role"], "content": h["content"]} for h in history)
        messages.append({"role": "user", "content": user_text})

        result = await Runner.run(
            starting_agent=get_onboarding_agent(),
            input=messages,
            context=app_context,
        )

        response_text = result.final_output or "Let me help you get set up. What's your risk tolerance?"

        # Store both messages in DB
        async with get_db(settings.db_path) as db:
            await queries.store_chat_message(db, chat_id, "user", user_text)
            await queries.store_chat_message(db, chat_id, "assistant", response_text)

        response_text = truncate_for_telegram(response_text)
        try:
            await update.message.reply_text(response_text, parse_mode="Markdown")
        except Exception:
            await update.message.reply_text(response_text)

    except Exception as e:
        logger.exception(f"Onboarding agent error: {e}")
        await update.message.reply_text(
            f"Sorry, I encountered an error. Please try again.\n\nError: {str(e)[:200]}"
        )


async def _handle_chat(update: Update, user_text: str, chat_id: int) -> None:
    """Route message to the chat agent."""
    settings = get_settings()

    async with get_db(settings.db_path) as db:
        prefs = await queries.get_user_preferences(db)
    watchlist = prefs.get("watchlist", settings.default_watchlist)

    app_context = AppContext(
        db_path=settings.db_path,
        telegram_chat_id=settings.telegram_chat_id,
        run_date=date.today(),
        watchlist=watchlist,
        token_budget_remaining=settings.daily_token_budget,
        max_web_search_calls_daily=settings.max_web_searches_daily,
        providers=_get_providers(),
    )

    try:
        from portfolio_advisor.agents.chat import get_chat_agent

        # Load conversation history from DB
        async with get_db(settings.db_path) as db:
            history = await queries.get_chat_history(db, chat_id, limit=20)

        # Build messages list for the agent
        messages = [{"role": h["role"], "content": h["content"]} for h in history]
        messages.append({"role": "user", "content": user_text})

        result = await Runner.run(
            starting_agent=get_chat_agent(),
            input=messages,
            context=app_context,
        )

        response_text = result.final_output or "I couldn't generate a response. Please try again."

        # Store both messages in DB
        async with get_db(settings.db_path) as db:
            await queries.store_chat_message(db, chat_id, "user", user_text)
            await queries.store_chat_message(db, chat_id, "assistant", response_text)

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
            f"Sorry, I encountered an error processing your request. Please try again.\n\n"
            f"Error: {str(e)[:200]}"
        )
