"""Entry point — wires scheduler + Telegram bot together."""

from __future__ import annotations

import asyncio
import logging
import os

from portfolio_advisor.config import get_settings
from portfolio_advisor.db.connection import init_db
from portfolio_advisor.scheduler.runner import setup_scheduler, start_scheduler, stop_scheduler
from portfolio_advisor.telegram_bot.bot import build_application
from portfolio_advisor.utils.logging import setup_logging


async def startup() -> None:
    """Initialize database, scheduler, and seed defaults."""
    settings = get_settings()

    # Set OpenAI API key
    os.environ["OPENAI_API_KEY"] = settings.openai_api_key

    # Initialize database
    await init_db(settings.db_path)
    logging.getLogger(__name__).info(f"Database initialized at {settings.db_path}")

    # Setup and start scheduler
    await setup_scheduler()
    await start_scheduler()


def main() -> None:
    """Main entry point: start scheduler + Telegram bot polling."""
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Starting Portfolio Advisor")

    # Run async startup (db + scheduler)
    asyncio.run(startup())

    # Build and run Telegram bot (this blocks with its own event loop)
    app = build_application()

    try:
        logger.info("Starting Telegram bot polling")
        app.run_polling(drop_pending_updates=True)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        asyncio.run(stop_scheduler())
        logger.info("Portfolio Advisor stopped")


if __name__ == "__main__":
    main()
