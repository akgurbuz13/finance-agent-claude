"""Entry point — wires scheduler + Telegram bot together."""

from __future__ import annotations

import logging
import os

from portfolio_advisor.config import get_settings
from portfolio_advisor.db.connection import init_db
from portfolio_advisor.scheduler.runner import start_scheduler, stop_scheduler
from portfolio_advisor.telegram_bot.bot import build_application
from portfolio_advisor.utils.logging import setup_logging

# Module-level references for cleanup
_health_runner = None
_registry = None


async def startup(app) -> None:
    """Initialize database, provider registry, health server, and scheduler.

    Called by python-telegram-bot's post_init hook, which ensures we share
    the same event loop as the Telegram bot and APScheduler.
    """
    global _health_runner, _registry

    settings = get_settings()

    # Set OpenAI API key
    os.environ["OPENAI_API_KEY"] = settings.openai_api_key

    # Initialize database (with WAL mode)
    await init_db(settings.db_path)
    logging.getLogger(__name__).info(f"Database initialized at {settings.db_path}")

    # Initialize provider registry (singleton — shared across all jobs)
    from portfolio_advisor.providers.registry import create_registry, set_global_registry

    _registry = create_registry(settings)
    set_global_registry(_registry)
    logging.getLogger(__name__).info("Provider registry initialized")

    # Start health check server
    try:
        from portfolio_advisor.health import configure_health, set_scheduler_status, start_health_server

        configure_health(settings.db_path, _registry)
        _health_runner = await start_health_server(settings.health_port)
    except Exception as e:
        logging.getLogger(__name__).warning(f"Health server failed to start: {e}")

    # Start scheduler (configures tasks + registers schedules)
    await start_scheduler()

    # Update health check scheduler status
    try:
        from portfolio_advisor.health import set_scheduler_status
        set_scheduler_status(True)
    except Exception:
        pass


async def shutdown(app) -> None:
    """Cleanup resources on shutdown."""
    global _health_runner, _registry

    await stop_scheduler()

    if _health_runner is not None:
        await _health_runner.cleanup()

    if _registry is not None:
        await _registry.close()


def main() -> None:
    """Main entry point: start scheduler + Telegram bot polling."""
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Starting Portfolio Advisor v4")

    # Build Telegram bot app, wire startup/shutdown as lifecycle hooks
    # so everything shares a single event loop
    app = build_application()
    app.post_init = startup
    app.post_shutdown = shutdown

    try:
        logger.info("Starting Telegram bot polling")
        app.run_polling(drop_pending_updates=True)
    except KeyboardInterrupt:
        logger.info("Shutting down...")

    logger.info("Portfolio Advisor stopped")


if __name__ == "__main__":
    main()
