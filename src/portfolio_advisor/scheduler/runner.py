"""APScheduler 4.x setup and lifecycle."""

from __future__ import annotations

import logging

from apscheduler import AsyncScheduler
from apscheduler.triggers.cron import CronTrigger

from portfolio_advisor.config import get_settings

logger = logging.getLogger(__name__)

_scheduler: AsyncScheduler | None = None


def get_scheduler() -> AsyncScheduler:
    global _scheduler
    if _scheduler is None:
        _scheduler = AsyncScheduler()
    return _scheduler


async def setup_scheduler() -> AsyncScheduler:
    """Configure and return the scheduler with daily + weekly jobs."""
    from portfolio_advisor.scheduler.jobs import daily_job, weekly_job

    settings = get_settings()
    scheduler = get_scheduler()

    # Configure tasks
    scheduler.configure_task("daily_monitoring", func=daily_job, misfire_grace_time=3600)
    scheduler.configure_task("weekly_report", func=weekly_job, misfire_grace_time=3600)

    # Daily schedule
    await scheduler.add_schedule(
        task_id="daily_monitoring",
        trigger=CronTrigger(hour=settings.daily_run_hour, minute=0, timezone="UTC"),
        id="daily_monitoring_schedule",
        conflict_policy="replace",
    )

    # Weekly schedule
    day_map = {"mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6}
    day_of_week = day_map.get(settings.weekly_run_day.lower(), 6)

    await scheduler.add_schedule(
        task_id="weekly_report",
        trigger=CronTrigger(
            day_of_week=day_of_week,
            hour=settings.weekly_run_hour,
            minute=0,
            timezone="UTC",
        ),
        id="weekly_report_schedule",
        conflict_policy="replace",
    )

    logger.info(
        f"Scheduler configured: daily at {settings.daily_run_hour}:00 UTC, "
        f"weekly on {settings.weekly_run_day} at {settings.weekly_run_hour}:00 UTC"
    )
    return scheduler


async def start_scheduler() -> None:
    """Start the scheduler in the background."""
    scheduler = get_scheduler()
    await scheduler.start_in_background()
    logger.info("Scheduler started in background")


async def stop_scheduler() -> None:
    """Stop the scheduler."""
    scheduler = get_scheduler()
    await scheduler.stop()
    logger.info("Scheduler stopped")
