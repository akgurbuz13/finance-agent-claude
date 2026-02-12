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
    """Configure task definitions on the scheduler (before it is started)."""
    from portfolio_advisor.scheduler.jobs import (
        daily_job,
        evening_summary_job,
        forecast_evaluation_job,
        midday_update_job,
        news_check_job,
        precompute_job,
        weekly_job,
    )

    settings = get_settings()
    scheduler = get_scheduler()

    # Configure all tasks (this does not require the scheduler to be running)
    scheduler.configure_task(
        "precompute_morning", func=precompute_job, misfire_grace_time=3600
    )
    scheduler.configure_task(
        "daily_monitoring", func=daily_job, misfire_grace_time=3600
    )
    scheduler.configure_task(
        "precompute_midday", func=precompute_job, misfire_grace_time=3600
    )
    scheduler.configure_task(
        "midday_update", func=midday_update_job, misfire_grace_time=3600
    )
    scheduler.configure_task(
        "evening_summary", func=evening_summary_job, misfire_grace_time=3600
    )
    scheduler.configure_task(
        "weekly_report", func=weekly_job, misfire_grace_time=3600
    )
    scheduler.configure_task(
        "forecast_eval", func=forecast_evaluation_job, misfire_grace_time=3600
    )

    for i, hour in enumerate(settings.news_check_hours):
        scheduler.configure_task(
            f"news_check_{i}", func=news_check_job, misfire_grace_time=3600
        )

    return scheduler


async def start_scheduler() -> None:
    """Start the scheduler and register all cron schedules.

    add_schedule() requires the scheduler to be initialized (started),
    so schedules are registered here after start_in_background().
    """
    settings = get_settings()
    scheduler = get_scheduler()
    await scheduler.start_in_background()
    logger.info("Scheduler started in background")

    # ── Register schedules (scheduler must be running) ──────────────────

    await scheduler.add_schedule(
        func_or_task_id="precompute_morning",
        trigger=CronTrigger(hour=settings.morning_run_hour, minute=0, timezone="UTC"),
        id="precompute_morning_schedule",
        conflict_policy="replace",
    )
    await scheduler.add_schedule(
        func_or_task_id="daily_monitoring",
        trigger=CronTrigger(hour=settings.morning_run_hour, minute=30, timezone="UTC"),
        id="daily_monitoring_schedule",
        conflict_policy="replace",
    )
    await scheduler.add_schedule(
        func_or_task_id="precompute_midday",
        trigger=CronTrigger(hour=settings.midday_run_hour, minute=0, timezone="UTC"),
        id="precompute_midday_schedule",
        conflict_policy="replace",
    )
    await scheduler.add_schedule(
        func_or_task_id="midday_update",
        trigger=CronTrigger(hour=settings.midday_run_hour, minute=30, timezone="UTC"),
        id="midday_update_schedule",
        conflict_policy="replace",
    )
    await scheduler.add_schedule(
        func_or_task_id="evening_summary",
        trigger=CronTrigger(hour=settings.evening_run_hour, minute=0, timezone="UTC"),
        id="evening_summary_schedule",
        conflict_policy="replace",
    )

    day_map = {"mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6}
    day_of_week = day_map.get(settings.weekly_run_day.lower(), 6)
    await scheduler.add_schedule(
        func_or_task_id="weekly_report",
        trigger=CronTrigger(
            day_of_week=day_of_week,
            hour=settings.weekly_run_hour,
            minute=0,
            timezone="UTC",
        ),
        id="weekly_report_schedule",
        conflict_policy="replace",
    )
    await scheduler.add_schedule(
        func_or_task_id="forecast_eval",
        trigger=CronTrigger(hour=22, minute=0, timezone="UTC"),
        id="forecast_eval_schedule",
        conflict_policy="replace",
    )

    for i, hour in enumerate(settings.news_check_hours):
        await scheduler.add_schedule(
            func_or_task_id=f"news_check_{i}",
            trigger=CronTrigger(hour=hour, minute=0, timezone="UTC"),
            id=f"news_check_{i}_schedule",
            conflict_policy="replace",
        )

    news_hours_str = ", ".join(f"{h}:00" for h in settings.news_check_hours)
    logger.info(
        f"Scheduler configured: "
        f"precompute at {settings.morning_run_hour}:00/{settings.midday_run_hour}:00, "
        f"daily at {settings.morning_run_hour}:30, "
        f"midday at {settings.midday_run_hour}:30, "
        f"news checks at {news_hours_str}, "
        f"evening at {settings.evening_run_hour}:00, "
        f"weekly on {settings.weekly_run_day} at {settings.weekly_run_hour}:00, "
        f"forecast eval at 22:00 UTC"
    )


async def stop_scheduler() -> None:
    """Stop the scheduler."""
    scheduler = get_scheduler()
    await scheduler.stop()
    logger.info("Scheduler stopped")
