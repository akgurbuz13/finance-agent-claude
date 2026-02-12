"""APScheduler 4.x setup and lifecycle.

Slim schedule (3 LLM jobs/day + 2 data-only):
  06:00 — precompute (no LLM, batch data + indicators)
  06:30 — daily brief (LLM: analysis + news research)
  20:00 — evening summary (LLM: end-of-day recap + news)
  22:00 — forecast eval (no LLM, accuracy tracking)
  Sun 18:00 — weekly report (LLM: full portfolio review)

Removed: midday precompute/update, standalone news checks.
The daily brief and evening summary already include news research,
so separate news_check jobs are redundant and waste tokens.
"""

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


async def start_scheduler() -> None:
    """Initialize the scheduler, configure tasks, and register all cron schedules.

    APScheduler 4.x requires __aenter__ before any method calls including
    configure_task() and add_schedule().
    """
    from portfolio_advisor.scheduler.jobs import (
        daily_job,
        evening_summary_job,
        forecast_evaluation_job,
        precompute_job,
        weekly_job,
    )

    settings = get_settings()
    scheduler = get_scheduler()

    # APScheduler 4.x requires __aenter__ before any method calls
    await scheduler.__aenter__()
    await scheduler.start_in_background()
    logger.info("Scheduler started in background")

    # ── Configure tasks (requires initialized scheduler) ─────────────────

    await scheduler.configure_task(
        "precompute_morning", func=precompute_job, misfire_grace_time=3600
    )
    await scheduler.configure_task(
        "daily_monitoring", func=daily_job, misfire_grace_time=3600
    )
    await scheduler.configure_task(
        "evening_summary", func=evening_summary_job, misfire_grace_time=3600
    )
    await scheduler.configure_task(
        "weekly_report", func=weekly_job, misfire_grace_time=3600
    )
    await scheduler.configure_task(
        "forecast_eval", func=forecast_evaluation_job, misfire_grace_time=3600
    )

    # ── Register schedules (scheduler must be running) ───────────────────

    # Morning: batch data crunch (no LLM cost)
    await scheduler.add_schedule(
        func_or_task_id="precompute_morning",
        trigger=CronTrigger(hour=settings.morning_run_hour, minute=0, timezone="UTC"),
        id="precompute_morning_schedule",
        conflict_policy="replace",
    )

    # Morning: daily brief with news research (LLM)
    await scheduler.add_schedule(
        func_or_task_id="daily_monitoring",
        trigger=CronTrigger(hour=settings.morning_run_hour, minute=30, timezone="UTC"),
        id="daily_monitoring_schedule",
        conflict_policy="replace",
    )

    # Evening: end-of-day recap (LLM)
    await scheduler.add_schedule(
        func_or_task_id="evening_summary",
        trigger=CronTrigger(hour=settings.evening_run_hour, minute=0, timezone="UTC"),
        id="evening_summary_schedule",
        conflict_policy="replace",
    )

    # Weekly: full portfolio review (LLM)
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

    # Forecast eval: accuracy tracking (no LLM cost)
    await scheduler.add_schedule(
        func_or_task_id="forecast_eval",
        trigger=CronTrigger(hour=22, minute=0, timezone="UTC"),
        id="forecast_eval_schedule",
        conflict_policy="replace",
    )

    logger.info(
        f"Scheduler configured: "
        f"precompute at {settings.morning_run_hour}:00, "
        f"daily brief at {settings.morning_run_hour}:30, "
        f"evening at {settings.evening_run_hour}:00, "
        f"weekly on {settings.weekly_run_day} at {settings.weekly_run_hour}:00, "
        f"forecast eval at 22:00 UTC"
    )


async def stop_scheduler() -> None:
    """Stop the scheduler and clean up."""
    scheduler = get_scheduler()
    try:
        await scheduler.stop()
    except Exception:
        pass
    try:
        await scheduler.__aexit__(None, None, None)
    except Exception:
        pass
    logger.info("Scheduler stopped")
