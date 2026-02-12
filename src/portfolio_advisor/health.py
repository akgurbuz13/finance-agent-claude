"""Lightweight health check HTTP server for production monitoring."""

from __future__ import annotations

import logging
from datetime import datetime

from aiohttp import web

logger = logging.getLogger(__name__)

# Module-level references set at startup
_db_path: str = ""
_registry = None
_scheduler_running: bool = False


def configure_health(db_path: str, registry=None) -> None:
    """Configure health check with runtime references."""
    global _db_path, _registry
    _db_path = db_path
    _registry = registry


def set_scheduler_status(running: bool) -> None:
    """Update scheduler running status."""
    global _scheduler_running
    _scheduler_running = running


async def _health_handler(request: web.Request) -> web.Response:
    """GET /health — liveness probe (200 if process running)."""
    return web.json_response({"status": "ok", "timestamp": datetime.now(tz=None).isoformat()})


async def _status_handler(request: web.Request) -> web.Response:
    """GET /status — detailed status: DB, last precompute, providers, scheduler."""
    status = {
        "status": "ok",
        "timestamp": datetime.now(tz=None).isoformat(),
        "scheduler_running": _scheduler_running,
    }

    # DB connectivity
    try:
        if _db_path:
            import aiosqlite
            async with aiosqlite.connect(_db_path) as db:
                cursor = await db.execute("SELECT 1")
                await cursor.fetchone()
            status["db"] = "connected"

            # Last precompute time
            async with aiosqlite.connect(_db_path) as db:
                db.row_factory = aiosqlite.Row
                cursor = await db.execute(
                    "SELECT completed_at FROM analysis_runs "
                    "WHERE run_type = 'precompute' AND status = 'completed' "
                    "ORDER BY completed_at DESC LIMIT 1"
                )
                row = await cursor.fetchone()
                if row:
                    status["last_precompute"] = dict(row).get("completed_at")
                else:
                    status["last_precompute"] = None
        else:
            status["db"] = "not_configured"
    except Exception as e:
        status["db"] = f"error: {e}"

    # Provider status
    if _registry is not None:
        status["providers"] = _registry.provider_status()

    return web.json_response(status)


async def start_health_server(port: int = 8080) -> web.AppRunner:
    """Start the health check HTTP server. Returns the runner for cleanup."""
    app = web.Application()
    app.router.add_get("/health", _health_handler)
    app.router.add_get("/status", _status_handler)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", port)
    await site.start()
    logger.info(f"Health check server started on port {port}")
    return runner
