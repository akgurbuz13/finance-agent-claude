"""Integration tests for the health check HTTP server."""

from __future__ import annotations

from unittest.mock import MagicMock

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from portfolio_advisor.health import (
    _health_handler,
    _status_handler,
    configure_health,
    set_scheduler_status,
)


def _make_app() -> web.Application:
    app = web.Application()
    app.router.add_get("/health", _health_handler)
    app.router.add_get("/status", _status_handler)
    return app


async def test_health_endpoint_returns_200():
    """GET /health should return 200 with status 'ok'."""
    app = _make_app()
    async with TestClient(TestServer(app)) as client:
        resp = await client.get("/health")
        assert resp.status == 200
        data = await resp.json()
        assert data["status"] == "ok"
        assert "timestamp" in data


async def test_status_endpoint_shows_providers(tmp_path):
    """GET /status with a registry set should include provider info."""
    from portfolio_advisor.db.connection import init_db

    db_path = str(tmp_path / "health_test.db")
    await init_db(db_path)

    # Create a mock registry with provider_status
    mock_registry = MagicMock()
    mock_registry.provider_status.return_value = {
        "yfinance": {"available": True},
        "coingecko": {"available": True},
    }

    configure_health(db_path=db_path, registry=mock_registry)
    set_scheduler_status(True)

    app = _make_app()
    async with TestClient(TestServer(app)) as client:
        resp = await client.get("/status")
        assert resp.status == 200
        data = await resp.json()

        assert "providers" in data
        assert "yfinance" in data["providers"]
        assert data["providers"]["yfinance"]["available"] is True


async def test_status_endpoint_format(tmp_path):
    """GET /status JSON response should have expected keys."""
    from portfolio_advisor.db.connection import init_db

    db_path = str(tmp_path / "health_format.db")
    await init_db(db_path)

    configure_health(db_path=db_path, registry=None)
    set_scheduler_status(False)

    app = _make_app()
    async with TestClient(TestServer(app)) as client:
        resp = await client.get("/status")
        assert resp.status == 200
        data = await resp.json()

        # Core keys
        assert "status" in data
        assert data["status"] == "ok"
        assert "timestamp" in data
        assert "scheduler_running" in data
        assert data["scheduler_running"] is False
        assert "db" in data
        assert data["db"] == "connected"
        # last_precompute can be None if no runs have been done
        assert "last_precompute" in data
