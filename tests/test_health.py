"""Tests for the health check server."""

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from portfolio_advisor.health import (
    _health_handler,
    _status_handler,
    configure_health,
    set_scheduler_status,
)


def _make_app():
    app = web.Application()
    app.router.add_get("/health", _health_handler)
    app.router.add_get("/status", _status_handler)
    return app


async def test_health_endpoint_returns_ok():
    app = _make_app()
    async with TestClient(TestServer(app)) as client:
        resp = await client.get("/health")
        assert resp.status == 200
        data = await resp.json()
        assert data["status"] == "ok"
        assert "timestamp" in data


async def test_status_endpoint_returns_scheduler_state():
    set_scheduler_status(True)
    app = _make_app()
    async with TestClient(TestServer(app)) as client:
        resp = await client.get("/status")
        assert resp.status == 200
        data = await resp.json()
        assert data["scheduler_running"] is True


async def test_status_without_db():
    configure_health(db_path="", registry=None)
    app = _make_app()
    async with TestClient(TestServer(app)) as client:
        resp = await client.get("/status")
        data = await resp.json()
        assert data["db"] == "not_configured"


async def test_status_with_db(tmp_path):
    from portfolio_advisor.db.connection import init_db
    db_path = str(tmp_path / "test.db")
    await init_db(db_path)

    configure_health(db_path=db_path, registry=None)
    app = _make_app()
    async with TestClient(TestServer(app)) as client:
        resp = await client.get("/status")
        data = await resp.json()
        assert data["db"] == "connected"
