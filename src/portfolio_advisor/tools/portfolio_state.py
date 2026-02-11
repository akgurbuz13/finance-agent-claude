"""Tools for reading/updating the current portfolio state."""

from __future__ import annotations

import json

from agents import RunContextWrapper, function_tool

from portfolio_advisor.agents.context import AppContext
from portfolio_advisor.db.connection import get_db
from portfolio_advisor.db import queries


@function_tool
async def get_current_portfolio(
    ctx: RunContextWrapper[AppContext],
) -> str:
    """Get current portfolio positions with weights, cash%, and last update."""
    async with get_db(ctx.context.db_path) as db:
        positions = await queries.get_portfolio_state(db)

    total_invested = sum(p["weight_pct"] for p in positions)
    cash_pct = round(100.0 - total_invested, 2)

    last_update = max((p["updated_at"] for p in positions), default="never")

    return json.dumps({
        "positions": positions,
        "cash_pct": cash_pct,
        "total_invested_pct": round(total_invested, 2),
        "updated_at": last_update,
    })


@function_tool
async def update_portfolio(
    ctx: RunContextWrapper[AppContext],
    ticker: str,
    new_weight_pct: float,
    reason: str = "",
) -> str:
    """Update a portfolio position (user-confirmed trade). Set weight to 0 to remove."""
    asset_class = _classify_asset(ticker)

    async with get_db(ctx.context.db_path) as db:
        if new_weight_pct <= 0:
            await queries.remove_portfolio_position(db, ticker)
            action = "removed"
        else:
            await queries.update_portfolio_position(db, ticker, new_weight_pct, asset_class)
            action = "updated"

        # Snapshot after change
        await queries.snapshot_portfolio(db, trigger="user_confirmed")

    return json.dumps({
        "status": "ok",
        "action": action,
        "ticker": ticker,
        "new_weight_pct": new_weight_pct,
        "reason": reason,
    })


@function_tool
async def get_portfolio_history(
    ctx: RunContextWrapper[AppContext],
    days: int = 30,
) -> str:
    """Get portfolio snapshots over the last N days."""
    async with get_db(ctx.context.db_path) as db:
        history = await queries.get_portfolio_history(db, days)
    return json.dumps(history, default=str)


def _classify_asset(ticker: str) -> str:
    ticker = ticker.upper()
    bond_set = {"TLT", "IEF", "HYG", "AGG", "BND", "LQD"}
    commodity_set = {"GLD", "SLV", "USO", "DBA"}
    crypto_set = {"BTC", "ETH", "SOL", "AVAX"}

    if ticker in crypto_set:
        return "crypto"
    elif ticker in bond_set:
        return "bond"
    elif ticker in commodity_set:
        return "commodity"
    return "equity"
