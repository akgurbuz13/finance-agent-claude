"""Tools for reading/updating user preferences and watchlist."""

from __future__ import annotations

import json

from agents import RunContextWrapper, function_tool

from portfolio_advisor.agents.context import AppContext
from portfolio_advisor.db.connection import get_db
from portfolio_advisor.db import queries


@function_tool
async def get_user_preferences(
    ctx: RunContextWrapper[AppContext],
) -> str:
    """Get current user preferences (risk_tolerance, time_horizon, watchlist, etc.)."""
    async with get_db(ctx.context.db_path) as db:
        prefs = await queries.get_user_preferences(db)
    return json.dumps(prefs, default=str)


@function_tool
async def update_user_preference(
    ctx: RunContextWrapper[AppContext],
    key: str,
    value: str,
) -> str:
    """Update a single user preference. Key must be a valid column name.
    For list values (excluded_assets, allowed_regions), pass JSON array string."""
    # Parse JSON values for list fields
    parsed_value: str | list = value
    if key in ("excluded_assets", "allowed_regions", "watchlist"):
        try:
            parsed_value = json.loads(value)
        except json.JSONDecodeError:
            pass

    # Validate key
    valid_keys = {
        "risk_tolerance", "time_horizon", "excluded_assets",
        "allowed_regions", "cash_target_pct", "max_position_pct", "watchlist",
    }
    if key not in valid_keys:
        return json.dumps({"error": f"Invalid preference key: {key}. Valid: {sorted(valid_keys)}"})

    # Type coercion for numeric fields
    if key in ("cash_target_pct", "max_position_pct"):
        parsed_value = float(value)

    async with get_db(ctx.context.db_path) as db:
        await queries.update_user_preference(db, key, parsed_value)

    return json.dumps({"status": "ok", "key": key, "new_value": parsed_value})


@function_tool
async def update_watchlist(
    ctx: RunContextWrapper[AppContext],
    action: str,
    tickers: str,
) -> str:
    """Add or remove tickers from the watchlist. action: 'add' or 'remove'. tickers: comma-separated."""
    ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]

    async with get_db(ctx.context.db_path) as db:
        prefs = await queries.get_user_preferences(db)
        current = prefs.get("watchlist", [])

        if action == "add":
            updated = list(set(current + ticker_list))
            added = [t for t in ticker_list if t not in current]
            msg = f"Added: {', '.join(added)}" if added else "All tickers already in watchlist"
        elif action == "remove":
            updated = [t for t in current if t not in ticker_list]
            removed = [t for t in ticker_list if t in current]
            msg = f"Removed: {', '.join(removed)}" if removed else "None of those tickers were in watchlist"
        else:
            return json.dumps({"error": f"Invalid action: {action}. Use 'add' or 'remove'."})

        await queries.update_user_preference(db, "watchlist", updated)

    return json.dumps({
        "status": "ok",
        "action": action,
        "message": msg,
        "watchlist": sorted(updated),
    })
