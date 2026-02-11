"""Token usage logging and budget enforcement."""

from __future__ import annotations

import json

from agents import RunContextWrapper, function_tool

from portfolio_advisor.agents.context import AppContext
from portfolio_advisor.db.connection import get_db
from portfolio_advisor.db import queries


# Approximate cost per 1M tokens (USD) — update as pricing changes
COST_PER_1M = {
    "gpt-5.2": {"input": 10.0, "output": 30.0},
    "gpt-5-mini": {"input": 0.4, "output": 1.6},
}


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    rates = COST_PER_1M.get(model, {"input": 1.0, "output": 3.0})
    cost = (input_tokens / 1_000_000 * rates["input"]) + (
        output_tokens / 1_000_000 * rates["output"]
    )
    return round(cost, 6)


@function_tool
async def log_token_usage(
    ctx: RunContextWrapper[AppContext],
    model: str,
    run_type: str,
    input_tokens: int,
    output_tokens: int,
) -> str:
    """Log token usage for a model call."""
    cost = estimate_cost(model, input_tokens, output_tokens)
    async with get_db(ctx.context.db_path) as db:
        await queries.log_token_usage(db, model, run_type, input_tokens, output_tokens, cost)
    return json.dumps({
        "status": "ok",
        "estimated_cost_usd": cost,
    })


@function_tool
async def get_usage_summary(
    ctx: RunContextWrapper[AppContext],
    days: int = 30,
) -> str:
    """Get token usage summary for the last N days."""
    async with get_db(ctx.context.db_path) as db:
        summary = await queries.get_token_usage_summary(db, days)
        daily_used = await queries.get_daily_token_usage(db)

    return json.dumps({
        "period_days": days,
        "total_input_tokens": summary.get("total_input", 0),
        "total_output_tokens": summary.get("total_output", 0),
        "total_cost_usd": round(summary.get("total_cost", 0), 4),
        "total_calls": summary.get("call_count", 0),
        "today_tokens_used": daily_used,
        "today_budget_remaining": max(0, ctx.context.token_budget_remaining - daily_used),
    })


async def check_budget(ctx: AppContext) -> bool:
    """Check if we're within daily token budget. Returns True if OK to proceed."""
    async with get_db(ctx.db_path) as db:
        daily_used = await queries.get_daily_token_usage(db)
    return daily_used < ctx.token_budget_remaining
