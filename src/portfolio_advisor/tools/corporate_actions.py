"""Corporate actions tools — dividends and stock splits."""

from __future__ import annotations

import json
import logging

from agents import RunContextWrapper, function_tool

from portfolio_advisor.agents.context import AppContext

logger = logging.getLogger(__name__)


async def fetch_dividends_raw(ticker: str, providers) -> list[dict]:
    """Fetch dividend history from Massive provider."""
    if providers is None:
        return []
    return await providers.fetch_dividends(ticker)


async def fetch_splits_raw(ticker: str, providers) -> list[dict]:
    """Fetch stock split history from Massive provider."""
    if providers is None:
        return []
    return await providers.fetch_splits(ticker)


@function_tool
async def get_dividend_info(
    ctx: RunContextWrapper[AppContext],
    ticker: str,
) -> str:
    """Get dividend information for a ticker: yield, next ex-date, payout history, and frequency."""
    ticker = ticker.upper()
    dividends = await fetch_dividends_raw(ticker, ctx.context.providers)

    if not dividends:
        return json.dumps({
            "ticker": ticker,
            "has_data": False,
            "message": "Dividend data not available.",
        })

    # Compute summary
    latest = dividends[0]
    annual_amount = 0.0
    frequency = latest.get("frequency", "unknown")

    # Sum last 4 quarters or 12 months
    for d in dividends[:4]:
        amt = d.get("amount")
        if amt:
            annual_amount += amt

    return json.dumps({
        "ticker": ticker,
        "has_data": True,
        "next_ex_date": latest.get("ex_date"),
        "last_amount": latest.get("amount"),
        "annualized_amount": round(annual_amount, 4),
        "frequency": frequency,
        "payout_history": dividends[:8],
    }, default=str)
