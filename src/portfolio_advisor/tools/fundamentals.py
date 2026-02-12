"""Fundamentals tools — PE, PB, ROE, margins, analyst consensus."""

from __future__ import annotations

import json
import logging
from datetime import date, datetime

from agents import RunContextWrapper, function_tool

from portfolio_advisor.agents.context import AppContext
from portfolio_advisor.db import queries
from portfolio_advisor.db.connection import get_db

logger = logging.getLogger(__name__)


async def fetch_fundamentals_raw(ticker: str, providers) -> dict | None:
    """Fetch fundamentals from Massive (primary) or Alpha Vantage (fallback)."""
    if providers is None:
        return None
    data, source = await providers.fetch_fundamentals(ticker)
    return data


async def fetch_analyst_ratings_raw(ticker: str, providers) -> list[dict]:
    """Fetch analyst ratings from Massive."""
    if providers is None:
        return []
    return await providers.fetch_analyst_ratings(ticker)


@function_tool
async def get_fundamentals(
    ctx: RunContextWrapper[AppContext],
    ticker: str,
) -> str:
    """Get fundamental ratios for a ticker: PE, PB, ROE, margins, debt ratios, analyst consensus. Uses cached data (< 7 days) or fetches live from Massive/Alpha Vantage."""
    ticker = ticker.upper()

    # Check cache first
    async with get_db(ctx.context.db_path) as db:
        cached = await queries.get_fundamentals(db, ticker)

    if cached:
        try:
            fetch_dt = datetime.strptime(cached["fetch_date"], "%Y-%m-%d")
            age_days = (datetime.utcnow() - fetch_dt).days
            if age_days < 7:
                cached["cached"] = True
                cached["age_days"] = age_days
                return json.dumps(cached, default=str)
        except (ValueError, TypeError):
            pass

    # Fetch fresh
    data = await fetch_fundamentals_raw(ticker, ctx.context.providers)
    if data is None:
        return json.dumps({
            "ticker": ticker,
            "has_data": False,
            "message": "Fundamentals not available. Configure PA_MASSIVE_API_KEY or PA_ALPHA_VANTAGE_API_KEY.",
        })

    # Store in DB
    data["fetch_date"] = date.today().isoformat()
    async with get_db(ctx.context.db_path) as db:
        await queries.store_fundamentals(db, data)

    data["cached"] = False
    return json.dumps(data, default=str)


@function_tool
async def get_valuation_comparison(
    ctx: RunContextWrapper[AppContext],
    tickers: str,
) -> str:
    """Compare valuation metrics side-by-side across multiple tickers. Pass tickers as comma-separated string."""
    ticker_list = [t.strip().upper() for t in tickers.split(",")]
    comparisons = []

    for ticker in ticker_list:
        async with get_db(ctx.context.db_path) as db:
            cached = await queries.get_fundamentals(db, ticker)

        if cached:
            comparisons.append({
                "ticker": ticker,
                "pe_ratio": cached.get("pe_ratio"),
                "forward_pe": cached.get("forward_pe"),
                "pb_ratio": cached.get("pb_ratio"),
                "ps_ratio": cached.get("ps_ratio"),
                "ev_ebitda": cached.get("ev_ebitda"),
                "roe": cached.get("roe"),
                "profit_margin": cached.get("profit_margin"),
                "dividend_yield": cached.get("dividend_yield"),
                "market_cap": cached.get("market_cap"),
                "sector": cached.get("sector"),
            })
        else:
            # Try live fetch
            data = await fetch_fundamentals_raw(ticker, ctx.context.providers)
            if data:
                data["fetch_date"] = date.today().isoformat()
                async with get_db(ctx.context.db_path) as db:
                    await queries.store_fundamentals(db, data)
                comparisons.append({
                    "ticker": ticker,
                    "pe_ratio": data.get("pe_ratio"),
                    "forward_pe": data.get("forward_pe"),
                    "pb_ratio": data.get("pb_ratio"),
                    "ps_ratio": data.get("ps_ratio"),
                    "ev_ebitda": data.get("ev_ebitda"),
                    "roe": data.get("roe"),
                    "profit_margin": data.get("profit_margin"),
                    "dividend_yield": data.get("dividend_yield"),
                    "market_cap": data.get("market_cap"),
                    "sector": data.get("sector"),
                })
            else:
                comparisons.append({"ticker": ticker, "has_data": False})

    return json.dumps({
        "tickers": ticker_list,
        "count": len(comparisons),
        "comparisons": comparisons,
    }, default=str)


@function_tool
async def get_analyst_consensus(
    ctx: RunContextWrapper[AppContext],
    ticker: str,
) -> str:
    """Get analyst ratings, price targets, and consensus for a ticker."""
    ticker = ticker.upper()
    ratings = await fetch_analyst_ratings_raw(ticker, ctx.context.providers)

    if not ratings:
        return json.dumps({
            "ticker": ticker,
            "has_data": False,
            "message": "Analyst ratings not available.",
        })

    # Compute consensus
    price_targets = [r["price_target"] for r in ratings if r.get("price_target")]
    avg_target = round(sum(price_targets) / len(price_targets), 2) if price_targets else None

    return json.dumps({
        "ticker": ticker,
        "has_data": True,
        "rating_count": len(ratings),
        "avg_price_target": avg_target,
        "recent_ratings": ratings[:5],
    }, default=str)
