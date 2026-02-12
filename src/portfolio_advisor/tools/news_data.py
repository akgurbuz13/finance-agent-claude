"""News data tools — Massive API news with per-ticker sentiment."""

from __future__ import annotations

import json
import logging

from agents import RunContextWrapper, function_tool

from portfolio_advisor.agents.context import AppContext

logger = logging.getLogger(__name__)


async def fetch_ticker_news_raw(
    tickers: list[str],
    providers,
    limit: int = 10,
) -> list[dict]:
    """Fetch news with sentiment from Massive provider.

    Returns list of article dicts with per-ticker sentiment when available.
    Falls back to empty list if no provider configured.
    """
    if providers is None:
        return []
    articles = await providers.fetch_news(tickers, limit)
    return articles


@function_tool
async def get_ticker_news(
    ctx: RunContextWrapper[AppContext],
    tickers: str,
    limit: int = 10,
) -> str:
    """Get recent news with sentiment for specified tickers. Returns articles with per-ticker sentiment scores. Pass tickers as comma-separated string (e.g. 'AAPL,MSFT,NVDA')."""
    ticker_list = [t.strip().upper() for t in tickers.split(",")]
    articles = await fetch_ticker_news_raw(
        ticker_list, ctx.context.providers, limit
    )
    if not articles:
        return json.dumps({
            "tickers": ticker_list,
            "has_news": False,
            "message": "No news available. Massive API key may not be configured.",
        })
    return json.dumps({
        "tickers": ticker_list,
        "has_news": True,
        "article_count": len(articles),
        "articles": articles,
    })
