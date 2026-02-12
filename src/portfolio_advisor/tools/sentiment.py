"""Sentiment tools — short interest, squeeze risk scoring."""

from __future__ import annotations

import json
import logging

from agents import RunContextWrapper, function_tool

from portfolio_advisor.agents.context import AppContext

logger = logging.getLogger(__name__)


async def fetch_short_interest_raw(tickers: list[str], providers) -> list[dict]:
    """Fetch short interest data from Massive provider."""
    if providers is None:
        return []
    return await providers.fetch_short_interest(tickers)


def _compute_squeeze_risk(short_pct_float: float | None, days_to_cover: float | None) -> dict:
    """Compute a short squeeze risk score from short interest metrics."""
    score = 0.0
    factors = []

    if short_pct_float is not None:
        if short_pct_float > 20:
            score += 0.4
            factors.append(f"Very high short interest ({short_pct_float:.1f}% of float)")
        elif short_pct_float > 10:
            score += 0.25
            factors.append(f"Elevated short interest ({short_pct_float:.1f}% of float)")
        elif short_pct_float > 5:
            score += 0.1
            factors.append(f"Moderate short interest ({short_pct_float:.1f}% of float)")

    if days_to_cover is not None:
        if days_to_cover > 5:
            score += 0.4
            factors.append(f"High days-to-cover ({days_to_cover:.1f})")
        elif days_to_cover > 3:
            score += 0.2
            factors.append(f"Moderate days-to-cover ({days_to_cover:.1f})")

    if score >= 0.6:
        level = "high"
    elif score >= 0.3:
        level = "moderate"
    else:
        level = "low"

    return {"score": round(score, 2), "level": level, "factors": factors}


@function_tool
async def get_short_interest(
    ctx: RunContextWrapper[AppContext],
    ticker: str,
) -> str:
    """Get short interest data with squeeze risk score for a ticker. Shows shares short, % of float, days to cover, and squeeze risk assessment."""
    ticker = ticker.upper()
    entries = await fetch_short_interest_raw([ticker], ctx.context.providers)

    if not entries:
        return json.dumps({
            "ticker": ticker,
            "has_data": False,
            "message": "Short interest data not available. Configure PA_MASSIVE_API_KEY.",
        })

    entry = entries[0]
    squeeze = _compute_squeeze_risk(
        entry.get("short_pct_float"),
        entry.get("days_to_cover"),
    )
    entry["squeeze_risk"] = squeeze
    entry["has_data"] = True
    return json.dumps(entry, default=str)
