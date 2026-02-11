"""Earnings calendar tools — fetch, store, and query earnings data."""

from __future__ import annotations

import json
import logging
from datetime import datetime

import yfinance as yf
from agents import RunContextWrapper, function_tool

from portfolio_advisor.agents.context import AppContext
from portfolio_advisor.db import queries
from portfolio_advisor.db.connection import get_db
from portfolio_advisor.tools.market_data import CRYPTO_MAP

logger = logging.getLogger(__name__)


# ── Pure computation function (no ctx) ───────────────────────────────────────


def fetch_earnings_calendar_raw(tickers: list[str]) -> list[dict]:
    """Fetch upcoming earnings dates and estimates for a list of tickers.

    Uses yfinance .calendar and .earnings_dates attributes.
    Skips crypto tickers (no earnings). Returns list of dicts.
    """
    results = []
    for ticker in tickers:
        if ticker.upper() in CRYPTO_MAP:
            continue
        try:
            t = yf.Ticker(ticker)

            # Try .calendar for next earnings date
            cal = t.calendar
            if cal is None or (hasattr(cal, "empty") and cal.empty):
                continue

            entry = {"ticker": ticker.upper(), "source": "yfinance"}

            # calendar can be a dict or DataFrame depending on yfinance version
            if isinstance(cal, dict):
                earnings_date = cal.get("Earnings Date")
                if isinstance(earnings_date, list) and earnings_date:
                    earnings_date = earnings_date[0]
                if earnings_date is not None:
                    if hasattr(earnings_date, "strftime"):
                        entry["earnings_date"] = earnings_date.strftime("%Y-%m-%d")
                    else:
                        entry["earnings_date"] = str(earnings_date)[:10]

                # Estimates
                eps_est = cal.get("EPS Estimate")
                rev_est = cal.get("Revenue Estimate")
                if eps_est is not None:
                    try:
                        entry["eps_estimate"] = float(eps_est)
                    except (ValueError, TypeError):
                        pass
                if rev_est is not None:
                    try:
                        entry["revenue_estimate"] = float(rev_est)
                    except (ValueError, TypeError):
                        pass
            else:
                # DataFrame format (older yfinance)
                try:
                    if "Earnings Date" in cal.index:
                        ed = cal.loc["Earnings Date"].iloc[0]
                        if hasattr(ed, "strftime"):
                            entry["earnings_date"] = ed.strftime("%Y-%m-%d")
                        else:
                            entry["earnings_date"] = str(ed)[:10]
                except (KeyError, IndexError, AttributeError):
                    pass

            if "earnings_date" not in entry:
                continue

            # Determine earnings time from historical pattern
            entry["earnings_time"] = "unknown"
            entry["status"] = "upcoming"

            # Try to get historical earnings for surprise data
            try:
                earnings_dates = t.earnings_dates
                if earnings_dates is not None and not earnings_dates.empty:
                    # Most recent reported earnings
                    past = earnings_dates[
                        earnings_dates.index <= datetime.now()
                    ]
                    if not past.empty:
                        last = past.iloc[0]
                        eps_actual = last.get("Reported EPS")
                        eps_est_hist = last.get("EPS Estimate")
                        if eps_actual is not None and eps_est_hist is not None:
                            try:
                                actual = float(eps_actual)
                                estimate = float(eps_est_hist)
                                if estimate != 0:
                                    surprise = ((actual - estimate) / abs(estimate)) * 100
                                    # Store as separate "reported" entry
                                    reported_date = past.index[0]
                                    if hasattr(reported_date, "strftime"):
                                        reported_entry = {
                                            "ticker": ticker.upper(),
                                            "earnings_date": reported_date.strftime(
                                                "%Y-%m-%d"
                                            ),
                                            "eps_estimate": round(estimate, 2),
                                            "eps_actual": round(actual, 2),
                                            "eps_surprise_pct": round(surprise, 1),
                                            "status": "reported",
                                            "source": "yfinance",
                                        }
                                        results.append(reported_entry)
                            except (ValueError, TypeError):
                                pass
            except Exception:
                pass

            results.append(entry)

        except Exception as e:
            logger.debug(f"Earnings fetch failed for {ticker}: {e}")
            continue

    return results


# ── @function_tool wrappers (used by chat agent) ────────────────────────────


@function_tool
async def get_upcoming_earnings(
    ctx: RunContextWrapper[AppContext],
    days: int = 14,
) -> str:
    """Get upcoming earnings for watchlist tickers in the next N days.

    Returns earnings dates, times, and consensus estimates from the pre-computed
    earnings calendar. Updated 2-3x daily by the pre-compute pipeline.
    """
    async with get_db(ctx.context.db_path) as db:
        upcoming = await queries.get_upcoming_earnings(db, days)

    if not upcoming:
        return json.dumps({
            "has_earnings": False,
            "days_searched": days,
            "message": "No upcoming earnings found for watchlist tickers.",
        })

    return json.dumps({
        "has_earnings": True,
        "count": len(upcoming),
        "days_searched": days,
        "earnings": upcoming,
    }, default=str)


@function_tool
async def get_earnings_results(
    ctx: RunContextWrapper[AppContext],
    ticker: str,
) -> str:
    """Get earnings history for a specific ticker.

    Returns past earnings with EPS estimates vs actuals and surprise percentages.
    """
    async with get_db(ctx.context.db_path) as db:
        entries = await queries.get_earnings_for_ticker(db, ticker.upper())

    if not entries:
        return json.dumps({
            "ticker": ticker.upper(),
            "has_data": False,
            "message": "No earnings data found for this ticker.",
        })

    return json.dumps({
        "ticker": ticker.upper(),
        "has_data": True,
        "count": len(entries),
        "earnings": entries,
    }, default=str)
