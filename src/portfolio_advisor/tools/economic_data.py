"""Economic data tools — FRED series, yield curve, economic calendar, macro regime."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta

from agents import RunContextWrapper, function_tool

from portfolio_advisor.agents.context import AppContext
from portfolio_advisor.config import get_settings

logger = logging.getLogger(__name__)


# ── Helper: fetch treasury yields via yfinance ────────────────────────────────


def _fetch_treasury_yields() -> dict | None:
    """Fetch current treasury yields using yfinance ticker symbols."""
    import yfinance as yf

    # Treasury yield tickers in yfinance
    # ^IRX = 13-week, ^FVX = 5-year, ^TNX = 10-year, ^TYX = 30-year
    yield_tickers = {
        "3m": "^IRX",
        "5y": "^FVX",
        "10y": "^TNX",
        "30y": "^TYX",
    }

    yields = {}
    try:
        tickers_str = " ".join(yield_tickers.values())
        data = yf.download(tickers_str, period="5d", progress=False)
        if data.empty:
            return None

        for label, ticker in yield_tickers.items():
            try:
                if len(yield_tickers) == 1:
                    val = float(data["Close"].dropna().iloc[-1])
                else:
                    val = float(data["Close"][ticker].dropna().iloc[-1])
                yields[label] = round(val, 3)
            except (KeyError, IndexError):
                pass
    except Exception as e:
        logger.warning(f"Treasury yield fetch failed: {e}")
        return None

    return yields if yields else None


# ── FRED API helper ──────────────────────────────────────────────────────────


def _fetch_fred_data(series_id: str, start: str | None = None, end: str | None = None) -> list[dict] | None:
    """Fetch data from FRED API. Returns list of {date, value} dicts."""
    import httpx

    api_key = get_settings().fred_api_key
    if not api_key:
        return None

    if end is None:
        end = datetime.utcnow().strftime("%Y-%m-%d")
    if start is None:
        start = (datetime.utcnow() - timedelta(days=365 * 2)).strftime("%Y-%m-%d")

    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": start,
        "observation_end": end,
        "sort_order": "desc",
        "limit": 500,
    }

    try:
        with httpx.Client(timeout=15) as client:
            resp = client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()

        observations = []
        for obs in data.get("observations", []):
            val = obs.get("value", ".")
            if val != ".":
                observations.append({
                    "date": obs["date"],
                    "value": float(val),
                })
        return observations
    except Exception as e:
        logger.warning(f"FRED fetch failed for {series_id}: {e}")
        return None


# ── @function_tool wrappers ──────────────────────────────────────────────────


@function_tool
async def fetch_fred_series(
    ctx: RunContextWrapper[AppContext],
    series_id: str,
    start_date: str = "",
    end_date: str = "",
) -> str:
    """Fetch economic data from FRED (Federal Reserve). Common series: CPIAUCSL (CPI), UNRATE (unemployment), DFF (fed funds rate), DGS10 (10Y treasury), BAMLH0A0HYM2 (HY spread), GDP, INDPRO (industrial production). Requires PA_FRED_API_KEY."""
    start = start_date if start_date else None
    end = end_date if end_date else None

    data = _fetch_fred_data(series_id, start, end)

    if data is None:
        # Fallback: try yfinance for common series
        if series_id in ("DGS10", "DGS2", "DGS5", "DGS30"):
            yields = _fetch_treasury_yields()
            if yields:
                mapping = {"DGS10": "10y", "DGS2": "5y", "DGS5": "5y", "DGS30": "30y"}
                key = mapping.get(series_id, "10y")
                if key in yields:
                    return json.dumps({
                        "series_id": series_id,
                        "source": "yfinance_fallback",
                        "latest_value": yields[key],
                        "observations": [{"date": datetime.utcnow().strftime("%Y-%m-%d"), "value": yields[key]}],
                    })
        return json.dumps({
            "series_id": series_id,
            "error": "FRED API key not configured or fetch failed. Set PA_FRED_API_KEY.",
        })

    latest = data[0] if data else None

    return json.dumps({
        "series_id": series_id,
        "source": "fred",
        "n_observations": len(data),
        "latest_date": latest["date"] if latest else None,
        "latest_value": latest["value"] if latest else None,
        "observations": data[:60],  # Last 60 observations
    })


@function_tool
async def get_yield_curve(
    ctx: RunContextWrapper[AppContext],
) -> str:
    """Get current US Treasury yield curve. Reports 3M, 5Y, 10Y, 30Y yields, curve slope (10Y-3M), and inversion status. Uses yfinance real-time data."""
    yields = _fetch_treasury_yields()

    if not yields:
        # Try FRED fallback
        fred_series = {"2y": "DGS2", "5y": "DGS5", "10y": "DGS10", "30y": "DGS30"}
        yields = {}
        for label, series_id in fred_series.items():
            data = _fetch_fred_data(series_id)
            if data:
                yields[label] = data[0]["value"]

    if not yields:
        return json.dumps({"error": "Could not fetch yield data from yfinance or FRED"})

    # Compute curve metrics
    y_3m = yields.get("3m")
    y_10y = yields.get("10y")
    y_30y = yields.get("30y")

    slope_10y_3m = round(y_10y - y_3m, 3) if y_10y is not None and y_3m is not None else None
    slope_30y_10y = round(y_30y - y_10y, 3) if y_30y is not None and y_10y is not None else None

    is_inverted = slope_10y_3m is not None and slope_10y_3m < 0

    # Curvature: 2*10Y - 3M - 30Y (butterfly spread approximation using available maturities)
    curvature = None
    if y_10y is not None and y_3m is not None and y_30y is not None:
        curvature = round(2 * y_10y - y_3m - y_30y, 3)

    return json.dumps({
        "yields": yields,
        "slope_10y_3m": slope_10y_3m,
        "slope_30y_10y": slope_30y_10y,
        "curvature": curvature,
        "is_inverted": is_inverted,
        "interpretation": (
            f"Yield curve {'INVERTED' if is_inverted else 'normal'}. "
            f"Slope (10Y-3M): {slope_10y_3m:+.3f}% " if slope_10y_3m is not None else ""
            f"{'— recession signal historically leads by 12-18 months. ' if is_inverted else ''}"
            f"10Y: {y_10y:.3f}%" if y_10y is not None else ""
        ),
    })


@function_tool
async def get_economic_calendar(
    ctx: RunContextWrapper[AppContext],
) -> str:
    """Return a static list of major recurring US economic events with typical market impact. For real-time calendar, use WebSearchTool. Events: FOMC, CPI, NFP, GDP, PCE."""
    # Static calendar of major recurring events
    events = [
        {
            "event": "FOMC Rate Decision",
            "frequency": "8x/year (~6 weeks)",
            "typical_impact": "high",
            "affected_assets": ["SPY", "QQQ", "TLT", "GLD"],
            "description": "Federal Reserve interest rate decision and statement. Forward guidance moves bonds and equities.",
        },
        {
            "event": "CPI (Consumer Price Index)",
            "frequency": "monthly (2nd week)",
            "typical_impact": "high",
            "affected_assets": ["SPY", "TLT", "GLD"],
            "description": "Inflation gauge. Surprise above consensus = hawkish rates, bearish bonds.",
        },
        {
            "event": "Non-Farm Payrolls (NFP)",
            "frequency": "monthly (1st Friday)",
            "typical_impact": "high",
            "affected_assets": ["SPY", "IWM", "TLT"],
            "description": "Employment report. Strong = hawkish rates, mixed equity impact.",
        },
        {
            "event": "GDP (Gross Domestic Product)",
            "frequency": "quarterly (advance/2nd/final)",
            "typical_impact": "medium",
            "affected_assets": ["SPY", "IWM"],
            "description": "Economic growth measure. Below expectations = recession fears.",
        },
        {
            "event": "PCE Deflator",
            "frequency": "monthly (last week)",
            "typical_impact": "high",
            "affected_assets": ["SPY", "TLT"],
            "description": "Fed's preferred inflation measure. Core PCE drives rate expectations.",
        },
        {
            "event": "ISM Manufacturing PMI",
            "frequency": "monthly (1st business day)",
            "typical_impact": "medium",
            "affected_assets": ["SPY", "XLI", "IWM"],
            "description": "Leading indicator. Below 50 = contraction. New orders most forward-looking.",
        },
        {
            "event": "Retail Sales",
            "frequency": "monthly (mid-month)",
            "typical_impact": "medium",
            "affected_assets": ["SPY", "XLY"],
            "description": "Consumer spending gauge. Strong = economic resilience.",
        },
    ]

    return json.dumps({
        "events": events,
        "note": "These are recurring events. For specific dates and expectations, use WebSearchTool with query 'US economic calendar this week'.",
    })


@function_tool
async def compute_macro_regime(
    ctx: RunContextWrapper[AppContext],
) -> str:
    """Compute current macro regime from yield curve + credit spread + market momentum. Returns: expansion, slowdown, contraction, or recovery. Uses yfinance data (no API key needed)."""
    import yfinance as yf

    signals = {}

    # 1. Yield curve slope (10Y - 3M)
    yields = _fetch_treasury_yields()
    if yields and "10y" in yields and "3m" in yields:
        slope = yields["10y"] - yields["3m"]
        signals["yield_curve_slope"] = round(slope, 3)
        signals["yield_curve_signal"] = (
            "inverted" if slope < -0.5 else
            "flat" if slope < 0.5 else
            "steep" if slope > 1.5 else
            "normal"
        )
    else:
        signals["yield_curve_signal"] = "unavailable"

    # 2. Credit spread proxy: HYG vs IEF (high yield vs investment grade)
    try:
        credit_data = yf.download(["HYG", "IEF"], period="6mo", progress=False)
        if not credit_data.empty:
            hyg_ret = credit_data["Close"]["HYG"].pct_change().dropna()
            ief_ret = credit_data["Close"]["IEF"].pct_change().dropna()
            spread_proxy = (ief_ret - hyg_ret).rolling(21).mean().iloc[-1]
            signals["credit_spread_proxy"] = round(float(spread_proxy) * 252 * 100, 2)
            signals["credit_signal"] = (
                "stress" if spread_proxy > 0.001 else
                "tightening" if spread_proxy < -0.001 else
                "neutral"
            )
    except Exception:
        signals["credit_signal"] = "unavailable"

    # 3. Equity momentum: SPY 50d vs 200d SMA
    try:
        spy = yf.download("SPY", period="1y", progress=False)
        if not spy.empty:
            close = spy["Close"]
            sma50 = float(close.rolling(50).mean().iloc[-1])
            sma200 = float(close.rolling(200).mean().iloc[-1])
            current = float(close.iloc[-1])

            signals["spy_vs_sma50"] = round((current / sma50 - 1) * 100, 2)
            signals["spy_vs_sma200"] = round((current / sma200 - 1) * 100, 2)
            signals["sma50_vs_sma200"] = round((sma50 / sma200 - 1) * 100, 2)
            signals["equity_signal"] = (
                "bullish" if current > sma50 > sma200 else
                "bearish" if current < sma50 < sma200 else
                "mixed"
            )
    except Exception:
        signals["equity_signal"] = "unavailable"

    # 4. VIX level
    try:
        vix = yf.download("^VIX", period="1mo", progress=False)
        if not vix.empty:
            vix_level = float(vix["Close"].iloc[-1])
            signals["vix"] = round(vix_level, 2)
            signals["vol_signal"] = (
                "panic" if vix_level > 30 else
                "elevated" if vix_level > 20 else
                "calm" if vix_level < 15 else
                "normal"
            )
    except Exception:
        signals["vol_signal"] = "unavailable"

    # Composite regime determination
    regime_scores = {"expansion": 0, "slowdown": 0, "contraction": 0, "recovery": 0}

    yc = signals.get("yield_curve_signal", "unavailable")
    if yc == "steep":
        regime_scores["expansion"] += 2
    elif yc == "normal":
        regime_scores["expansion"] += 1
    elif yc == "flat":
        regime_scores["slowdown"] += 2
    elif yc == "inverted":
        regime_scores["contraction"] += 2

    cs = signals.get("credit_signal", "unavailable")
    if cs == "tightening":
        regime_scores["expansion"] += 1
    elif cs == "stress":
        regime_scores["contraction"] += 2

    eq = signals.get("equity_signal", "unavailable")
    if eq == "bullish":
        regime_scores["expansion"] += 2
    elif eq == "bearish":
        regime_scores["contraction"] += 2
    elif eq == "mixed":
        regime_scores["slowdown"] += 1

    vs = signals.get("vol_signal", "unavailable")
    if vs == "calm":
        regime_scores["expansion"] += 1
    elif vs == "elevated":
        regime_scores["slowdown"] += 1
    elif vs == "panic":
        regime_scores["contraction"] += 2

    # Determine regime
    regime = max(regime_scores, key=regime_scores.get)
    confidence = regime_scores[regime] / max(sum(regime_scores.values()), 1)

    return json.dumps({
        "regime": regime,
        "confidence": round(confidence, 2),
        "regime_scores": regime_scores,
        "signals": signals,
        "interpretation": (
            f"Macro regime: {regime.upper()} (confidence: {confidence:.0%}). "
            f"Yield curve: {signals.get('yield_curve_signal', 'N/A')}, "
            f"Credit: {signals.get('credit_signal', 'N/A')}, "
            f"Equity: {signals.get('equity_signal', 'N/A')}, "
            f"Volatility: {signals.get('vol_signal', 'N/A')}."
        ),
    })
