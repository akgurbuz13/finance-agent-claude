"""Technical analysis indicator computations."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from agents import RunContextWrapper, function_tool

from portfolio_advisor.agents.context import AppContext


def _prices_to_series(prices_json: str) -> pd.DataFrame:
    """Convert JSON price bars to a DataFrame with OHLCV columns."""
    bars = json.loads(prices_json) if isinstance(prices_json, str) else prices_json
    df = pd.DataFrame(bars)
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    for col in ("open", "high", "low", "close", "volume"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _interpret_level(value: float, overbought: float, oversold: float) -> str:
    if value >= overbought:
        return "bearish"
    elif value <= oversold:
        return "bullish"
    return "neutral"


# ── Pure computation functions (no ctx, return dicts) ────────────────────────


def compute_sma_ema_raw(df: pd.DataFrame) -> dict:
    """Compute SMA(50/200), EMA(12/26), golden/death cross — pure function."""
    close = df["close"]

    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()

    last_price = float(close.iloc[-1])
    last_sma50 = float(sma50.iloc[-1]) if not np.isnan(sma50.iloc[-1]) else None
    last_sma200 = float(sma200.iloc[-1]) if not np.isnan(sma200.iloc[-1]) else None
    last_ema12 = float(ema12.iloc[-1])
    last_ema26 = float(ema26.iloc[-1])

    # Trend direction
    trend = "neutral"
    if last_sma50 and last_sma200:
        if last_sma50 > last_sma200 and last_price > last_sma50:
            trend = "bullish"
        elif last_sma50 < last_sma200 and last_price < last_sma50:
            trend = "bearish"

    # Golden / Death cross detection (last 5 bars)
    cross = "none"
    if last_sma50 and last_sma200 and len(sma50.dropna()) >= 5:
        recent_diff = (sma50 - sma200).dropna().iloc[-5:]
        if len(recent_diff) >= 2:
            if recent_diff.iloc[-1] > 0 and recent_diff.iloc[-2] <= 0:
                cross = "golden_cross"
            elif recent_diff.iloc[-1] < 0 and recent_diff.iloc[-2] >= 0:
                cross = "death_cross"

    # EMA crossover
    ema_signal = "bullish" if last_ema12 > last_ema26 else "bearish"

    confidence = 0.5
    if trend != "neutral":
        confidence = 0.7
    if cross != "none":
        confidence = 0.85

    interpretation = trend
    if cross == "golden_cross":
        interpretation = "bullish"
    elif cross == "death_cross":
        interpretation = "bearish"

    return {
        "price": last_price,
        "sma50": last_sma50,
        "sma200": last_sma200,
        "ema12": last_ema12,
        "ema26": last_ema26,
        "cross": cross,
        "ema_signal": ema_signal,
        "trend": trend,
        "interpretation": interpretation,
        "confidence": round(confidence, 2),
    }


def compute_rsi_raw(df: pd.DataFrame, period: int = 14) -> dict:
    """Compute RSI with overbought/oversold interpretation — pure function."""
    delta = df["close"].diff()

    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi_val = float(rsi.iloc[-1]) if not np.isnan(rsi.iloc[-1]) else 50.0

    interpretation = _interpret_level(rsi_val, 70, 30)
    confidence = 0.6 if 40 <= rsi_val <= 60 else 0.75

    # Simple divergence check: price making new highs but RSI not
    divergence = "none"
    if len(df) >= 20:
        price_high = df["close"].iloc[-10:].max() >= df["close"].iloc[-20:-10].max()
        rsi_high = rsi.iloc[-10:].max() >= rsi.iloc[-20:-10].max()
        if price_high and not rsi_high:
            divergence = "bearish_divergence"
        elif not price_high and rsi_high:
            divergence = "bullish_divergence"

    return {
        "rsi": round(rsi_val, 2),
        "period": period,
        "divergence": divergence,
        "interpretation": interpretation,
        "confidence": round(confidence, 2),
    }


def compute_macd_raw(df: pd.DataFrame) -> dict:
    """Compute MACD line, signal, histogram, and crossover — pure function."""
    close = df["close"]

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    histogram = macd_line - signal_line

    last_macd = float(macd_line.iloc[-1])
    last_signal = float(signal_line.iloc[-1])
    last_hist = float(histogram.iloc[-1])

    # Crossover detection
    crossover = "none"
    if len(histogram.dropna()) >= 2:
        if histogram.iloc[-1] > 0 and histogram.iloc[-2] <= 0:
            crossover = "bullish_crossover"
        elif histogram.iloc[-1] < 0 and histogram.iloc[-2] >= 0:
            crossover = "bearish_crossover"

    interpretation = "bullish" if last_macd > last_signal else "bearish"
    confidence = 0.65
    if crossover != "none":
        confidence = 0.8

    return {
        "macd_line": round(last_macd, 4),
        "signal_line": round(last_signal, 4),
        "histogram": round(last_hist, 4),
        "crossover": crossover,
        "interpretation": interpretation,
        "confidence": round(confidence, 2),
    }


def compute_atr_bollinger_raw(df: pd.DataFrame) -> dict:
    """Compute ATR(14) and Bollinger Bands (20,2) — pure function."""
    # ATR
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()

    # Bollinger Bands
    sma20 = df["close"].rolling(20).mean()
    std20 = df["close"].rolling(20).std()
    upper = sma20 + 2 * std20
    lower = sma20 - 2 * std20

    last_close = float(df["close"].iloc[-1])
    last_atr = float(atr.iloc[-1]) if not np.isnan(atr.iloc[-1]) else 0.0
    last_upper = float(upper.iloc[-1]) if not np.isnan(upper.iloc[-1]) else last_close
    last_lower = float(lower.iloc[-1]) if not np.isnan(lower.iloc[-1]) else last_close
    last_sma20 = float(sma20.iloc[-1]) if not np.isnan(sma20.iloc[-1]) else last_close

    bandwidth = (last_upper - last_lower) / last_sma20 if last_sma20 != 0 else 0
    pct_b = (
        (last_close - last_lower) / (last_upper - last_lower)
        if (last_upper - last_lower) != 0
        else 0.5
    )

    interpretation = "neutral"
    if pct_b > 1.0:
        interpretation = "bearish"  # Above upper band
    elif pct_b < 0.0:
        interpretation = "bullish"  # Below lower band
    elif bandwidth < 0.05:
        interpretation = "neutral"  # Squeeze — expect breakout

    return {
        "atr_14": round(last_atr, 4),
        "bb_upper": round(last_upper, 4),
        "bb_middle": round(last_sma20, 4),
        "bb_lower": round(last_lower, 4),
        "bandwidth": round(bandwidth, 4),
        "pct_b": round(pct_b, 4),
        "interpretation": interpretation,
        "confidence": round(0.6 if interpretation == "neutral" else 0.7, 2),
    }


def compute_support_resistance_raw(df: pd.DataFrame) -> dict:
    """Compute pivot-based support and resistance levels — pure function."""
    recent = df.iloc[-20:]
    high = float(recent["high"].max())
    low = float(recent["low"].min())
    close = float(df["close"].iloc[-1])

    pivot = (high + low + close) / 3
    r1 = 2 * pivot - low
    s1 = 2 * pivot - high
    r2 = pivot + (high - low)
    s2 = pivot - (high - low)
    r3 = high + 2 * (pivot - low)
    s3 = low - 2 * (high - pivot)

    # Determine proximity
    nearest_support = (
        max(s for s in [s1, s2, s3] if s < close)
        if any(s < close for s in [s1, s2, s3])
        else s1
    )
    nearest_resistance = (
        min(r for r in [r1, r2, r3] if r > close)
        if any(r > close for r in [r1, r2, r3])
        else r1
    )

    dist_to_support_pct = ((close - nearest_support) / close) * 100
    dist_to_resistance_pct = ((nearest_resistance - close) / close) * 100

    interpretation = "neutral"
    if dist_to_support_pct < 1.0:
        interpretation = "bullish"  # Near support
    elif dist_to_resistance_pct < 1.0:
        interpretation = "bearish"  # Near resistance

    return {
        "pivot": round(pivot, 2),
        "r1": round(r1, 2),
        "r2": round(r2, 2),
        "r3": round(r3, 2),
        "s1": round(s1, 2),
        "s2": round(s2, 2),
        "s3": round(s3, 2),
        "nearest_support": round(nearest_support, 2),
        "nearest_resistance": round(nearest_resistance, 2),
        "dist_to_support_pct": round(dist_to_support_pct, 2),
        "dist_to_resistance_pct": round(dist_to_resistance_pct, 2),
        "interpretation": interpretation,
        "confidence": 0.55,
    }


# ── @function_tool wrappers (thin layer over raw functions) ──────────────────


@function_tool
async def compute_sma_ema_crossovers(
    ctx: RunContextWrapper[AppContext],
    ticker: str,
    prices_json: str,
) -> str:
    """Compute SMA(50/200), EMA(12/26), golden/death cross signals."""
    df = _prices_to_series(prices_json)
    raw = compute_sma_ema_raw(df)
    result = {
        "ticker": ticker,
        "indicator": "sma_ema_crossovers",
        "values": {
            "price": raw["price"],
            "sma50": raw["sma50"],
            "sma200": raw["sma200"],
            "ema12": raw["ema12"],
            "ema26": raw["ema26"],
            "cross": raw["cross"],
            "ema_signal": raw["ema_signal"],
        },
        "trend": raw["trend"],
        "interpretation": raw["interpretation"],
        "confidence": raw["confidence"],
    }
    return json.dumps(result)


@function_tool
async def compute_rsi(
    ctx: RunContextWrapper[AppContext],
    ticker: str,
    prices_json: str,
    period: int = 14,
) -> str:
    """Compute RSI with overbought/oversold interpretation."""
    df = _prices_to_series(prices_json)
    raw = compute_rsi_raw(df, period)
    result = {
        "ticker": ticker,
        "indicator": "rsi",
        "values": {"rsi": raw["rsi"], "period": raw["period"], "divergence": raw["divergence"]},
        "interpretation": raw["interpretation"],
        "confidence": raw["confidence"],
    }
    return json.dumps(result)


@function_tool
async def compute_macd(
    ctx: RunContextWrapper[AppContext],
    ticker: str,
    prices_json: str,
) -> str:
    """Compute MACD line, signal, histogram, and crossover detection."""
    df = _prices_to_series(prices_json)
    raw = compute_macd_raw(df)
    result = {
        "ticker": ticker,
        "indicator": "macd",
        "values": {
            "macd_line": raw["macd_line"],
            "signal_line": raw["signal_line"],
            "histogram": raw["histogram"],
            "crossover": raw["crossover"],
        },
        "interpretation": raw["interpretation"],
        "confidence": raw["confidence"],
    }
    return json.dumps(result)


@function_tool
async def compute_atr_bollinger(
    ctx: RunContextWrapper[AppContext],
    ticker: str,
    prices_json: str,
) -> str:
    """Compute ATR(14) and Bollinger Bands (20,2) with bandwidth and %B."""
    df = _prices_to_series(prices_json)
    raw = compute_atr_bollinger_raw(df)
    result = {
        "ticker": ticker,
        "indicator": "atr_bollinger",
        "values": {
            "atr_14": raw["atr_14"],
            "bb_upper": raw["bb_upper"],
            "bb_middle": raw["bb_middle"],
            "bb_lower": raw["bb_lower"],
            "bandwidth": raw["bandwidth"],
            "pct_b": raw["pct_b"],
        },
        "interpretation": raw["interpretation"],
        "confidence": raw["confidence"],
    }
    return json.dumps(result)


@function_tool
async def compute_support_resistance(
    ctx: RunContextWrapper[AppContext],
    ticker: str,
    prices_json: str,
) -> str:
    """Compute pivot-based support and resistance levels (S1-S3, R1-R3)."""
    df = _prices_to_series(prices_json)
    raw = compute_support_resistance_raw(df)
    result = {
        "ticker": ticker,
        "indicator": "support_resistance",
        "values": {
            "pivot": raw["pivot"],
            "r1": raw["r1"],
            "r2": raw["r2"],
            "r3": raw["r3"],
            "s1": raw["s1"],
            "s2": raw["s2"],
            "s3": raw["s3"],
            "nearest_support": raw["nearest_support"],
            "nearest_resistance": raw["nearest_resistance"],
            "dist_to_support_pct": raw["dist_to_support_pct"],
            "dist_to_resistance_pct": raw["dist_to_resistance_pct"],
        },
        "interpretation": raw["interpretation"],
        "confidence": raw["confidence"],
    }
    return json.dumps(result)


@function_tool
async def compute_weekly_signals(
    ctx: RunContextWrapper[AppContext],
    ticker: str,
    period: str = "2y",
) -> str:
    """Compute weekly-timeframe SMA/RSI/MACD for multi-timeframe confirmation."""
    import yfinance as yf

    df = yf.download(ticker, period=period, interval="1wk", progress=False)
    if df.empty:
        return json.dumps({"ticker": ticker, "error": "No weekly data available"})

    bars = []
    for idx, row in df.iterrows():
        bars.append({
            "date": idx.strftime("%Y-%m-%d"),
            "open": float(row["Open"]),
            "high": float(row["High"]),
            "low": float(row["Low"]),
            "close": float(row["Close"]),
            "volume": float(row.get("Volume", 0)),
        })

    prices_json = json.dumps(bars)

    # Reuse the daily functions on weekly data
    df_w = _prices_to_series(prices_json)
    close = df_w["close"]

    # Weekly SMA
    sma20w = close.rolling(20).mean()

    # Weekly RSI
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(14).mean()
    loss = (-delta).where(delta < 0, 0.0).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi_w = 100 - (100 / (1 + rs))

    # Weekly MACD
    ema12w = close.ewm(span=12, adjust=False).mean()
    ema26w = close.ewm(span=26, adjust=False).mean()
    macd_w = ema12w - ema26w
    signal_w = macd_w.ewm(span=9, adjust=False).mean()

    last_close = float(close.iloc[-1])
    last_sma20w = float(sma20w.iloc[-1]) if not np.isnan(sma20w.iloc[-1]) else None
    last_rsi_w = float(rsi_w.iloc[-1]) if not np.isnan(rsi_w.iloc[-1]) else 50.0
    last_macd_w = float(macd_w.iloc[-1]) if not np.isnan(macd_w.iloc[-1]) else 0.0
    last_signal_w = float(signal_w.iloc[-1]) if not np.isnan(signal_w.iloc[-1]) else 0.0

    weekly_trend = "neutral"
    if last_sma20w and last_close > last_sma20w and last_macd_w > last_signal_w:
        weekly_trend = "bullish"
    elif last_sma20w and last_close < last_sma20w and last_macd_w < last_signal_w:
        weekly_trend = "bearish"

    result = {
        "ticker": ticker,
        "indicator": "weekly_signals",
        "timeframe": "weekly",
        "values": {
            "close": last_close,
            "sma20w": last_sma20w,
            "rsi_weekly": round(last_rsi_w, 2),
            "macd_weekly": round(last_macd_w, 4),
            "signal_weekly": round(last_signal_w, 4),
        },
        "weekly_trend": weekly_trend,
        "interpretation": weekly_trend,
        "confidence": 0.7 if weekly_trend != "neutral" else 0.5,
    }
    return json.dumps(result)
