"""Advanced technical indicators — Ichimoku, VWAP, OBV, ADX, Stochastic, Fibonacci, Volume Profile."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from agents import RunContextWrapper, function_tool

from portfolio_advisor.agents.context import AppContext
from portfolio_advisor.tools.technical_indicators import _prices_to_series


# ── Pure computation functions (no ctx, return dicts) ────────────────────────


def compute_ichimoku_raw(df: pd.DataFrame) -> dict:
    """Compute Ichimoku Cloud components — pure function.

    Tenkan-sen (conversion, 9), Kijun-sen (base, 26), Senkou A/B (cloud),
    Chikou (lagging, 26-period shift).
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]

    # Tenkan-sen (conversion line): (9-period high + 9-period low) / 2
    tenkan = (high.rolling(9).max() + low.rolling(9).min()) / 2

    # Kijun-sen (base line): (26-period high + 26-period low) / 2
    kijun = (high.rolling(26).max() + low.rolling(26).min()) / 2

    # Senkou Span A (leading span A): (Tenkan + Kijun) / 2, shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2).shift(26)

    # Senkou Span B (leading span B): (52-period high + 52-period low) / 2, shifted 26 ahead
    senkou_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)

    # Chikou Span (lagging span): close shifted 26 periods back
    chikou = close.shift(-26)

    last_close = float(close.iloc[-1])
    last_tenkan = float(tenkan.iloc[-1]) if not np.isnan(tenkan.iloc[-1]) else None
    last_kijun = float(kijun.iloc[-1]) if not np.isnan(kijun.iloc[-1]) else None

    # Current cloud edges (use the values from 26 bars ago, which represent today's cloud)
    current_senkou_a = float(senkou_a.iloc[-1]) if not np.isnan(senkou_a.iloc[-1]) else None
    current_senkou_b = float(senkou_b.iloc[-1]) if not np.isnan(senkou_b.iloc[-1]) else None

    last_chikou = float(chikou.dropna().iloc[-1]) if not chikou.dropna().empty else None

    # Signal interpretation
    interpretation = "neutral"
    confidence = 0.5
    signals = []

    if current_senkou_a is not None and current_senkou_b is not None:
        cloud_top = max(current_senkou_a, current_senkou_b)
        cloud_bottom = min(current_senkou_a, current_senkou_b)

        if last_close > cloud_top:
            signals.append("above_cloud")
            interpretation = "bullish"
            confidence = 0.7
        elif last_close < cloud_bottom:
            signals.append("below_cloud")
            interpretation = "bearish"
            confidence = 0.7
        else:
            signals.append("inside_cloud")
            interpretation = "neutral"
            confidence = 0.4

    # TK cross (Tenkan crosses above/below Kijun)
    if last_tenkan is not None and last_kijun is not None:
        if len(tenkan.dropna()) >= 2 and len(kijun.dropna()) >= 2:
            tk_diff = (tenkan - kijun).dropna()
            if len(tk_diff) >= 2:
                if tk_diff.iloc[-1] > 0 and tk_diff.iloc[-2] <= 0:
                    signals.append("bullish_tk_cross")
                    confidence = min(0.85, confidence + 0.15)
                elif tk_diff.iloc[-1] < 0 and tk_diff.iloc[-2] >= 0:
                    signals.append("bearish_tk_cross")
                    confidence = min(0.85, confidence + 0.15)

    # Cloud twist (future cloud direction change)
    future_a = ((tenkan + kijun) / 2)
    future_b = (high.rolling(52).max() + low.rolling(52).min()) / 2
    if len(future_a.dropna()) >= 2 and len(future_b.dropna()) >= 2:
        twist_diff = (future_a - future_b).dropna()
        if len(twist_diff) >= 2:
            if twist_diff.iloc[-1] > 0 and twist_diff.iloc[-2] <= 0:
                signals.append("bullish_cloud_twist")
            elif twist_diff.iloc[-1] < 0 and twist_diff.iloc[-2] >= 0:
                signals.append("bearish_cloud_twist")

    return {
        "tenkan": round(last_tenkan, 4) if last_tenkan else None,
        "kijun": round(last_kijun, 4) if last_kijun else None,
        "senkou_a": round(current_senkou_a, 4) if current_senkou_a else None,
        "senkou_b": round(current_senkou_b, 4) if current_senkou_b else None,
        "chikou": round(last_chikou, 4) if last_chikou else None,
        "price_vs_cloud": signals[0] if signals else "insufficient_data",
        "signals": signals,
        "interpretation": interpretation,
        "confidence": round(confidence, 2),
    }


def compute_vwap_raw(df: pd.DataFrame, window: int = 20) -> dict:
    """Compute rolling VWAP (Volume-Weighted Average Price) — pure function."""
    close = df["close"]
    volume = df["volume"] if "volume" in df.columns else pd.Series(1.0, index=df.index)

    # Typical price
    typical_price = (df["high"] + df["low"] + close) / 3

    # Rolling VWAP
    cum_tp_vol = (typical_price * volume).rolling(window).sum()
    cum_vol = volume.rolling(window).sum()
    vwap = cum_tp_vol / cum_vol.replace(0, np.nan)

    last_close = float(close.iloc[-1])
    last_vwap = float(vwap.iloc[-1]) if not np.isnan(vwap.iloc[-1]) else None

    interpretation = "neutral"
    confidence = 0.55

    if last_vwap is not None:
        pct_diff = ((last_close - last_vwap) / last_vwap) * 100
        if pct_diff > 1.0:
            interpretation = "bullish"
            confidence = 0.65
        elif pct_diff < -1.0:
            interpretation = "bearish"
            confidence = 0.65

        # Check volume significance
        avg_vol = float(volume.tail(window).mean())
        last_vol = float(volume.iloc[-1]) if not volume.empty else 0
        high_volume = last_vol > avg_vol * 1.5
    else:
        pct_diff = 0.0
        high_volume = False

    return {
        "vwap": round(last_vwap, 4) if last_vwap else None,
        "price": last_close,
        "price_vs_vwap_pct": round(pct_diff, 2),
        "window": window,
        "high_volume_confirmation": high_volume,
        "interpretation": interpretation,
        "confidence": round(confidence, 2),
    }


def compute_obv_raw(df: pd.DataFrame) -> dict:
    """Compute On-Balance Volume with divergence detection — pure function."""
    close = df["close"]
    volume = df["volume"] if "volume" in df.columns else pd.Series(0.0, index=df.index)

    # OBV calculation
    direction = np.sign(close.diff())
    obv = (direction * volume).fillna(0).cumsum()

    last_obv = float(obv.iloc[-1])

    # OBV trend (20-period SMA of OBV)
    obv_sma = obv.rolling(20).mean()
    obv_trending_up = float(obv.iloc[-1]) > float(obv_sma.iloc[-1]) if not np.isnan(
        obv_sma.iloc[-1]
    ) else None

    # Divergence detection (last 20 bars)
    divergence = "none"
    if len(df) >= 20:
        # Compare first half vs second half of last 20 bars
        price_first = close.iloc[-20:-10].mean()
        price_second = close.iloc[-10:].mean()
        obv_first = obv.iloc[-20:-10].mean()
        obv_second = obv.iloc[-10:].mean()

        price_rising = price_second > price_first
        obv_rising = obv_second > obv_first

        if price_rising and not obv_rising:
            divergence = "bearish_divergence"
        elif not price_rising and obv_rising:
            divergence = "bullish_divergence"

    interpretation = "neutral"
    confidence = 0.55

    if divergence == "bearish_divergence":
        interpretation = "bearish"
        confidence = 0.7
    elif divergence == "bullish_divergence":
        interpretation = "bullish"
        confidence = 0.7
    elif obv_trending_up is True:
        interpretation = "bullish"
        confidence = 0.6
    elif obv_trending_up is False:
        interpretation = "bearish"
        confidence = 0.6

    return {
        "obv": round(last_obv, 0),
        "obv_trending_up": obv_trending_up,
        "divergence": divergence,
        "interpretation": interpretation,
        "confidence": round(confidence, 2),
    }


def compute_adx_dmi_raw(df: pd.DataFrame, period: int = 14) -> dict:
    """Compute ADX trend strength + DMI directional indicators — pure function."""
    high = df["high"]
    low = df["low"]
    close = df["close"]

    # +DM and -DM
    plus_dm = high.diff()
    minus_dm = -low.diff()

    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    # True Range
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Smoothed with Wilder's method (equivalent to EMA with alpha=1/period)
    atr = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    smooth_plus_dm = plus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    smooth_minus_dm = minus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    # +DI and -DI
    plus_di = 100 * smooth_plus_dm / atr.replace(0, np.nan)
    minus_di = 100 * smooth_minus_dm / atr.replace(0, np.nan)

    # DX and ADX
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    last_adx = float(adx.iloc[-1]) if not np.isnan(adx.iloc[-1]) else None
    last_plus_di = float(plus_di.iloc[-1]) if not np.isnan(plus_di.iloc[-1]) else None
    last_minus_di = float(minus_di.iloc[-1]) if not np.isnan(minus_di.iloc[-1]) else None

    # Interpretation
    interpretation = "neutral"
    confidence = 0.5
    trend_strength = "no_trend"

    if last_adx is not None:
        if last_adx >= 40:
            trend_strength = "strong_trend"
            confidence = 0.8
        elif last_adx >= 25:
            trend_strength = "trending"
            confidence = 0.65
        elif last_adx >= 20:
            trend_strength = "weak_trend"
            confidence = 0.55
        else:
            trend_strength = "no_trend"
            confidence = 0.5

        if last_plus_di is not None and last_minus_di is not None:
            if last_plus_di > last_minus_di and last_adx >= 20:
                interpretation = "bullish"
            elif last_minus_di > last_plus_di and last_adx >= 20:
                interpretation = "bearish"

    # DI crossover detection
    di_crossover = "none"
    if last_plus_di is not None and len(plus_di.dropna()) >= 2:
        di_diff = (plus_di - minus_di).dropna()
        if len(di_diff) >= 2:
            if di_diff.iloc[-1] > 0 and di_diff.iloc[-2] <= 0:
                di_crossover = "bullish_di_cross"
            elif di_diff.iloc[-1] < 0 and di_diff.iloc[-2] >= 0:
                di_crossover = "bearish_di_cross"

    return {
        "adx": round(last_adx, 2) if last_adx else None,
        "plus_di": round(last_plus_di, 2) if last_plus_di else None,
        "minus_di": round(last_minus_di, 2) if last_minus_di else None,
        "trend_strength": trend_strength,
        "di_crossover": di_crossover,
        "interpretation": interpretation,
        "confidence": round(confidence, 2),
    }


def compute_stochastic_raw(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> dict:
    """Compute Stochastic Oscillator (%K/%D) with crossover detection — pure function."""
    high = df["high"]
    low = df["low"]
    close = df["close"]

    # %K = (Close - Lowest Low) / (Highest High - Lowest Low) * 100
    lowest_low = low.rolling(k_period).min()
    highest_high = high.rolling(k_period).max()
    k_line = 100 * (close - lowest_low) / (highest_high - lowest_low).replace(0, np.nan)

    # %D = SMA of %K
    d_line = k_line.rolling(d_period).mean()

    last_k = float(k_line.iloc[-1]) if not np.isnan(k_line.iloc[-1]) else 50.0
    last_d = float(d_line.iloc[-1]) if not np.isnan(d_line.iloc[-1]) else 50.0

    # Zone
    zone = "neutral"
    if last_k < 20:
        zone = "oversold"
    elif last_k > 80:
        zone = "overbought"

    # Crossover detection
    crossover = "none"
    if len(k_line.dropna()) >= 2 and len(d_line.dropna()) >= 2:
        kd_diff = (k_line - d_line).dropna()
        if len(kd_diff) >= 2:
            if kd_diff.iloc[-1] > 0 and kd_diff.iloc[-2] <= 0:
                crossover = "bullish_crossover"
            elif kd_diff.iloc[-1] < 0 and kd_diff.iloc[-2] >= 0:
                crossover = "bearish_crossover"

    # Signal logic: oversold + bullish crossover = strong buy; overbought + bearish = strong sell
    interpretation = "neutral"
    confidence = 0.55

    if zone == "oversold" and crossover == "bullish_crossover":
        interpretation = "bullish"
        confidence = 0.8
    elif zone == "overbought" and crossover == "bearish_crossover":
        interpretation = "bearish"
        confidence = 0.8
    elif zone == "oversold":
        interpretation = "bullish"
        confidence = 0.65
    elif zone == "overbought":
        interpretation = "bearish"
        confidence = 0.65
    elif crossover == "bullish_crossover":
        interpretation = "bullish"
        confidence = 0.6
    elif crossover == "bearish_crossover":
        interpretation = "bearish"
        confidence = 0.6

    return {
        "k": round(last_k, 2),
        "d": round(last_d, 2),
        "zone": zone,
        "crossover": crossover,
        "interpretation": interpretation,
        "confidence": round(confidence, 2),
    }


def compute_fibonacci_raw(df: pd.DataFrame, lookback: int = 60) -> dict:
    """Compute Fibonacci retracement levels from swing high/low — pure function."""
    close = df["close"]
    recent = df.iloc[-lookback:] if len(df) >= lookback else df

    swing_high = float(recent["high"].max())
    swing_low = float(recent["low"].min())
    last_close = float(close.iloc[-1])

    diff = swing_high - swing_low
    if diff == 0:
        return {
            "swing_high": swing_high,
            "swing_low": swing_low,
            "levels": {},
            "current_zone": "undefined",
            "interpretation": "neutral",
            "confidence": 0.4,
        }

    # Fibonacci retracement levels (measured from high)
    levels = {
        "0.0": round(swing_high, 4),
        "23.6": round(swing_high - 0.236 * diff, 4),
        "38.2": round(swing_high - 0.382 * diff, 4),
        "50.0": round(swing_high - 0.500 * diff, 4),
        "61.8": round(swing_high - 0.618 * diff, 4),
        "78.6": round(swing_high - 0.786 * diff, 4),
        "100.0": round(swing_low, 4),
    }

    # Determine current zone
    fib_pct = ((swing_high - last_close) / diff) * 100
    current_zone = "above_swing_high"
    if fib_pct <= 0:
        current_zone = "above_swing_high"
    elif fib_pct <= 23.6:
        current_zone = "0_to_23.6"
    elif fib_pct <= 38.2:
        current_zone = "23.6_to_38.2"
    elif fib_pct <= 50.0:
        current_zone = "38.2_to_50.0"
    elif fib_pct <= 61.8:
        current_zone = "50.0_to_61.8"
    elif fib_pct <= 78.6:
        current_zone = "61.8_to_78.6"
    else:
        current_zone = "below_78.6"

    # Find nearest support/resistance fib levels
    fib_values = sorted(levels.values(), reverse=True)
    nearest_above = next((v for v in fib_values if v > last_close), swing_high)
    nearest_below = next((v for v in reversed(fib_values) if v < last_close), swing_low)

    # Interpretation: near key levels (38.2% or 61.8%) = potential reversal
    interpretation = "neutral"
    confidence = 0.5

    dist_to_382 = abs(last_close - levels["38.2"]) / last_close * 100
    dist_to_618 = abs(last_close - levels["61.8"]) / last_close * 100

    if dist_to_382 < 1.0 or dist_to_618 < 1.0:
        # Near a golden ratio level — potential reversal zone
        if last_close > levels["50.0"]:
            interpretation = "bullish"  # Holding above midpoint after retest
        else:
            interpretation = "bearish"  # Failing to hold midpoint
        confidence = 0.65

    return {
        "swing_high": swing_high,
        "swing_low": swing_low,
        "levels": levels,
        "current_zone": current_zone,
        "fib_retrace_pct": round(fib_pct, 2),
        "nearest_fib_above": nearest_above,
        "nearest_fib_below": nearest_below,
        "interpretation": interpretation,
        "confidence": round(confidence, 2),
    }


def compute_volume_profile_raw(df: pd.DataFrame, n_bins: int = 20) -> dict:
    """Compute Volume Profile (Volume-at-Price) with POC and HVN/LVN — pure function."""
    close = df["close"]
    volume = df["volume"] if "volume" in df.columns else pd.Series(1.0, index=df.index)

    price_min = float(close.min())
    price_max = float(close.max())

    if price_max == price_min or volume.sum() == 0:
        return {
            "poc": float(close.iloc[-1]),
            "hvn_zones": [],
            "lvn_zones": [],
            "value_area_high": float(close.iloc[-1]),
            "value_area_low": float(close.iloc[-1]),
            "interpretation": "neutral",
            "confidence": 0.4,
        }

    # Create price bins
    bin_edges = np.linspace(price_min, price_max, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Assign volume to bins
    bin_volumes = np.zeros(n_bins)
    for i in range(len(close)):
        price = float(close.iloc[i])
        vol = float(volume.iloc[i])
        bin_idx = min(int((price - price_min) / (price_max - price_min) * n_bins), n_bins - 1)
        bin_volumes[bin_idx] += vol

    # Point of Control (POC) — price level with highest volume
    poc_idx = int(np.argmax(bin_volumes))
    poc = float(bin_centers[poc_idx])

    # Value Area (70% of total volume around POC)
    total_volume = bin_volumes.sum()
    va_target = total_volume * 0.7

    # Expand from POC outward
    included = {poc_idx}
    va_volume = bin_volumes[poc_idx]
    lo, hi = poc_idx, poc_idx

    while va_volume < va_target and (lo > 0 or hi < n_bins - 1):
        expand_lo = bin_volumes[lo - 1] if lo > 0 else 0
        expand_hi = bin_volumes[hi + 1] if hi < n_bins - 1 else 0

        if expand_lo >= expand_hi and lo > 0:
            lo -= 1
            included.add(lo)
            va_volume += expand_lo
        elif hi < n_bins - 1:
            hi += 1
            included.add(hi)
            va_volume += expand_hi
        elif lo > 0:
            lo -= 1
            included.add(lo)
            va_volume += expand_lo
        else:
            break

    value_area_high = float(bin_edges[hi + 1])
    value_area_low = float(bin_edges[lo])

    # HVN (High Volume Nodes) — bins above 75th percentile
    vol_threshold_high = np.percentile(bin_volumes, 75)
    vol_threshold_low = np.percentile(bin_volumes, 25)

    hvn_zones = [
        round(float(bin_centers[i]), 4)
        for i in range(n_bins)
        if bin_volumes[i] >= vol_threshold_high
    ]
    lvn_zones = [
        round(float(bin_centers[i]), 4)
        for i in range(n_bins)
        if 0 < bin_volumes[i] <= vol_threshold_low
    ]

    # Interpretation
    last_close = float(close.iloc[-1])
    interpretation = "neutral"
    confidence = 0.55

    if last_close > value_area_high:
        interpretation = "bullish"  # Trading above value area
        confidence = 0.65
    elif last_close < value_area_low:
        interpretation = "bearish"  # Trading below value area
        confidence = 0.65
    elif abs(last_close - poc) / poc < 0.01:
        interpretation = "neutral"  # At fair value
        confidence = 0.6

    return {
        "poc": round(poc, 4),
        "value_area_high": round(value_area_high, 4),
        "value_area_low": round(value_area_low, 4),
        "hvn_zones": hvn_zones[:5],  # Top 5
        "lvn_zones": lvn_zones[:5],
        "price_vs_poc_pct": round(((last_close - poc) / poc) * 100, 2),
        "interpretation": interpretation,
        "confidence": round(confidence, 2),
    }


# ── @function_tool wrappers ──────────────────────────────────────────────────


@function_tool
async def compute_ichimoku_cloud(
    ctx: RunContextWrapper[AppContext],
    ticker: str,
    prices_json: str,
) -> str:
    """Compute Ichimoku Cloud: Tenkan/Kijun/Senkou A&B/Chikou. Cloud = dynamic S/R zone. Price above cloud = bullish; TK cross = entry signal; cloud twist = reversal."""
    df = _prices_to_series(prices_json)
    raw = compute_ichimoku_raw(df)
    result = {
        "ticker": ticker,
        "indicator": "ichimoku_cloud",
        "values": {
            "tenkan": raw["tenkan"],
            "kijun": raw["kijun"],
            "senkou_a": raw["senkou_a"],
            "senkou_b": raw["senkou_b"],
            "chikou": raw["chikou"],
            "price_vs_cloud": raw["price_vs_cloud"],
            "signals": raw["signals"],
        },
        "interpretation": raw["interpretation"],
        "confidence": raw["confidence"],
    }
    return json.dumps(result)


@function_tool
async def compute_vwap(
    ctx: RunContextWrapper[AppContext],
    ticker: str,
    prices_json: str,
) -> str:
    """Compute rolling 20-day VWAP (institutional benchmark). Price > VWAP = bullish institutional flow."""
    df = _prices_to_series(prices_json)
    raw = compute_vwap_raw(df)
    result = {
        "ticker": ticker,
        "indicator": "vwap",
        "values": {
            "vwap": raw["vwap"],
            "price": raw["price"],
            "price_vs_vwap_pct": raw["price_vs_vwap_pct"],
            "high_volume_confirmation": raw["high_volume_confirmation"],
        },
        "interpretation": raw["interpretation"],
        "confidence": raw["confidence"],
    }
    return json.dumps(result)


@function_tool
async def compute_obv(
    ctx: RunContextWrapper[AppContext],
    ticker: str,
    prices_json: str,
) -> str:
    """Compute On-Balance Volume with divergence detection. OBV diverging from price = reversal warning."""
    df = _prices_to_series(prices_json)
    raw = compute_obv_raw(df)
    result = {
        "ticker": ticker,
        "indicator": "obv",
        "values": {
            "obv": raw["obv"],
            "obv_trending_up": raw["obv_trending_up"],
            "divergence": raw["divergence"],
        },
        "interpretation": raw["interpretation"],
        "confidence": raw["confidence"],
    }
    return json.dumps(result)


@function_tool
async def compute_adx_dmi(
    ctx: RunContextWrapper[AppContext],
    ticker: str,
    prices_json: str,
) -> str:
    """Compute ADX trend strength + DMI directional indicators. ADX > 25 = trending; +DI > -DI = uptrend."""
    df = _prices_to_series(prices_json)
    raw = compute_adx_dmi_raw(df)
    result = {
        "ticker": ticker,
        "indicator": "adx_dmi",
        "values": {
            "adx": raw["adx"],
            "plus_di": raw["plus_di"],
            "minus_di": raw["minus_di"],
            "trend_strength": raw["trend_strength"],
            "di_crossover": raw["di_crossover"],
        },
        "interpretation": raw["interpretation"],
        "confidence": raw["confidence"],
    }
    return json.dumps(result)


@function_tool
async def compute_stochastic_oscillator(
    ctx: RunContextWrapper[AppContext],
    ticker: str,
    prices_json: str,
) -> str:
    """Compute Stochastic Oscillator (%K/%D). %K < 20 = oversold; bullish crossover below 20 = buy signal."""
    df = _prices_to_series(prices_json)
    raw = compute_stochastic_raw(df)
    result = {
        "ticker": ticker,
        "indicator": "stochastic_oscillator",
        "values": {
            "k": raw["k"],
            "d": raw["d"],
            "zone": raw["zone"],
            "crossover": raw["crossover"],
        },
        "interpretation": raw["interpretation"],
        "confidence": raw["confidence"],
    }
    return json.dumps(result)


@function_tool
async def compute_fibonacci_retracements(
    ctx: RunContextWrapper[AppContext],
    ticker: str,
    prices_json: str,
) -> str:
    """Compute Fibonacci retracement levels from swing high/low (23.6%, 38.2%, 50%, 61.8%, 78.6%). Price at 38.2% or 61.8% = potential reversal zone."""
    df = _prices_to_series(prices_json)
    raw = compute_fibonacci_raw(df)
    result = {
        "ticker": ticker,
        "indicator": "fibonacci_retracements",
        "values": {
            "swing_high": raw["swing_high"],
            "swing_low": raw["swing_low"],
            "levels": raw["levels"],
            "current_zone": raw["current_zone"],
            "fib_retrace_pct": raw["fib_retrace_pct"],
            "nearest_fib_above": raw["nearest_fib_above"],
            "nearest_fib_below": raw["nearest_fib_below"],
        },
        "interpretation": raw["interpretation"],
        "confidence": raw["confidence"],
    }
    return json.dumps(result)


@function_tool
async def compute_volume_profile(
    ctx: RunContextWrapper[AppContext],
    ticker: str,
    prices_json: str,
) -> str:
    """Compute Volume Profile (Volume-at-Price) with POC and HVN/LVN zones. HVN = support/resistance; POC = fair value."""
    df = _prices_to_series(prices_json)
    raw = compute_volume_profile_raw(df)
    result = {
        "ticker": ticker,
        "indicator": "volume_profile",
        "values": {
            "poc": raw["poc"],
            "value_area_high": raw["value_area_high"],
            "value_area_low": raw["value_area_low"],
            "hvn_zones": raw["hvn_zones"],
            "lvn_zones": raw["lvn_zones"],
            "price_vs_poc_pct": raw["price_vs_poc_pct"],
        },
        "interpretation": raw["interpretation"],
        "confidence": raw["confidence"],
    }
    return json.dumps(result)
