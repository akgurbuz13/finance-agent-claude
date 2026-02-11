"""Quantitative models — return/vol forecasts, regime detection, correlations, factors."""

from __future__ import annotations

import json

import numpy as np
from agents import RunContextWrapper, function_tool

from portfolio_advisor.agents.context import AppContext
from portfolio_advisor.tools.technical_indicators import _prices_to_series


# ── Pure computation functions (no ctx, return dicts) ────────────────────────


def compute_return_forecast_raw(df) -> dict:
    """Momentum + mean-reversion blend return forecast — pure function.

    Expects a DataFrame with a 'close' column (from _prices_to_series).
    Returns forecast dicts for 1w, 1m, 3m horizons.
    """
    close = df["close"]
    returns = close.pct_change().dropna()

    if len(returns) < 60:
        return {"error": "Insufficient data (need 60+ bars)"}

    # Momentum component: trailing returns
    mom_1m = float(close.iloc[-1] / close.iloc[-21] - 1) if len(close) > 21 else 0
    mom_3m = float(close.iloc[-1] / close.iloc[-63] - 1) if len(close) > 63 else 0

    # Mean-reversion component: distance from rolling mean
    sma60 = close.rolling(60).mean()
    mr_signal = float((close.iloc[-1] / sma60.iloc[-1]) - 1)

    # Blend: 60% momentum, 40% mean-reversion (inverted)
    daily_vol = float(returns.std())
    annualized_vol = daily_vol * np.sqrt(252)

    forecasts = {}
    for label, days, mom_w, mr_w in [
        ("1w", 5, 0.7, 0.3),
        ("1m", 21, 0.6, 0.4),
        ("3m", 63, 0.4, 0.6),
    ]:
        mom_component = mom_1m if days <= 21 else mom_3m
        mr_component = -mr_signal * 0.5
        expected = mom_component * mom_w + mr_component * mr_w
        expected_horizon = expected * (days / 21)
        ci_width = daily_vol * np.sqrt(days) * 1.96

        forecasts[label] = {
            "expected_return_pct": round(expected_horizon * 100, 2),
            "ci_low_pct": round((expected_horizon - ci_width) * 100, 2),
            "ci_high_pct": round((expected_horizon + ci_width) * 100, 2),
            "confidence": round(max(0.3, min(0.8, 0.6 - abs(mr_signal))), 2),
        }

    return {
        "annualized_vol": round(annualized_vol * 100, 2),
        "momentum_1m_pct": round(mom_1m * 100, 2),
        "momentum_3m_pct": round(mom_3m * 100, 2),
        "mean_reversion_signal": round(mr_signal * 100, 2),
        "forecasts": forecasts,
    }


def compute_vol_forecast_raw(df) -> dict:
    """EWMA volatility forecast with regime classification — pure function."""
    returns = df["close"].pct_change().dropna()

    if len(returns) < 30:
        return {"error": "Insufficient data"}

    # EWMA vol (lambda=0.94, standard RiskMetrics)
    ewma_var = returns.ewm(span=30, adjust=False).var()
    ewma_vol = float(np.sqrt(ewma_var.iloc[-1]))
    annualized_ewma = ewma_vol * np.sqrt(252)

    # 1-year realized vol for percentile
    hist_vol = returns.rolling(252).std() * np.sqrt(252)
    if len(hist_vol.dropna()) > 0:
        current_percentile = float(
            (hist_vol.dropna() < annualized_ewma).sum() / len(hist_vol.dropna()) * 100
        )
    else:
        current_percentile = 50.0

    # Regime
    if current_percentile > 80:
        regime = "high"
    elif current_percentile < 20:
        regime = "low"
    else:
        regime = "normal"

    return {
        "ewma_vol_daily": round(ewma_vol * 100, 4),
        "ewma_vol_annualized": round(annualized_ewma * 100, 2),
        "vol_percentile_1y": round(current_percentile, 1),
        "vol_regime": regime,
    }


def detect_regime_raw(df) -> dict:
    """Detect market regime via Hurst exponent + volatility clustering — pure function."""
    close = df["close"]
    returns = close.pct_change().dropna()

    if len(returns) < 100:
        return {"error": "Insufficient data (need 100+ bars)"}

    # Simplified Hurst exponent via R/S analysis
    n = len(returns)
    max_k = min(n // 2, 128)
    rs_list = []
    sizes = [s for s in [16, 32, 64, 128] if s <= max_k]

    for size in sizes:
        num_chunks = n // size
        rs_vals = []
        for i in range(num_chunks):
            chunk = returns.iloc[i * size : (i + 1) * size].values
            mean_chunk = chunk.mean()
            cumdev = np.cumsum(chunk - mean_chunk)
            r = cumdev.max() - cumdev.min()
            s = chunk.std()
            if s > 0:
                rs_vals.append(r / s)
        if rs_vals:
            rs_list.append((np.log(size), np.log(np.mean(rs_vals))))

    hurst = 0.5  # default
    if len(rs_list) >= 2:
        x = np.array([p[0] for p in rs_list])
        y = np.array([p[1] for p in rs_list])
        slope, _ = np.polyfit(x, y, 1)
        hurst = float(slope)

    # Volatility clustering: autocorrelation of absolute returns
    abs_ret = returns.abs()
    vol_autocorr = float(abs_ret.autocorr(lag=1)) if len(abs_ret) > 2 else 0.0

    # Regime classification
    if hurst > 0.6:
        regime = "trending"
    elif hurst < 0.4:
        regime = "mean_reverting"
    elif vol_autocorr > 0.3:
        regime = "volatile"
    else:
        regime = "neutral"

    return {
        "hurst_exponent": round(hurst, 3),
        "hurst_interpretation": (
            "trending" if hurst > 0.55 else "mean_reverting" if hurst < 0.45 else "random_walk"
        ),
        "vol_autocorrelation": round(vol_autocorr, 3),
        "regime": regime,
        "confidence": round(min(0.85, abs(hurst - 0.5) * 2 + 0.4), 2),
    }


def compute_factor_exposures_raw(
    returns: np.ndarray,
    spy_returns: np.ndarray,
) -> dict:
    """OLS regression vs SPY for beta/alpha — pure function.

    Expects aligned numpy arrays of returns (same length).
    """
    if len(returns) < 30:
        return {"error": "Insufficient overlapping data"}

    x_with_const = np.column_stack([np.ones(len(spy_returns)), spy_returns])
    try:
        coeffs = np.linalg.lstsq(x_with_const, returns, rcond=None)[0]
        alpha = float(coeffs[0])
        beta = float(coeffs[1])
    except np.linalg.LinAlgError:
        alpha, beta = 0.0, 1.0

    # R-squared
    y_pred = x_with_const @ coeffs
    ss_res = np.sum((returns - y_pred) ** 2)
    ss_tot = np.sum((returns - returns.mean()) ** 2)
    r_squared = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    return {
        "market_beta": round(beta, 3),
        "alpha_daily": round(alpha, 6),
        "alpha_annualized_pct": round(alpha * 252 * 100, 2),
        "r_squared": round(r_squared, 3),
        "interpretation": (
            "defensive" if beta < 0.8 else "market_neutral" if beta < 1.2 else "aggressive"
        ),
    }


# ── @function_tool wrappers ──────────────────────────────────────────────────


@function_tool
async def compute_return_forecast(
    ctx: RunContextWrapper[AppContext],
    ticker: str,
    prices_json: str,
) -> str:
    """Momentum + mean-reversion blend return forecast for 1w/1m/3m horizons."""
    df = _prices_to_series(prices_json)
    raw = compute_return_forecast_raw(df)
    if "error" in raw:
        return json.dumps({"ticker": ticker, "error": raw["error"]})
    raw["ticker"] = ticker
    raw["model"] = "momentum_mean_reversion_blend"
    return json.dumps(raw)


@function_tool
async def compute_vol_forecast(
    ctx: RunContextWrapper[AppContext],
    ticker: str,
    prices_json: str,
) -> str:
    """EWMA volatility forecast with regime classification."""
    df = _prices_to_series(prices_json)
    raw = compute_vol_forecast_raw(df)
    if "error" in raw:
        return json.dumps({"ticker": ticker, "error": raw["error"]})
    raw["ticker"] = ticker
    return json.dumps(raw)


@function_tool
async def detect_regime(
    ctx: RunContextWrapper[AppContext],
    ticker: str,
    prices_json: str,
) -> str:
    """Detect market regime via rolling Hurst exponent approximation + volatility clustering."""
    df = _prices_to_series(prices_json)
    raw = detect_regime_raw(df)
    if "error" in raw:
        return json.dumps({"ticker": ticker, "error": raw["error"]})
    raw["ticker"] = ticker
    return json.dumps(raw)


@function_tool
async def compute_correlation_matrix(
    ctx: RunContextWrapper[AppContext],
    tickers: str,
    period: str = "6mo",
) -> str:
    """Compute NxN correlation matrix with diversification score."""
    import yfinance as yf

    ticker_list = [t.strip() for t in tickers.split(",")]
    from portfolio_advisor.tools.market_data import CRYPTO_MAP

    equity_tickers = [t for t in ticker_list if t.upper() not in CRYPTO_MAP]

    if len(equity_tickers) < 2:
        return json.dumps({"error": "Need at least 2 equity tickers for correlation"})

    df = yf.download(equity_tickers, period=period, progress=False)
    if "Close" not in df.columns and len(equity_tickers) == 1:
        return json.dumps({"error": "Download failed"})

    if len(equity_tickers) == 1:
        close = df[["Close"]]
        close.columns = equity_tickers
    else:
        close = df["Close"][equity_tickers]

    returns = close.pct_change().dropna()
    corr = returns.corr()

    n = len(equity_tickers)
    if n > 1:
        upper_tri = corr.values[np.triu_indices(n, k=1)]
        avg_corr = float(np.mean(upper_tri))
        diversification_score = round(1 - avg_corr, 2)
    else:
        avg_corr = 1.0
        diversification_score = 0.0

    notable = []
    for i in range(n):
        for j in range(i + 1, n):
            c = float(corr.iloc[i, j])
            if abs(c) > 0.8:
                notable.append({
                    "pair": f"{equity_tickers[i]}/{equity_tickers[j]}",
                    "correlation": round(c, 3),
                    "note": "highly correlated" if c > 0 else "highly inversely correlated",
                })

    result = {
        "tickers": equity_tickers,
        "correlation_matrix": {
            t: {t2: round(float(corr.loc[t, t2]), 3) for t2 in equity_tickers}
            for t in equity_tickers
        },
        "avg_pairwise_correlation": round(avg_corr, 3),
        "diversification_score": diversification_score,
        "notable_pairs": notable,
    }
    return json.dumps(result)


@function_tool
async def compute_factor_exposures(
    ctx: RunContextWrapper[AppContext],
    ticker: str,
    prices_json: str,
) -> str:
    """Compute market beta and simple factor loadings via OLS regression vs SPY."""
    import yfinance as yf

    df = _prices_to_series(prices_json)
    returns = df["close"].pct_change().dropna()

    spy = yf.download("SPY", period="1y", progress=False)
    if spy.empty:
        return json.dumps({"ticker": ticker, "error": "Could not fetch SPY data"})

    spy_returns = spy["Close"].pct_change().dropna()

    # Align dates
    common = returns.index.intersection(spy_returns.index)
    if len(common) < 30:
        return json.dumps({"ticker": ticker, "error": "Insufficient overlapping data"})

    raw = compute_factor_exposures_raw(
        returns.loc[common].values,
        spy_returns.loc[common].values,
    )
    raw["ticker"] = ticker
    return json.dumps(raw)
