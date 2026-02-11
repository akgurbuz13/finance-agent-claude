"""Time series analysis tools — autocorrelation, stationarity, decomposition, cointegration."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from agents import RunContextWrapper, function_tool

from portfolio_advisor.agents.context import AppContext
from portfolio_advisor.tools.technical_indicators import _prices_to_series


@function_tool
async def compute_autocorrelation(
    ctx: RunContextWrapper[AppContext],
    ticker: str,
    prices_json: str,
    max_lag: int = 20,
) -> str:
    """Compute autocorrelation (ACF) and partial autocorrelation (PACF) of returns.

    Identifies mean-reversion timescales (negative ACF) and momentum timescales
    (positive ACF). Returns significance bounds at 95% confidence.
    """
    df = _prices_to_series(prices_json)
    returns = df["close"].pct_change().dropna()
    n = len(returns)

    if n < max_lag + 20:
        return json.dumps({"ticker": ticker, "error": f"Insufficient data: need {max_lag + 20}+ bars, got {n}"})

    # ACF
    mean_r = returns.mean()
    denom = ((returns - mean_r) ** 2).sum()
    acf_values = []
    for lag in range(1, max_lag + 1):
        num = ((returns.iloc[lag:].values - mean_r) * (returns.iloc[:-lag].values - mean_r)).sum()
        acf_values.append(round(float(num / denom), 4))

    # PACF via Yule-Walker (simplified Durbin-Levinson)
    pacf_values = []
    r = np.array(acf_values)
    phi = np.zeros((max_lag, max_lag))
    phi[0, 0] = r[0]
    pacf_values.append(round(float(r[0]), 4))
    for k in range(1, max_lag):
        num = r[k] - sum(phi[k - 1, j] * r[k - 1 - j] for j in range(k))
        den = 1.0 - sum(phi[k - 1, j] * r[j] for j in range(k))
        if abs(den) < 1e-10:
            pacf_values.append(0.0)
            continue
        phi[k, k] = num / den
        for j in range(k):
            phi[k, j] = phi[k - 1, j] - phi[k, k] * phi[k - 1, k - 1 - j]
        pacf_values.append(round(float(phi[k, k]), 4))

    # 95% significance bound (Bartlett)
    sig_bound = round(1.96 / np.sqrt(n), 4)

    # Interpret
    significant_acf_lags = [i + 1 for i, v in enumerate(acf_values) if abs(v) > sig_bound]
    mean_reverting_lags = [i + 1 for i, v in enumerate(acf_values) if v < -sig_bound]
    momentum_lags = [i + 1 for i, v in enumerate(acf_values) if v > sig_bound]

    if mean_reverting_lags:
        behavior = "mean_reverting"
        interpretation = f"Significant negative autocorrelation at lags {mean_reverting_lags}: returns tend to reverse."
    elif momentum_lags:
        behavior = "momentum"
        interpretation = f"Significant positive autocorrelation at lags {momentum_lags}: returns tend to persist."
    else:
        behavior = "random_walk"
        interpretation = "No significant autocorrelation detected — returns behave like a random walk."

    return json.dumps({
        "ticker": ticker,
        "acf": acf_values,
        "pacf": pacf_values,
        "significance_bound_95": sig_bound,
        "significant_lags": significant_acf_lags,
        "mean_reverting_lags": mean_reverting_lags,
        "momentum_lags": momentum_lags,
        "behavior": behavior,
        "interpretation": interpretation,
        "observations": n,
    })


@function_tool
async def compute_stationarity_test(
    ctx: RunContextWrapper[AppContext],
    ticker: str,
    prices_json: str,
) -> str:
    """Run Augmented Dickey-Fuller test on returns and log-prices.

    Tests whether the time series has a unit root (non-stationary).
    Returns: log-prices (expected non-stationary) and returns (expected stationary).
    """
    df = _prices_to_series(prices_json)
    close = df["close"].dropna()
    returns = close.pct_change().dropna()
    log_prices = np.log(close)

    if len(returns) < 50:
        return json.dumps({"ticker": ticker, "error": "Need 50+ observations for ADF test"})

    def _adf_simple(series: pd.Series, max_lags: int = 10) -> dict:
        """Simplified ADF test using OLS regression."""
        y = series.values
        n = len(y)
        dy = np.diff(y)

        # Find optimal lag via AIC
        best_aic = np.inf
        best_lag = 1
        for lag in range(1, min(max_lags + 1, n // 4)):
            # Build regression matrix: dy_t = alpha + gamma*y_{t-1} + sum(beta_i * dy_{t-i}) + eps
            nobs = n - 1 - lag
            if nobs < 10:
                continue
            Y = dy[lag:]
            X = np.column_stack([
                np.ones(nobs),
                y[lag:-1],  # lagged level
                *[dy[lag - i - 1:-i - 1] for i in range(lag)]
            ])
            try:
                coeffs = np.linalg.lstsq(X, Y, rcond=None)[0]
                resid = Y - X @ coeffs
                sse = np.sum(resid ** 2)
                aic = nobs * np.log(sse / nobs) + 2 * (lag + 2)
                if aic < best_aic:
                    best_aic = aic
                    best_lag = lag
            except np.linalg.LinAlgError:
                continue

        # Run ADF with best lag
        nobs = n - 1 - best_lag
        Y = dy[best_lag:]
        X = np.column_stack([
            np.ones(nobs),
            y[best_lag:-1],
            *[dy[best_lag - i - 1:-i - 1] for i in range(best_lag)]
        ])
        try:
            coeffs, residuals, _, _ = np.linalg.lstsq(X, Y, rcond=None)
            gamma = coeffs[1]
            resid = Y - X @ coeffs
            se = np.sqrt(np.sum(resid ** 2) / (nobs - len(coeffs)))
            x_inv = np.linalg.pinv(X.T @ X)
            se_gamma = se * np.sqrt(x_inv[1, 1])
            t_stat = gamma / se_gamma if se_gamma > 0 else 0.0
        except np.linalg.LinAlgError:
            t_stat = 0.0

        # Critical values (MacKinnon approximate for constant, no trend)
        critical = {"1%": -3.43, "5%": -2.86, "10%": -2.57}
        is_stationary = t_stat < critical["5%"]

        return {
            "adf_statistic": round(float(t_stat), 4),
            "critical_values": critical,
            "is_stationary_5pct": is_stationary,
            "optimal_lag": best_lag,
        }

    adf_prices = _adf_simple(log_prices)
    adf_returns = _adf_simple(returns)

    if adf_returns["is_stationary_5pct"] and not adf_prices["is_stationary_5pct"]:
        interpretation = "Log-prices are non-stationary (unit root) while returns are stationary — consistent with I(1) integrated process. Standard for asset prices."
    elif adf_prices["is_stationary_5pct"]:
        interpretation = "Log-prices appear stationary — unusual. May indicate mean-reverting behavior or insufficient data."
    elif not adf_returns["is_stationary_5pct"]:
        interpretation = "Returns appear non-stationary — unusual. Check for structural breaks, regime changes, or data quality issues."
    else:
        interpretation = "Standard behavior: non-stationary prices, stationary returns."

    return json.dumps({
        "ticker": ticker,
        "log_prices_adf": adf_prices,
        "returns_adf": adf_returns,
        "interpretation": interpretation,
        "observations": len(returns),
    })


@function_tool
async def compute_seasonal_decomposition(
    ctx: RunContextWrapper[AppContext],
    ticker: str,
    prices_json: str,
    period: int = 21,
) -> str:
    """Decompose price series into trend, seasonal, and residual components.

    Uses moving average decomposition (additive model on log-returns).
    Default period=21 (monthly cycle). Use period=5 for weekly patterns.
    """
    df = _prices_to_series(prices_json)
    close = df["close"].dropna()

    if len(close) < period * 3:
        return json.dumps({"ticker": ticker, "error": f"Need {period * 3}+ bars for decomposition"})

    log_returns = np.log(close / close.shift(1)).dropna()

    # Trend: centered moving average
    trend = log_returns.rolling(window=period, center=True).mean()

    # Detrended
    detrended = log_returns - trend

    # Seasonal: average of detrended by position within period
    seasonal = detrended.copy()
    for i in range(period):
        mask = np.arange(len(detrended)) % period == i
        seasonal_mean = detrended.iloc[mask].mean()
        seasonal.iloc[mask] = seasonal_mean if not np.isnan(seasonal_mean) else 0.0

    # Residual
    residual = log_returns - trend - seasonal

    # Summary stats
    trend_clean = trend.dropna()
    residual_clean = residual.dropna()

    trend_strength = 1.0 - (residual_clean.var() / (trend_clean + residual_clean).dropna().var()) if len(trend_clean) > 0 else 0.0
    trend_strength = max(0.0, min(1.0, float(trend_strength)))

    seasonal_strength_val = float(seasonal.std() / log_returns.std()) if log_returns.std() > 0 else 0.0

    # Recent trend direction
    if len(trend_clean) >= 5:
        recent_trend = float(trend_clean.iloc[-5:].mean())
        trend_direction = "upward" if recent_trend > 0.001 else "downward" if recent_trend < -0.001 else "flat"
    else:
        trend_direction = "insufficient_data"
        recent_trend = 0.0

    return json.dumps({
        "ticker": ticker,
        "period": period,
        "trend_strength": round(trend_strength, 3),
        "seasonal_strength": round(seasonal_strength_val, 3),
        "residual_volatility": round(float(residual_clean.std()) * 100, 4) if len(residual_clean) > 0 else None,
        "trend_direction": trend_direction,
        "recent_trend_daily_pct": round(recent_trend * 100, 4),
        "interpretation": (
            f"Trend explains {trend_strength*100:.0f}% of variance ({trend_direction}). "
            f"Seasonal patterns are {'significant' if seasonal_strength_val > 0.1 else 'weak'} "
            f"(strength={seasonal_strength_val:.2f}). "
            f"Residual vol = {float(residual_clean.std())*100:.2f}% daily."
        ) if len(residual_clean) > 0 else "Insufficient data for interpretation.",
        "observations": len(log_returns),
    })


@function_tool
async def compute_cointegration_test(
    ctx: RunContextWrapper[AppContext],
    tickers: str,
    period: str = "1y",
) -> str:
    """Engle-Granger cointegration test between pairs of assets.

    Identifies long-run equilibrium relationships useful for pairs trading
    or understanding structural asset relationships.
    """
    import yfinance as yf

    ticker_list = [t.strip() for t in tickers.split(",")]
    if len(ticker_list) < 2:
        return json.dumps({"error": "Need at least 2 tickers for cointegration test"})

    df = yf.download(ticker_list, period=period, progress=False)
    if df.empty:
        return json.dumps({"error": "No data returned"})

    if len(ticker_list) == 1:
        return json.dumps({"error": "Need at least 2 tickers"})

    close = df["Close"][ticker_list].dropna()
    if len(close) < 60:
        return json.dumps({"error": f"Insufficient overlapping data: {len(close)} rows"})

    pairs = []
    for i in range(len(ticker_list)):
        for j in range(i + 1, len(ticker_list)):
            t1, t2 = ticker_list[i], ticker_list[j]
            y = np.log(close[t1].values)
            x = np.log(close[t2].values)

            # Step 1: OLS regression y = alpha + beta*x
            x_with_const = np.column_stack([np.ones(len(x)), x])
            coeffs = np.linalg.lstsq(x_with_const, y, rcond=None)[0]
            alpha, beta = coeffs[0], coeffs[1]

            # Step 2: ADF test on residuals (spread)
            spread = y - alpha - beta * x
            dy = np.diff(spread)
            Y = dy[1:]
            X = np.column_stack([spread[1:-1], dy[:-1]])
            try:
                c = np.linalg.lstsq(X, Y, rcond=None)[0]
                resid = Y - X @ c
                se = np.sqrt(np.sum(resid ** 2) / (len(Y) - 2))
                x_inv = np.linalg.pinv(X.T @ X)
                se_gamma = se * np.sqrt(x_inv[0, 0])
                adf_stat = float(c[0] / se_gamma) if se_gamma > 0 else 0.0
            except np.linalg.LinAlgError:
                adf_stat = 0.0

            # Engle-Granger critical values (more conservative than standard ADF)
            eg_critical = {"1%": -3.90, "5%": -3.34, "10%": -3.04}
            is_cointegrated = adf_stat < eg_critical["5%"]

            # Half-life of mean reversion
            if c[0] < 0:
                half_life = round(-np.log(2) / c[0], 1)
            else:
                half_life = None

            # Current spread z-score
            spread_mean = spread.mean()
            spread_std = spread.std()
            current_z = float((spread[-1] - spread_mean) / spread_std) if spread_std > 0 else 0.0

            pairs.append({
                "pair": f"{t1}/{t2}",
                "hedge_ratio": round(float(beta), 4),
                "adf_statistic": round(adf_stat, 4),
                "critical_values": eg_critical,
                "is_cointegrated": is_cointegrated,
                "half_life_days": half_life,
                "current_spread_zscore": round(current_z, 2),
                "signal": (
                    "long_spread" if is_cointegrated and current_z < -2 else
                    "short_spread" if is_cointegrated and current_z > 2 else
                    "no_signal"
                ),
            })

    return json.dumps({
        "tickers": ticker_list,
        "pairs": pairs,
        "cointegrated_pairs": [p["pair"] for p in pairs if p["is_cointegrated"]],
        "interpretation": (
            f"Found {sum(1 for p in pairs if p['is_cointegrated'])} cointegrated pair(s) "
            f"out of {len(pairs)} tested."
        ),
        "observations": len(close),
    })


@function_tool
async def compute_rolling_statistics(
    ctx: RunContextWrapper[AppContext],
    ticker: str,
    prices_json: str,
    windows: str = "21,63,126",
) -> str:
    """Compute rolling mean, std, skewness, and kurtosis across multiple windows.

    Windows are in trading days (21=1mo, 63=3mo, 126=6mo).
    Identifies regime changes and structural shifts in return distribution.
    """
    df = _prices_to_series(prices_json)
    returns = df["close"].pct_change().dropna()
    window_list = [int(w.strip()) for w in windows.split(",")]

    if len(returns) < max(window_list):
        return json.dumps({"ticker": ticker, "error": f"Need {max(window_list)}+ observations"})

    stats_by_window = {}
    for w in window_list:
        rolling_mean = returns.rolling(w).mean()
        rolling_std = returns.rolling(w).std()
        rolling_skew = returns.rolling(w).skew()
        rolling_kurt = returns.rolling(w).apply(lambda x: x.kurtosis(), raw=False)

        current = {
            "mean_annualized_pct": round(float(rolling_mean.iloc[-1]) * 252 * 100, 2),
            "vol_annualized_pct": round(float(rolling_std.iloc[-1]) * np.sqrt(252) * 100, 2),
            "skewness": round(float(rolling_skew.iloc[-1]), 3) if not np.isnan(rolling_skew.iloc[-1]) else 0.0,
            "excess_kurtosis": round(float(rolling_kurt.iloc[-1]), 3) if not np.isnan(rolling_kurt.iloc[-1]) else 0.0,
        }

        # Compare current to historical range
        vol_pctile = float((rolling_std.dropna() < rolling_std.iloc[-1]).mean() * 100)
        mean_pctile = float((rolling_mean.dropna() < rolling_mean.iloc[-1]).mean() * 100)

        current["vol_percentile"] = round(vol_pctile, 1)
        current["return_percentile"] = round(mean_pctile, 1)

        stats_by_window[f"{w}d"] = current

    # Detect regime changes: large difference between short and long window vol
    short_w = min(window_list)
    long_w = max(window_list)
    short_vol = stats_by_window[f"{short_w}d"]["vol_annualized_pct"]
    long_vol = stats_by_window[f"{long_w}d"]["vol_annualized_pct"]
    vol_ratio = short_vol / long_vol if long_vol > 0 else 1.0

    regime_shift = "none"
    if vol_ratio > 1.5:
        regime_shift = "volatility_expanding"
    elif vol_ratio < 0.6:
        regime_shift = "volatility_contracting"

    return json.dumps({
        "ticker": ticker,
        "windows": stats_by_window,
        "vol_ratio_short_long": round(vol_ratio, 2),
        "regime_shift": regime_shift,
        "interpretation": (
            f"Short-term vol ({short_vol:.1f}%) vs long-term ({long_vol:.1f}%): "
            f"{'expanding — risk increasing' if regime_shift == 'volatility_expanding' else 'contracting — risk decreasing' if regime_shift == 'volatility_contracting' else 'stable'}. "
            f"Current skew = {stats_by_window[f'{short_w}d']['skewness']:.2f} "
            f"({'left-tail risk' if stats_by_window[f'{short_w}d']['skewness'] < -0.5 else 'right-skewed' if stats_by_window[f'{short_w}d']['skewness'] > 0.5 else 'symmetric'})."
        ),
        "observations": len(returns),
    })
