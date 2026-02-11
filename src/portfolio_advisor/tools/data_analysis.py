"""Data analysis tools — distribution analysis, statistical tests, performance metrics, outliers."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from agents import RunContextWrapper, function_tool

from portfolio_advisor.agents.context import AppContext
from portfolio_advisor.tools.technical_indicators import _prices_to_series


@function_tool
async def compute_distribution_analysis(
    ctx: RunContextWrapper[AppContext],
    ticker: str,
    prices_json: str,
) -> str:
    """Analyze the return distribution: skewness, kurtosis, Jarque-Bera normality test, tail analysis.

    Essential for understanding whether standard risk models (which assume normality) are appropriate.
    """
    df = _prices_to_series(prices_json)
    returns = df["close"].pct_change().dropna()
    n = len(returns)

    if n < 30:
        return json.dumps({"ticker": ticker, "error": "Need 30+ observations"})

    mean_r = float(returns.mean())
    std_r = float(returns.std())
    skew = float(returns.skew())
    kurt = float(returns.kurtosis())  # Excess kurtosis (normal = 0)

    # Jarque-Bera test: JB = (n/6) * (S^2 + (K^2)/4)
    jb_stat = (n / 6) * (skew ** 2 + (kurt ** 2) / 4)
    # Chi-squared critical value at 5% with 2 df = 5.991
    is_normal = jb_stat < 5.991

    # Tail analysis
    left_5 = float(np.percentile(returns, 5))
    left_1 = float(np.percentile(returns, 1))
    right_95 = float(np.percentile(returns, 95))
    right_99 = float(np.percentile(returns, 99))

    # Compare tails to normal
    normal_left_1 = mean_r + std_r * (-2.326)
    tail_ratio_left = abs(left_1 / normal_left_1) if normal_left_1 != 0 else 1.0

    # Days beyond 2-sigma
    extreme_days = int((returns.abs() > 2 * std_r).sum())
    extreme_pct = round(extreme_days / n * 100, 1)
    # Normal expectation: ~4.6%
    normal_extreme_pct = 4.6

    return json.dumps({
        "ticker": ticker,
        "observations": n,
        "mean_daily_pct": round(mean_r * 100, 4),
        "std_daily_pct": round(std_r * 100, 4),
        "annualized_return_pct": round(mean_r * 252 * 100, 2),
        "annualized_vol_pct": round(std_r * np.sqrt(252) * 100, 2),
        "skewness": round(skew, 3),
        "excess_kurtosis": round(kurt, 3),
        "jarque_bera_statistic": round(jb_stat, 2),
        "is_normal_5pct": is_normal,
        "tails": {
            "left_5_pct": round(left_5 * 100, 3),
            "left_1_pct": round(left_1 * 100, 3),
            "right_95_pct": round(right_95 * 100, 3),
            "right_99_pct": round(right_99 * 100, 3),
            "tail_ratio_left": round(tail_ratio_left, 2),
        },
        "extreme_days": {
            "beyond_2sigma_count": extreme_days,
            "beyond_2sigma_pct": extreme_pct,
            "normal_expected_pct": normal_extreme_pct,
            "fat_tails": extreme_pct > normal_extreme_pct * 1.5,
        },
        "interpretation": (
            f"{'Non-normal' if not is_normal else 'Approximately normal'} distribution "
            f"(JB={jb_stat:.1f}). "
            f"{'Negative skew — larger left-tail risk. ' if skew < -0.3 else 'Positive skew — right-tail upside. ' if skew > 0.3 else ''}"
            f"{'Leptokurtic (fat tails) — extreme events more likely than normal model suggests. ' if kurt > 1 else ''}"
            f"{extreme_pct:.1f}% of days beyond 2-sigma (normal expects {normal_extreme_pct}%)."
        ),
    })


@function_tool
async def compute_drawdown_analysis(
    ctx: RunContextWrapper[AppContext],
    ticker: str,
    prices_json: str,
) -> str:
    """Full drawdown analysis: all drawdown episodes with depth, duration, recovery time.

    Identifies where the asset is in its drawdown cycle — crucial for timing and risk assessment.
    """
    df = _prices_to_series(prices_json)
    close = df["close"].dropna()

    if len(close) < 20:
        return json.dumps({"ticker": ticker, "error": "Need 20+ observations"})

    # Compute drawdown series
    running_max = close.cummax()
    drawdown = (close - running_max) / running_max

    # Current drawdown state
    current_dd = float(drawdown.iloc[-1])
    max_dd = float(drawdown.min())

    # Find all drawdown episodes (trough-to-recovery)
    episodes = []
    in_dd = False
    peak_date = None
    trough_date = None
    trough_val = 0.0

    for i in range(len(drawdown)):
        dd_val = float(drawdown.iloc[i])
        date_str = drawdown.index[i].strftime("%Y-%m-%d")

        if dd_val < -0.01 and not in_dd:
            # Start of drawdown
            in_dd = True
            peak_date = drawdown.index[i - 1].strftime("%Y-%m-%d") if i > 0 else date_str
            trough_date = date_str
            trough_val = dd_val
        elif in_dd and dd_val < trough_val:
            # Deeper trough
            trough_date = date_str
            trough_val = dd_val
        elif in_dd and dd_val >= -0.005:
            # Recovery
            recovery_date = date_str
            peak_dt = pd.Timestamp(peak_date)
            trough_dt = pd.Timestamp(trough_date)
            recovery_dt = pd.Timestamp(recovery_date)
            episodes.append({
                "peak_date": peak_date,
                "trough_date": trough_date,
                "recovery_date": recovery_date,
                "depth_pct": round(trough_val * 100, 2),
                "days_to_trough": (trough_dt - peak_dt).days,
                "days_to_recovery": (recovery_dt - trough_dt).days,
                "total_days": (recovery_dt - peak_dt).days,
            })
            in_dd = False

    # If still in drawdown, add open episode
    if in_dd:
        episodes.append({
            "peak_date": peak_date,
            "trough_date": trough_date,
            "recovery_date": None,
            "depth_pct": round(trough_val * 100, 2),
            "days_to_trough": (pd.Timestamp(trough_date) - pd.Timestamp(peak_date)).days,
            "days_to_recovery": None,
            "total_days": None,
            "status": "ongoing",
        })

    # Sort by depth
    episodes.sort(key=lambda x: x["depth_pct"])

    # Average recovery time for completed drawdowns
    completed = [e for e in episodes if e.get("days_to_recovery") is not None]
    avg_recovery = round(np.mean([e["days_to_recovery"] for e in completed]), 0) if completed else None

    return json.dumps({
        "ticker": ticker,
        "current_drawdown_pct": round(current_dd * 100, 2),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "total_episodes": len(episodes),
        "top_5_drawdowns": episodes[:5],
        "avg_recovery_days": avg_recovery,
        "currently_in_drawdown": current_dd < -0.01,
        "interpretation": (
            f"Currently {'in drawdown of {:.1f}%'.format(current_dd * 100) if current_dd < -0.01 else 'near highs'}. "
            f"Worst drawdown: {max_dd*100:.1f}%. "
            f"{len(episodes)} drawdown episodes detected. "
            f"{'Avg recovery: ' + str(int(avg_recovery)) + ' days.' if avg_recovery else ''}"
        ),
        "observations": len(close),
    })


@function_tool
async def compute_performance_metrics(
    ctx: RunContextWrapper[AppContext],
    ticker: str,
    prices_json: str,
    benchmark: str = "SPY",
) -> str:
    """Compute comprehensive performance metrics: Sharpe, Sortino, Calmar, Information Ratio, etc.

    Compares against a benchmark (default SPY). Essential for evaluating risk-adjusted returns.
    """
    import yfinance as yf

    df = _prices_to_series(prices_json)
    returns = df["close"].pct_change().dropna()
    n = len(returns)

    if n < 30:
        return json.dumps({"ticker": ticker, "error": "Need 30+ observations"})

    mean_r = float(returns.mean())
    std_r = float(returns.std())
    # Annualized metrics
    ann_return = mean_r * 252
    ann_vol = std_r * np.sqrt(252)

    # Sharpe Ratio
    sharpe = (ann_return - 0.05) / ann_vol if ann_vol > 0 else 0.0

    # Sortino Ratio (downside deviation)
    downside = returns[returns < 0]
    downside_std = float(downside.std()) * np.sqrt(252) if len(downside) > 0 else ann_vol
    sortino = (ann_return - 0.05) / downside_std if downside_std > 0 else 0.0

    # Max drawdown
    cum_returns = (1 + returns).cumprod()
    max_dd = float(((cum_returns / cum_returns.cummax()) - 1).min())

    # Calmar Ratio
    calmar = ann_return / abs(max_dd) if max_dd < 0 else 0.0

    # Win rate
    win_rate = float((returns > 0).sum() / n * 100)

    # Profit factor
    gross_profit = float(returns[returns > 0].sum())
    gross_loss = float(abs(returns[returns < 0].sum()))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    result = {
        "ticker": ticker,
        "observations": n,
        "annualized_return_pct": round(ann_return * 100, 2),
        "annualized_vol_pct": round(ann_vol * 100, 2),
        "sharpe_ratio": round(sharpe, 3),
        "sortino_ratio": round(sortino, 3),
        "calmar_ratio": round(calmar, 3),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "win_rate_pct": round(win_rate, 1),
        "profit_factor": round(profit_factor, 2),
        "best_day_pct": round(float(returns.max()) * 100, 2),
        "worst_day_pct": round(float(returns.min()) * 100, 2),
    }

    # Benchmark comparison
    try:
        spy = yf.download(benchmark, period="1y", progress=False)
        if not spy.empty:
            spy_returns = spy["Close"].pct_change().dropna()
            common = returns.index.intersection(spy_returns.index)
            if len(common) > 20:
                active_returns = returns.loc[common] - spy_returns.loc[common]
                tracking_error = float(active_returns.std()) * np.sqrt(252)
                info_ratio = float(active_returns.mean()) * 252 / tracking_error if tracking_error > 0 else 0.0
                result["vs_benchmark"] = {
                    "benchmark": benchmark,
                    "information_ratio": round(info_ratio, 3),
                    "tracking_error_pct": round(tracking_error * 100, 2),
                    "excess_return_pct": round(float(active_returns.mean()) * 252 * 100, 2),
                }
    except Exception:
        pass

    result["interpretation"] = (
        f"Sharpe={sharpe:.2f} "
        f"({'excellent' if sharpe > 1.5 else 'good' if sharpe > 1 else 'acceptable' if sharpe > 0.5 else 'poor'}), "
        f"Sortino={sortino:.2f}, Calmar={calmar:.2f}. "
        f"Win rate {win_rate:.0f}%, profit factor {profit_factor:.1f}x. "
        f"Max DD {max_dd*100:.1f}%."
    )

    return json.dumps(result)


@function_tool
async def compute_outlier_detection(
    ctx: RunContextWrapper[AppContext],
    ticker: str,
    prices_json: str,
) -> str:
    """Detect outlier days using Z-score and IQR methods.

    Identifies anomalous return days that may indicate news events, earnings, or data errors.
    """
    df = _prices_to_series(prices_json)
    returns = df["close"].pct_change().dropna()
    n = len(returns)

    if n < 30:
        return json.dumps({"ticker": ticker, "error": "Need 30+ observations"})

    mean_r = float(returns.mean())
    std_r = float(returns.std())

    # Z-score method (|z| > 3)
    z_scores = (returns - mean_r) / std_r
    z_outliers = returns[z_scores.abs() > 3]

    # IQR method
    q1 = returns.quantile(0.25)
    q3 = returns.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    iqr_outliers = returns[(returns < lower_bound) | (returns > upper_bound)]

    # Recent outliers (last 30 days)
    recent = returns.iloc[-30:]
    recent_z = (recent - mean_r) / std_r
    recent_outliers = recent[recent_z.abs() > 2]

    z_outlier_list = [
        {"date": idx.strftime("%Y-%m-%d"), "return_pct": round(float(val) * 100, 2), "z_score": round(float(z_scores.loc[idx]), 2)}
        for idx, val in z_outliers.items()
    ]

    # Sort by absolute z-score
    z_outlier_list.sort(key=lambda x: abs(x["z_score"]), reverse=True)

    recent_outlier_list = [
        {"date": idx.strftime("%Y-%m-%d"), "return_pct": round(float(val) * 100, 2), "z_score": round(float(recent_z.loc[idx]), 2)}
        for idx, val in recent_outliers.items()
    ]

    return json.dumps({
        "ticker": ticker,
        "observations": n,
        "z_score_outliers_3sigma": {
            "count": len(z_outlier_list),
            "pct_of_total": round(len(z_outlier_list) / n * 100, 2),
            "expected_normal_pct": 0.27,
            "top_outliers": z_outlier_list[:10],
        },
        "iqr_outliers": {
            "count": len(iqr_outliers),
            "lower_bound_pct": round(float(lower_bound) * 100, 3),
            "upper_bound_pct": round(float(upper_bound) * 100, 3),
        },
        "recent_30d_outliers_2sigma": recent_outlier_list,
        "interpretation": (
            f"{len(z_outlier_list)} outlier days detected ({len(z_outlier_list)/n*100:.1f}% vs 0.27% normal expectation). "
            f"{'Fat tails present — standard risk models may underestimate tail risk. ' if len(z_outlier_list)/n*100 > 0.5 else ''}"
            f"{len(recent_outlier_list)} outlier(s) in last 30 days — "
            f"{'elevated recent volatility.' if recent_outlier_list else 'calm recent period.'}"
        ),
    })


@function_tool
async def compute_cross_asset_analysis(
    ctx: RunContextWrapper[AppContext],
    tickers: str,
    period: str = "6mo",
) -> str:
    """Cross-asset analysis: rolling correlations, lead-lag, relative strength.

    Identifies which assets are leading, lagging, and how relationships change over time.
    """
    import yfinance as yf

    ticker_list = [t.strip() for t in tickers.split(",")]
    if len(ticker_list) < 2:
        return json.dumps({"error": "Need 2+ tickers"})

    from portfolio_advisor.tools.market_data import CRYPTO_MAP
    equity_tickers = [t for t in ticker_list if t not in CRYPTO_MAP]

    if len(equity_tickers) < 2:
        return json.dumps({"error": "Need 2+ equity tickers for cross-asset analysis"})

    df = yf.download(equity_tickers, period=period, progress=False)
    if df.empty:
        return json.dumps({"error": "No data returned"})

    close = df["Close"][equity_tickers].dropna()
    returns = close.pct_change().dropna()
    n = len(returns)

    if n < 60:
        return json.dumps({"error": f"Insufficient data: {n} rows"})

    # Rolling 30-day correlations (current vs 60-day ago)
    rolling_corr = {}
    for i, t1 in enumerate(equity_tickers):
        for j, t2 in enumerate(equity_tickers):
            if j <= i:
                continue
            rc = returns[t1].rolling(30).corr(returns[t2])
            current_corr = float(rc.iloc[-1]) if not np.isnan(rc.iloc[-1]) else 0.0
            prev_corr = float(rc.iloc[-60]) if len(rc) > 60 and not np.isnan(rc.iloc[-60]) else current_corr
            rolling_corr[f"{t1}/{t2}"] = {
                "current_30d_corr": round(current_corr, 3),
                "prev_60d_ago_corr": round(prev_corr, 3),
                "change": round(current_corr - prev_corr, 3),
                "trend": "increasing" if current_corr > prev_corr + 0.1 else "decreasing" if current_corr < prev_corr - 0.1 else "stable",
            }

    # Lead-lag analysis (1-day cross-correlation)
    lead_lag = {}
    for i, t1 in enumerate(equity_tickers):
        for j, t2 in enumerate(equity_tickers):
            if j <= i:
                continue
            # Does t1 lead t2? (corr of t1 return today with t2 return tomorrow)
            corr_lead = float(returns[t1].iloc[:-1].reset_index(drop=True).corr(
                returns[t2].iloc[1:].reset_index(drop=True)
            ))
            corr_lag = float(returns[t2].iloc[:-1].reset_index(drop=True).corr(
                returns[t1].iloc[1:].reset_index(drop=True)
            ))

            if abs(corr_lead) > 0.1 or abs(corr_lag) > 0.1:
                if abs(corr_lead) > abs(corr_lag):
                    lead_lag[f"{t1}/{t2}"] = {"leader": t1, "follower": t2, "lead_corr": round(corr_lead, 3)}
                else:
                    lead_lag[f"{t1}/{t2}"] = {"leader": t2, "follower": t1, "lead_corr": round(corr_lag, 3)}

    # Relative strength (return over period)
    total_returns = {}
    for t in equity_tickers:
        total_ret = float(close[t].iloc[-1] / close[t].iloc[0] - 1)
        total_returns[t] = round(total_ret * 100, 2)

    # Rank by return
    ranked = sorted(total_returns.items(), key=lambda x: x[1], reverse=True)

    return json.dumps({
        "tickers": equity_tickers,
        "observations": n,
        "rolling_correlations": rolling_corr,
        "lead_lag_relationships": lead_lag,
        "relative_strength": {
            "total_returns_pct": total_returns,
            "ranking": [{"ticker": t, "return_pct": r, "rank": i + 1} for i, (t, r) in enumerate(ranked)],
        },
        "interpretation": (
            f"Relative strength leaders: {ranked[0][0]} ({ranked[0][1]:+.1f}%) > {ranked[1][0]} ({ranked[1][1]:+.1f}%). "
            f"{'Notable correlation shifts: ' + ', '.join(k + ' ' + v['trend'] for k, v in rolling_corr.items() if v['trend'] != 'stable') if any(v['trend'] != 'stable' for v in rolling_corr.values()) else 'Correlations stable.'}"
        ),
    })
