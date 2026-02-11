"""Portfolio risk metrics — VaR, ES, drawdown, beta/duration."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from agents import RunContextWrapper, function_tool

from portfolio_advisor.agents.context import AppContext
from portfolio_advisor.tools.technical_indicators import _prices_to_series


def _build_portfolio_returns(
    weights: dict[str, float], prices_dict: dict[str, list[dict]]
) -> pd.Series:
    """Build weighted portfolio return series from per-ticker prices."""
    all_returns = {}
    for ticker, bars in prices_dict.items():
        df = _prices_to_series(json.dumps(bars))
        all_returns[ticker] = df["close"].pct_change()

    ret_df = pd.DataFrame(all_returns).dropna()
    portfolio_returns = pd.Series(0.0, index=ret_df.index)
    for ticker, weight in weights.items():
        if ticker in ret_df.columns:
            portfolio_returns += ret_df[ticker] * weight
    return portfolio_returns.dropna()


@function_tool
async def compute_var(
    ctx: RunContextWrapper[AppContext],
    portfolio_weights_json: str,
    prices_json: str,
    confidence: float = 0.95,
) -> str:
    """Historical simulation Value-at-Risk. weights_json: {ticker: weight}, prices_json: {ticker: [bars]}."""
    weights = json.loads(portfolio_weights_json)
    prices_dict = json.loads(prices_json)

    port_returns = _build_portfolio_returns(weights, prices_dict)
    if len(port_returns) < 30:
        return json.dumps({"error": "Insufficient data for VaR calculation"})

    var_level = float(np.percentile(port_returns, (1 - confidence) * 100))

    result = {
        "var_confidence": confidence,
        "var_1day_pct": round(var_level * 100, 3),
        "var_1week_pct": round(var_level * np.sqrt(5) * 100, 3),
        "interpretation": (
            f"With {confidence*100:.0f}% confidence, daily loss will not exceed "
            f"{abs(var_level)*100:.2f}%"
        ),
        "observations": len(port_returns),
    }
    return json.dumps(result)


@function_tool
async def compute_expected_shortfall(
    ctx: RunContextWrapper[AppContext],
    portfolio_weights_json: str,
    prices_json: str,
    confidence: float = 0.95,
) -> str:
    """Expected Shortfall (CVaR) — average loss beyond VaR threshold."""
    weights = json.loads(portfolio_weights_json)
    prices_dict = json.loads(prices_json)

    port_returns = _build_portfolio_returns(weights, prices_dict)
    if len(port_returns) < 30:
        return json.dumps({"error": "Insufficient data"})

    var_threshold = np.percentile(port_returns, (1 - confidence) * 100)
    tail_losses = port_returns[port_returns <= var_threshold]
    es = float(tail_losses.mean()) if len(tail_losses) > 0 else float(var_threshold)

    result = {
        "es_confidence": confidence,
        "expected_shortfall_1day_pct": round(es * 100, 3),
        "var_1day_pct": round(float(var_threshold) * 100, 3),
        "tail_observations": len(tail_losses),
        "interpretation": (
            f"Average loss in the worst {(1-confidence)*100:.0f}% of days: "
            f"{abs(es)*100:.2f}%"
        ),
    }
    return json.dumps(result)


@function_tool
async def compute_max_drawdown(
    ctx: RunContextWrapper[AppContext],
    portfolio_weights_json: str,
    prices_json: str,
) -> str:
    """Compute current and maximum drawdown with duration."""
    weights = json.loads(portfolio_weights_json)
    prices_dict = json.loads(prices_json)

    port_returns = _build_portfolio_returns(weights, prices_dict)
    if len(port_returns) < 5:
        return json.dumps({"error": "Insufficient data"})

    cum_returns = (1 + port_returns).cumprod()
    running_max = cum_returns.cummax()
    drawdown = (cum_returns - running_max) / running_max

    max_dd = float(drawdown.min())
    current_dd = float(drawdown.iloc[-1])

    # Drawdown duration (current)
    if current_dd < 0:
        peak_idx = cum_returns[:drawdown.idxmin()].idxmax() if max_dd < 0 else cum_returns.index[-1]
        current_dd_days = (drawdown.index[-1] - peak_idx).days if hasattr(peak_idx, 'date') else 0
    else:
        current_dd_days = 0

    result = {
        "max_drawdown_pct": round(max_dd * 100, 2),
        "current_drawdown_pct": round(current_dd * 100, 2),
        "current_drawdown_days": current_dd_days,
        "interpretation": (
            "no drawdown" if current_dd >= -0.001 else
            "minor drawdown" if current_dd > -0.05 else
            "moderate drawdown" if current_dd > -0.10 else
            "significant drawdown" if current_dd > -0.20 else
            "severe drawdown"
        ),
    }
    return json.dumps(result)


@function_tool
async def compute_beta_exposure(
    ctx: RunContextWrapper[AppContext],
    portfolio_weights_json: str,
    prices_json: str,
) -> str:
    """Portfolio beta vs SPY and sector breakdown."""
    import yfinance as yf

    weights = json.loads(portfolio_weights_json)
    prices_dict = json.loads(prices_json)

    # Get SPY returns
    spy = yf.download("SPY", period="1y", progress=False)
    if spy.empty:
        return json.dumps({"error": "Could not fetch SPY"})
    spy_returns = spy["Close"].pct_change().dropna()

    port_returns = _build_portfolio_returns(weights, prices_dict)
    common = port_returns.index.intersection(spy_returns.index)
    if len(common) < 30:
        return json.dumps({"error": "Insufficient overlapping data"})

    y = port_returns.loc[common].values
    x = spy_returns.loc[common].values

    # Portfolio beta
    cov = np.cov(y, x)
    beta = float(cov[0, 1] / cov[1, 1]) if cov[1, 1] != 0 else 1.0

    # Classify tickers by asset class
    equity_tickers = []
    bond_tickers = []
    commodity_tickers = []
    crypto_tickers = []
    bond_set = {"TLT", "IEF", "HYG", "AGG", "BND", "LQD"}
    commodity_set = {"GLD", "SLV", "USO", "DBA"}
    crypto_set = {"BTC", "ETH", "SOL", "AVAX"}

    for t in weights:
        if t in crypto_set:
            crypto_tickers.append(t)
        elif t in bond_set:
            bond_tickers.append(t)
        elif t in commodity_set:
            commodity_tickers.append(t)
        else:
            equity_tickers.append(t)

    equity_weight = sum(weights.get(t, 0) for t in equity_tickers)
    bond_weight = sum(weights.get(t, 0) for t in bond_tickers)
    commodity_weight = sum(weights.get(t, 0) for t in commodity_tickers)
    crypto_weight = sum(weights.get(t, 0) for t in crypto_tickers)

    result = {
        "portfolio_beta": round(beta, 3),
        "beta_interpretation": (
            "defensive" if beta < 0.8 else
            "moderate" if beta < 1.2 else
            "aggressive"
        ),
        "exposure_breakdown": {
            "equity_pct": round(equity_weight * 100, 1),
            "bond_pct": round(bond_weight * 100, 1),
            "commodity_pct": round(commodity_weight * 100, 1),
            "crypto_pct": round(crypto_weight * 100, 1),
        },
    }
    return json.dumps(result)
