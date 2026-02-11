"""Portfolio optimization — risk-parity, mean-variance, max Sharpe, efficient frontier, Black-Litterman, constraints."""

from __future__ import annotations

import json

import numpy as np
from scipy.optimize import minimize
from agents import RunContextWrapper, function_tool

from portfolio_advisor.agents.context import AppContext


@function_tool
async def optimize_risk_parity(
    ctx: RunContextWrapper[AppContext],
    tickers: str,
    vol_forecasts_json: str,
    correlation_json: str,
) -> str:
    """Inverse-volatility weighting adjusted for correlations (risk parity)."""
    ticker_list = [t.strip() for t in tickers.split(",")]
    vol_data = json.loads(vol_forecasts_json)  # {ticker: annualized_vol}
    corr_data = json.loads(correlation_json)  # {ticker: {ticker: corr}}

    n = len(ticker_list)
    vols = np.array([vol_data.get(t, 20.0) / 100.0 for t in ticker_list])

    # Build covariance matrix
    corr_matrix = np.eye(n)
    for i, t1 in enumerate(ticker_list):
        for j, t2 in enumerate(ticker_list):
            if t1 in corr_data and t2 in corr_data.get(t1, {}):
                corr_matrix[i, j] = corr_data[t1][t2]

    cov_matrix = np.outer(vols, vols) * corr_matrix

    # Risk parity: each asset contributes equally to portfolio risk
    def risk_budget_obj(w):
        port_var = w @ cov_matrix @ w
        port_vol = np.sqrt(port_var)
        marginal_contrib = cov_matrix @ w
        risk_contrib = w * marginal_contrib / port_vol
        target = port_vol / n
        return np.sum((risk_contrib - target) ** 2)

    w0 = np.ones(n) / n
    bounds = [(0.01, 0.5) for _ in range(n)]
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    result_opt = minimize(risk_budget_obj, w0, method="SLSQP", bounds=bounds, constraints=constraints)
    weights = result_opt.x if result_opt.success else w0

    # Normalize
    weights = weights / weights.sum()

    allocation = {t: round(float(w) * 100, 2) for t, w in zip(ticker_list, weights)}
    port_vol = float(np.sqrt(weights @ cov_matrix @ weights)) * 100

    return json.dumps({
        "method": "risk_parity",
        "allocations_pct": allocation,
        "portfolio_vol_annualized_pct": round(port_vol, 2),
        "optimization_success": bool(result_opt.success),
    })


@function_tool
async def optimize_mean_variance(
    ctx: RunContextWrapper[AppContext],
    tickers: str,
    return_forecasts_json: str,
    vol_forecasts_json: str,
    correlation_json: str,
    risk_tolerance: str = "moderate",
) -> str:
    """Mean-variance optimization with risk aversion mapped from tolerance level."""
    ticker_list = [t.strip() for t in tickers.split(",")]
    ret_data = json.loads(return_forecasts_json)  # {ticker: expected_return_pct}
    vol_data = json.loads(vol_forecasts_json)
    corr_data = json.loads(correlation_json)

    risk_aversion_map = {"conservative": 4.0, "moderate": 2.0, "aggressive": 0.5}
    gamma = risk_aversion_map.get(risk_tolerance, 2.0)

    n = len(ticker_list)
    mu = np.array([ret_data.get(t, 0.0) / 100.0 for t in ticker_list])
    vols = np.array([vol_data.get(t, 20.0) / 100.0 for t in ticker_list])

    corr_matrix = np.eye(n)
    for i, t1 in enumerate(ticker_list):
        for j, t2 in enumerate(ticker_list):
            if t1 in corr_data and t2 in corr_data.get(t1, {}):
                corr_matrix[i, j] = corr_data[t1][t2]

    cov_matrix = np.outer(vols, vols) * corr_matrix

    # Maximize: mu'w - (gamma/2) * w'Σw
    def neg_utility(w):
        ret = mu @ w
        risk = w @ cov_matrix @ w
        return -(ret - gamma / 2 * risk)

    w0 = np.ones(n) / n
    bounds = [(0.0, 0.30) for _ in range(n)]
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    result_opt = minimize(neg_utility, w0, method="SLSQP", bounds=bounds, constraints=constraints)
    weights = result_opt.x if result_opt.success else w0
    weights = np.maximum(weights, 0)
    weights = weights / weights.sum()

    allocation = {t: round(float(w) * 100, 2) for t, w in zip(ticker_list, weights)}
    port_ret = float(mu @ weights) * 100
    port_vol = float(np.sqrt(weights @ cov_matrix @ weights)) * 100
    sharpe = port_ret / port_vol if port_vol > 0 else 0

    return json.dumps({
        "method": "mean_variance",
        "risk_tolerance": risk_tolerance,
        "risk_aversion_gamma": gamma,
        "allocations_pct": allocation,
        "expected_return_pct": round(port_ret, 2),
        "portfolio_vol_annualized_pct": round(port_vol, 2),
        "sharpe_ratio": round(sharpe, 3),
        "optimization_success": bool(result_opt.success),
    })


@function_tool
async def optimize_max_sharpe(
    ctx: RunContextWrapper[AppContext],
    tickers: str,
    return_forecasts_json: str,
    vol_forecasts_json: str,
    correlation_json: str,
) -> str:
    """Find the tangency portfolio that maximizes the Sharpe ratio (return per unit of risk).

    The max-Sharpe portfolio lies on the efficient frontier at the point where the
    Capital Market Line is tangent. Uses risk-free rate = 5% annualized.
    """
    ticker_list = [t.strip() for t in tickers.split(",")]
    ret_data = json.loads(return_forecasts_json)
    vol_data = json.loads(vol_forecasts_json)
    corr_data = json.loads(correlation_json)

    n = len(ticker_list)
    mu = np.array([ret_data.get(t, 0.0) / 100.0 for t in ticker_list])
    vols = np.array([vol_data.get(t, 20.0) / 100.0 for t in ticker_list])
    rf = 0.05  # Risk-free rate

    corr_matrix = np.eye(n)
    for i, t1 in enumerate(ticker_list):
        for j, t2 in enumerate(ticker_list):
            if t1 in corr_data and t2 in corr_data.get(t1, {}):
                corr_matrix[i, j] = corr_data[t1][t2]
    cov_matrix = np.outer(vols, vols) * corr_matrix

    # Maximize Sharpe: max (mu'w - rf) / sqrt(w'Σw)
    def neg_sharpe(w):
        port_ret = mu @ w
        port_vol = np.sqrt(w @ cov_matrix @ w)
        return -(port_ret - rf) / port_vol if port_vol > 1e-10 else 0.0

    w0 = np.ones(n) / n
    bounds = [(0.0, 0.40) for _ in range(n)]
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    result_opt = minimize(neg_sharpe, w0, method="SLSQP", bounds=bounds, constraints=constraints)
    weights = result_opt.x if result_opt.success else w0
    weights = np.maximum(weights, 0)
    weights = weights / weights.sum()

    allocation = {t: round(float(w) * 100, 2) for t, w in zip(ticker_list, weights)}
    port_ret = float(mu @ weights) * 100
    port_vol = float(np.sqrt(weights @ cov_matrix @ weights)) * 100
    sharpe = (port_ret - rf * 100) / port_vol if port_vol > 0 else 0

    return json.dumps({
        "method": "max_sharpe",
        "allocations_pct": allocation,
        "expected_return_pct": round(port_ret, 2),
        "portfolio_vol_annualized_pct": round(port_vol, 2),
        "sharpe_ratio": round(sharpe, 3),
        "risk_free_rate_pct": rf * 100,
        "optimization_success": bool(result_opt.success),
    })


@function_tool
async def compute_efficient_frontier(
    ctx: RunContextWrapper[AppContext],
    tickers: str,
    return_forecasts_json: str,
    vol_forecasts_json: str,
    correlation_json: str,
    n_points: int = 15,
) -> str:
    """Generate points along the efficient frontier — the set of portfolios offering
    maximum return for each level of risk.

    Returns n_points portfolios from minimum-variance to maximum-return, plus the
    current portfolio and max-Sharpe portfolio for comparison.
    """
    ticker_list = [t.strip() for t in tickers.split(",")]
    ret_data = json.loads(return_forecasts_json)
    vol_data = json.loads(vol_forecasts_json)
    corr_data = json.loads(correlation_json)

    n = len(ticker_list)
    mu = np.array([ret_data.get(t, 0.0) / 100.0 for t in ticker_list])
    vols = np.array([vol_data.get(t, 20.0) / 100.0 for t in ticker_list])

    corr_matrix = np.eye(n)
    for i, t1 in enumerate(ticker_list):
        for j, t2 in enumerate(ticker_list):
            if t1 in corr_data and t2 in corr_data.get(t1, {}):
                corr_matrix[i, j] = corr_data[t1][t2]
    cov_matrix = np.outer(vols, vols) * corr_matrix

    # Target returns from min to max
    target_returns = np.linspace(mu.min(), mu.max(), n_points)
    frontier = []

    for target_ret in target_returns:
        # Minimize variance subject to target return
        def portfolio_vol(w):
            return w @ cov_matrix @ w

        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
            {"type": "eq", "fun": lambda w, tr=target_ret: mu @ w - tr},
        ]
        bounds = [(0.0, 0.40) for _ in range(n)]
        w0 = np.ones(n) / n

        result_opt = minimize(portfolio_vol, w0, method="SLSQP", bounds=bounds, constraints=constraints)
        if result_opt.success:
            w = result_opt.x
            p_ret = float(mu @ w) * 100
            p_vol = float(np.sqrt(w @ cov_matrix @ w)) * 100
            sharpe = (p_ret - 5.0) / p_vol if p_vol > 0 else 0
            frontier.append({
                "return_pct": round(p_ret, 2),
                "vol_pct": round(p_vol, 2),
                "sharpe": round(sharpe, 3),
                "weights": {t: round(float(w_i) * 100, 1) for t, w_i in zip(ticker_list, w)},
            })

    return json.dumps({
        "method": "efficient_frontier",
        "tickers": ticker_list,
        "frontier_points": frontier,
        "n_points": len(frontier),
        "min_vol_portfolio": frontier[0] if frontier else None,
        "max_return_portfolio": frontier[-1] if frontier else None,
        "interpretation": (
            f"Efficient frontier with {len(frontier)} points. "
            f"Min-vol portfolio: {frontier[0]['vol_pct']:.1f}% vol, {frontier[0]['return_pct']:.1f}% return. "
            f"Max-return portfolio: {frontier[-1]['vol_pct']:.1f}% vol, {frontier[-1]['return_pct']:.1f}% return."
        ) if frontier else "Could not generate efficient frontier.",
    })


@function_tool
async def optimize_black_litterman(
    ctx: RunContextWrapper[AppContext],
    tickers: str,
    market_caps_json: str,
    views_json: str,
    vol_forecasts_json: str,
    correlation_json: str,
    tau: float = 0.05,
) -> str:
    """Black-Litterman model: blends market equilibrium with analyst views.

    Starts from market-cap-weighted equilibrium (implied returns), then tilts
    based on subjective views. More stable than pure mean-variance.

    market_caps_json: {ticker: market_cap_billions}
    views_json: [{ticker: str, view_return_pct: float, confidence: float}]
    """
    ticker_list = [t.strip() for t in tickers.split(",")]
    mcaps = json.loads(market_caps_json)
    views_list = json.loads(views_json)
    vol_data = json.loads(vol_forecasts_json)
    corr_data = json.loads(correlation_json)

    n = len(ticker_list)
    vols = np.array([vol_data.get(t, 20.0) / 100.0 for t in ticker_list])

    corr_matrix = np.eye(n)
    for i, t1 in enumerate(ticker_list):
        for j, t2 in enumerate(ticker_list):
            if t1 in corr_data and t2 in corr_data.get(t1, {}):
                corr_matrix[i, j] = corr_data[t1][t2]
    cov_matrix = np.outer(vols, vols) * corr_matrix

    # Market-cap weights (equilibrium)
    caps = np.array([mcaps.get(t, 1.0) for t in ticker_list])
    w_mkt = caps / caps.sum()

    # Risk aversion coefficient (implied from market)
    delta = 2.5  # Standard assumption

    # Implied equilibrium returns: pi = delta * Sigma * w_mkt
    pi = delta * cov_matrix @ w_mkt

    # Build view matrices P, Q, Omega
    k = len(views_list)  # Number of views
    if k == 0:
        # No views — return equilibrium portfolio
        allocation = {t: round(float(w) * 100, 2) for t, w in zip(ticker_list, w_mkt)}
        return json.dumps({
            "method": "black_litterman",
            "note": "No views provided — returning market equilibrium",
            "allocations_pct": allocation,
            "implied_returns_pct": {t: round(float(r) * 100, 2) for t, r in zip(ticker_list, pi)},
        })

    P = np.zeros((k, n))
    Q = np.zeros(k)
    omega_diag = np.zeros(k)

    for i, view in enumerate(views_list):
        t = view.get("ticker", "")
        if t in ticker_list:
            idx = ticker_list.index(t)
            P[i, idx] = 1.0
            Q[i] = view.get("view_return_pct", 0.0) / 100.0
            # Confidence → omega: lower confidence = higher uncertainty
            conf = max(0.1, min(1.0, view.get("confidence", 0.5)))
            omega_diag[i] = (1.0 / conf - 1.0) * tau * cov_matrix[idx, idx]

    Omega = np.diag(omega_diag)

    # BL posterior expected returns
    tau_sigma = tau * cov_matrix
    inv_tau_sigma = np.linalg.inv(tau_sigma)
    inv_omega = np.linalg.inv(Omega) if k > 0 else np.zeros((k, k))

    # E[R] = [(tau*Sigma)^-1 + P'*Omega^-1*P]^-1 * [(tau*Sigma)^-1 * pi + P'*Omega^-1 * Q]
    M = inv_tau_sigma + P.T @ inv_omega @ P
    bl_returns = np.linalg.solve(M, inv_tau_sigma @ pi + P.T @ inv_omega @ Q)

    # BL optimal weights via mean-variance with BL returns
    def neg_utility(w):
        ret = bl_returns @ w
        risk = w @ cov_matrix @ w
        return -(ret - delta / 2 * risk)

    w0 = w_mkt.copy()
    bounds = [(0.0, 0.35) for _ in range(n)]
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    result_opt = minimize(neg_utility, w0, method="SLSQP", bounds=bounds, constraints=constraints)
    weights = result_opt.x if result_opt.success else w_mkt
    weights = np.maximum(weights, 0)
    weights = weights / weights.sum()

    allocation = {t: round(float(w) * 100, 2) for t, w in zip(ticker_list, weights)}
    port_ret = float(bl_returns @ weights) * 100
    port_vol = float(np.sqrt(weights @ cov_matrix @ weights)) * 100

    # Show tilt from equilibrium
    tilts = {t: round((float(weights[i]) - float(w_mkt[i])) * 100, 2) for i, t in enumerate(ticker_list)}

    return json.dumps({
        "method": "black_litterman",
        "allocations_pct": allocation,
        "equilibrium_weights_pct": {t: round(float(w) * 100, 2) for t, w in zip(ticker_list, w_mkt)},
        "implied_returns_pct": {t: round(float(r) * 100, 2) for t, r in zip(ticker_list, pi)},
        "bl_expected_returns_pct": {t: round(float(r) * 100, 2) for t, r in zip(ticker_list, bl_returns)},
        "tilts_from_equilibrium_pct": tilts,
        "expected_return_pct": round(port_ret, 2),
        "portfolio_vol_annualized_pct": round(port_vol, 2),
        "views_applied": len(views_list),
        "tau": tau,
        "optimization_success": bool(result_opt.success),
    })


@function_tool
async def check_concentration_limits(
    ctx: RunContextWrapper[AppContext],
    proposed_weights_json: str,
    max_position_pct: float = 15.0,
) -> str:
    """Validate proposed allocations against concentration limits."""
    weights = json.loads(proposed_weights_json)  # {ticker: weight_pct}
    violations = []
    adjusted = {}

    for ticker, weight in weights.items():
        if weight > max_position_pct:
            violations.append({
                "ticker": ticker,
                "proposed_pct": weight,
                "max_pct": max_position_pct,
            })
            adjusted[ticker] = max_position_pct
        else:
            adjusted[ticker] = weight

    # Redistribute excess to underweight positions
    total = sum(adjusted.values())
    if total != 100.0 and total > 0:
        factor = 100.0 / total
        adjusted = {t: round(w * factor, 2) for t, w in adjusted.items()}

    return json.dumps({
        "violations": violations,
        "has_violations": len(violations) > 0,
        "adjusted_weights_pct": adjusted,
    })


@function_tool
async def apply_risk_controls(
    ctx: RunContextWrapper[AppContext],
    proposed_weights_json: str,
    current_portfolio_json: str,
    risk_metrics_json: str,
    user_prefs_json: str,
) -> str:
    """Full constraint pass: concentration, drawdown awareness, vol-aware sizing, excluded assets, cash target."""
    proposed = json.loads(proposed_weights_json)  # {ticker: weight_pct}
    current = json.loads(current_portfolio_json)  # {ticker: weight_pct}
    risk = json.loads(risk_metrics_json)
    prefs = json.loads(user_prefs_json)

    max_position = prefs.get("max_position_pct", 15.0)
    cash_target = prefs.get("cash_target_pct", 10.0)
    excluded = set(prefs.get("excluded_assets", []))
    adjustments = []

    final_weights = dict(proposed)

    # 1. Remove excluded assets
    for ticker in list(final_weights.keys()):
        if ticker in excluded:
            adjustments.append(f"Removed {ticker} (excluded)")
            del final_weights[ticker]

    # 2. Concentration limits
    for ticker in list(final_weights.keys()):
        if final_weights[ticker] > max_position:
            adjustments.append(
                f"Capped {ticker}: {final_weights[ticker]:.1f}% → {max_position:.1f}%"
            )
            final_weights[ticker] = max_position

    # 3. Drawdown awareness — reduce equity exposure if in significant drawdown
    current_dd = risk.get("current_drawdown_pct", 0)
    if current_dd < -10:
        dd_factor = max(0.7, 1 + current_dd / 50)  # Scale down as drawdown deepens
        for ticker in list(final_weights.keys()):
            # Only scale risk assets, not bonds/cash
            if ticker not in {"TLT", "IEF", "HYG", "GLD", "SLV"}:
                old = final_weights[ticker]
                final_weights[ticker] = round(old * dd_factor, 2)
        adjustments.append(
            f"Applied drawdown factor {dd_factor:.2f} (current DD: {current_dd:.1f}%)"
        )

    # 4. Ensure cash target
    invested = sum(final_weights.values())
    cash_pct = 100.0 - invested
    if cash_pct < cash_target:
        scale = (100.0 - cash_target) / invested if invested > 0 else 1.0
        final_weights = {t: round(w * scale, 2) for t, w in final_weights.items()}
        adjustments.append(f"Scaled to maintain {cash_target:.0f}% cash target")

    # Normalize
    total_invested = sum(final_weights.values())
    cash_final = round(100.0 - total_invested, 2)

    return json.dumps({
        "final_weights_pct": final_weights,
        "cash_pct": cash_final,
        "adjustments_made": adjustments,
        "total_adjustments": len(adjustments),
    })
