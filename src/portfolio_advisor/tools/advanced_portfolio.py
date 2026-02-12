"""Advanced portfolio optimization — CVaR, HRP, Kelly, Max Diversification, Entropy, Transaction Costs."""

from __future__ import annotations

import json
import logging

import numpy as np
from scipy.optimize import minimize
from agents import RunContextWrapper, function_tool

from portfolio_advisor.agents.context import AppContext

logger = logging.getLogger(__name__)


# ── Helper: build covariance matrix from inputs ──────────────────────────────


def _build_cov_matrix(
    ticker_list: list[str],
    vol_data: dict,
    corr_data: dict,
) -> np.ndarray:
    """Build NxN covariance matrix from volatility and correlation data."""
    n = len(ticker_list)
    vols = np.array([vol_data.get(t, 20.0) / 100.0 for t in ticker_list])
    corr_matrix = np.eye(n)
    for i, t1 in enumerate(ticker_list):
        for j, t2 in enumerate(ticker_list):
            if t1 in corr_data and t2 in corr_data.get(t1, {}):
                corr_matrix[i, j] = corr_data[t1][t2]
    return np.outer(vols, vols) * corr_matrix, vols


# ── Pure computation functions ────────────────────────────────────────────────


def optimize_cvar_raw(
    returns_matrix: np.ndarray,
    tickers: list[str],
    confidence: float = 0.95,
) -> dict:
    """CVaR (Conditional Value-at-Risk) optimization via Rockafellar-Uryasev.

    Minimize: alpha + 1/(1-beta) * 1/T * sum(max(0, -r_t @ w - alpha))
    Subject to: sum(w) = 1, w >= 0
    """
    n_obs, n_assets = returns_matrix.shape
    if n_obs < 60:
        return {"error": "Insufficient data (need 60+ observations)"}

    beta = confidence

    # SLSQP formulation (smooth approximation)
    def cvar_objective(params):
        w = params[:n_assets]
        alpha = params[n_assets]
        losses = -returns_matrix @ w  # portfolio losses
        excess = np.maximum(0.0, losses - alpha)
        return alpha + np.mean(excess) / (1 - beta)

    # Initial guess
    w0 = np.ones(n_assets) / n_assets
    alpha0 = np.percentile(-returns_matrix @ w0, beta * 100)
    x0 = np.concatenate([w0, [alpha0]])

    bounds = [(0.0, 0.40) for _ in range(n_assets)] + [(-1.0, 1.0)]
    constraints = [{"type": "eq", "fun": lambda x: np.sum(x[:n_assets]) - 1.0}]

    result = minimize(cvar_objective, x0, method="SLSQP", bounds=bounds, constraints=constraints)

    if result.success:
        weights = result.x[:n_assets]
    else:
        weights = np.ones(n_assets) / n_assets

    weights = np.maximum(weights, 0)
    weights = weights / weights.sum()

    # Compute resulting metrics
    port_returns = returns_matrix @ weights
    port_var = float(np.percentile(port_returns, (1 - beta) * 100))
    tail = port_returns[port_returns <= port_var]
    port_cvar = float(tail.mean()) if len(tail) > 0 else port_var
    port_vol = float(np.std(port_returns)) * np.sqrt(252)
    port_ret = float(np.mean(port_returns)) * 252

    return {
        "method": "cvar_optimization",
        "confidence": beta,
        "allocations_pct": {t: round(float(w) * 100, 2) for t, w in zip(tickers, weights)},
        "var_daily_pct": round(port_var * 100, 3),
        "cvar_daily_pct": round(port_cvar * 100, 3),
        "expected_return_pct": round(port_ret * 100, 2),
        "portfolio_vol_annualized_pct": round(port_vol * 100, 2),
        "optimization_success": bool(result.success),
    }


def optimize_hrp_raw(
    returns_matrix: np.ndarray,
    tickers: list[str],
) -> dict:
    """Hierarchical Risk Parity (Lopez de Prado, 2016).

    1. Correlation distance: d = sqrt(2(1-rho))
    2. Hierarchical clustering (single linkage)
    3. Quasi-diagonalization
    4. Recursive bisection with inverse-variance weighting
    """
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import squareform

    n_obs, n_assets = returns_matrix.shape
    if n_obs < 30:
        return {"error": "Insufficient data (need 30+ observations)"}

    # Covariance and correlation
    cov = np.cov(returns_matrix.T)
    corr = np.corrcoef(returns_matrix.T)

    # Step 1: Distance matrix
    dist = np.sqrt(2.0 * (1.0 - corr))
    np.fill_diagonal(dist, 0.0)
    condensed = squareform(dist, checks=False)

    # Step 2: Hierarchical clustering
    Z = linkage(condensed, method="single")

    # Step 3: Quasi-diagonalization (get leaf order from dendrogram)
    sort_ix = list(leaves_list(Z))

    # Step 4: Recursive bisection
    weights = np.ones(n_assets)

    def _get_cluster_var(cov_matrix, indices):
        """Inverse-variance portfolio variance for a cluster."""
        sub_cov = cov_matrix[np.ix_(indices, indices)]
        ivp = 1.0 / np.diag(sub_cov)
        ivp = ivp / ivp.sum()
        return float(ivp @ sub_cov @ ivp)

    def _recursive_bisection(cov_matrix, sort_idx, w):
        if len(sort_idx) <= 1:
            return
        mid = len(sort_idx) // 2
        left = sort_idx[:mid]
        right = sort_idx[mid:]

        left_var = _get_cluster_var(cov_matrix, left)
        right_var = _get_cluster_var(cov_matrix, right)

        # Allocate inversely proportional to cluster variance
        alloc = 1.0 - left_var / (left_var + right_var) if (left_var + right_var) > 0 else 0.5

        for i in left:
            w[i] *= alloc
        for i in right:
            w[i] *= (1.0 - alloc)

        _recursive_bisection(cov_matrix, left, w)
        _recursive_bisection(cov_matrix, right, w)

    _recursive_bisection(cov, sort_ix, weights)
    weights = weights / weights.sum()

    # Portfolio metrics
    port_vol = float(np.sqrt(weights @ cov @ weights)) * np.sqrt(252) * 100

    return {
        "method": "hierarchical_risk_parity",
        "allocations_pct": {t: round(float(w) * 100, 2) for t, w in zip(tickers, weights)},
        "portfolio_vol_annualized_pct": round(port_vol, 2),
        "cluster_order": [tickers[i] for i in sort_ix],
    }


def compute_kelly_raw(
    mu: np.ndarray,
    cov: np.ndarray,
    tickers: list[str],
) -> dict:
    """Kelly Criterion for optimal geometric growth.

    Full Kelly: f* = Sigma^{-1} @ mu
    Half-Kelly: f*/2 (more practical, reduces variance)
    """
    n = len(tickers)
    if n == 0:
        return {"error": "No assets provided"}

    try:
        cov_inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        cov_inv = np.linalg.pinv(cov)

    # Full Kelly
    full_kelly = cov_inv @ mu
    half_kelly = full_kelly / 2.0

    # Normalize to sum to 1 (constrained Kelly)
    def _normalize(w):
        # Clip negative weights for long-only
        w_long = np.maximum(w, 0)
        total = w_long.sum()
        return w_long / total if total > 0 else np.ones(n) / n

    full_norm = _normalize(full_kelly)
    half_norm = _normalize(half_kelly)

    # Expected log growth (Kelly objective)
    def _log_growth(w):
        return float(w @ mu - 0.5 * w @ cov @ w) * 252

    return {
        "method": "kelly_criterion",
        "full_kelly_pct": {t: round(float(w) * 100, 2) for t, w in zip(tickers, full_norm)},
        "half_kelly_pct": {t: round(float(w) * 100, 2) for t, w in zip(tickers, half_norm)},
        "full_kelly_raw": {t: round(float(w), 4) for t, w in zip(tickers, full_kelly)},
        "expected_log_growth_full_pct": round(_log_growth(full_norm) * 100, 2),
        "expected_log_growth_half_pct": round(_log_growth(half_norm) * 100, 2),
        "recommendation": "half_kelly",
        "interpretation": (
            "Half-Kelly recommended for practical use (reduces variance of growth rate "
            "by 75% with only 50% reduction in expected geometric growth). "
            "Full Kelly is optimal theoretically but too aggressive for most portfolios."
        ),
    }


def optimize_max_diversification_raw(
    returns_matrix: np.ndarray,
    tickers: list[str],
) -> dict:
    """Maximize the Diversification Ratio: DR = (w' @ sigma) / sqrt(w' @ Sigma @ w).

    Higher DR means more diversification benefit from correlation.
    """
    n_obs, n_assets = returns_matrix.shape
    if n_obs < 30:
        return {"error": "Insufficient data (need 30+ observations)"}

    cov = np.cov(returns_matrix.T)
    vols = np.sqrt(np.diag(cov))

    def neg_div_ratio(w):
        weighted_vol = w @ vols
        port_vol = np.sqrt(w @ cov @ w)
        return -weighted_vol / port_vol if port_vol > 1e-10 else 0.0

    w0 = np.ones(n_assets) / n_assets
    bounds = [(0.0, 0.40) for _ in range(n_assets)]
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    result = minimize(neg_div_ratio, w0, method="SLSQP", bounds=bounds, constraints=constraints)
    weights = result.x if result.success else w0
    weights = np.maximum(weights, 0)
    weights = weights / weights.sum()

    # Compute metrics
    port_vol = float(np.sqrt(weights @ cov @ weights)) * np.sqrt(252) * 100
    weighted_avg_vol = float(weights @ vols) * np.sqrt(252) * 100
    div_ratio = weighted_avg_vol / port_vol if port_vol > 0 else 1.0

    return {
        "method": "max_diversification",
        "allocations_pct": {t: round(float(w) * 100, 2) for t, w in zip(tickers, weights)},
        "diversification_ratio": round(div_ratio, 3),
        "portfolio_vol_annualized_pct": round(port_vol, 2),
        "weighted_avg_vol_pct": round(weighted_avg_vol, 2),
        "diversification_benefit_pct": round((1 - port_vol / weighted_avg_vol) * 100, 1)
        if weighted_avg_vol > 0
        else 0.0,
        "optimization_success": bool(result.success),
    }


def optimize_entropy_raw(
    returns_matrix: np.ndarray,
    tickers: list[str],
    target_return: float | None = None,
) -> dict:
    """Maximize Shannon entropy: H = -sum(w_i * ln(w_i)).

    Information-theoretic diversification. Optionally constrained to a target return.
    """
    n_obs, n_assets = returns_matrix.shape
    if n_obs < 30:
        return {"error": "Insufficient data (need 30+ observations)"}

    mu = returns_matrix.mean(axis=0)

    def neg_entropy(w):
        w_pos = np.maximum(w, 1e-10)
        return float(np.sum(w_pos * np.log(w_pos)))

    w0 = np.ones(n_assets) / n_assets
    bounds = [(0.0, 0.50) for _ in range(n_assets)]
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    if target_return is not None:
        constraints.append({
            "type": "ineq",
            "fun": lambda w: mu @ w - target_return,
        })

    result = minimize(neg_entropy, w0, method="SLSQP", bounds=bounds, constraints=constraints)
    weights = result.x if result.success else w0
    weights = np.maximum(weights, 0)
    weights = weights / weights.sum()

    # Entropy of result
    entropy = float(-np.sum(weights * np.log(np.maximum(weights, 1e-10))))
    max_entropy = float(np.log(n_assets))

    # Portfolio metrics
    cov = np.cov(returns_matrix.T)
    port_vol = float(np.sqrt(weights @ cov @ weights)) * np.sqrt(252) * 100
    port_ret = float(mu @ weights) * 252 * 100

    return {
        "method": "max_entropy",
        "allocations_pct": {t: round(float(w) * 100, 2) for t, w in zip(tickers, weights)},
        "entropy": round(entropy, 4),
        "max_entropy": round(max_entropy, 4),
        "normalized_entropy": round(entropy / max_entropy if max_entropy > 0 else 0, 4),
        "expected_return_pct": round(port_ret, 2),
        "portfolio_vol_annualized_pct": round(port_vol, 2),
        "optimization_success": bool(result.success),
    }


# ── @function_tool wrappers ──────────────────────────────────────────────────


@function_tool
async def optimize_cvar(
    ctx: RunContextWrapper[AppContext],
    tickers: str,
    prices_json: str,
    confidence: float = 0.95,
) -> str:
    """CVaR (tail-risk-aware) portfolio optimization. Minimizes expected loss in the worst (1-confidence)% of scenarios. prices_json: {ticker: [bars]}."""
    import pandas as pd

    from portfolio_advisor.tools.technical_indicators import _prices_to_series

    ticker_list = [t.strip() for t in tickers.split(",")]
    prices_dict = json.loads(prices_json)

    all_returns = {}
    for t in ticker_list:
        if t not in prices_dict:
            continue
        df = _prices_to_series(json.dumps(prices_dict[t]))
        all_returns[t] = df["close"].pct_change().dropna()

    if len(all_returns) < 2:
        return json.dumps({"error": "Need data for at least 2 tickers"})

    ret_df = pd.DataFrame(all_returns).dropna()
    valid_tickers = list(ret_df.columns)

    raw = optimize_cvar_raw(ret_df.values, valid_tickers, confidence)
    if "error" in raw:
        return json.dumps({"error": raw["error"]})
    return json.dumps(raw)


@function_tool
async def optimize_hrp(
    ctx: RunContextWrapper[AppContext],
    tickers: str,
    prices_json: str,
) -> str:
    """Hierarchical Risk Parity optimization (Lopez de Prado). No optimizer instability — robust to estimation error. Uses correlation clustering + recursive bisection. prices_json: {ticker: [bars]}."""
    import pandas as pd

    from portfolio_advisor.tools.technical_indicators import _prices_to_series

    ticker_list = [t.strip() for t in tickers.split(",")]
    prices_dict = json.loads(prices_json)

    all_returns = {}
    for t in ticker_list:
        if t not in prices_dict:
            continue
        df = _prices_to_series(json.dumps(prices_dict[t]))
        all_returns[t] = df["close"].pct_change().dropna()

    if len(all_returns) < 2:
        return json.dumps({"error": "Need data for at least 2 tickers"})

    ret_df = pd.DataFrame(all_returns).dropna()
    valid_tickers = list(ret_df.columns)

    raw = optimize_hrp_raw(ret_df.values, valid_tickers)
    if "error" in raw:
        return json.dumps({"error": raw["error"]})
    return json.dumps(raw)


@function_tool
async def compute_kelly_criterion(
    ctx: RunContextWrapper[AppContext],
    tickers: str,
    return_forecasts_json: str,
    vol_forecasts_json: str,
    correlation_json: str,
) -> str:
    """Kelly Criterion for optimal position sizing. Returns Full Kelly (theoretical max) and Half Kelly (practical recommendation). Inputs: expected returns, vols, correlations."""
    ticker_list = [t.strip() for t in tickers.split(",")]
    ret_data = json.loads(return_forecasts_json)
    vol_data = json.loads(vol_forecasts_json)
    corr_data = json.loads(correlation_json)

    mu = np.array([ret_data.get(t, 0.0) / 100.0 / 252.0 for t in ticker_list])  # daily

    cov_matrix, _ = _build_cov_matrix(ticker_list, vol_data, corr_data)
    # Convert annualized vol to daily
    cov_daily = cov_matrix / 252.0

    raw = compute_kelly_raw(mu, cov_daily, ticker_list)
    if "error" in raw:
        return json.dumps({"error": raw["error"]})
    return json.dumps(raw)


@function_tool
async def optimize_max_diversification(
    ctx: RunContextWrapper[AppContext],
    tickers: str,
    prices_json: str,
) -> str:
    """Maximize the Diversification Ratio (weighted avg vol / portfolio vol). Exploits correlation structure for maximum diversification benefit. prices_json: {ticker: [bars]}."""
    import pandas as pd

    from portfolio_advisor.tools.technical_indicators import _prices_to_series

    ticker_list = [t.strip() for t in tickers.split(",")]
    prices_dict = json.loads(prices_json)

    all_returns = {}
    for t in ticker_list:
        if t not in prices_dict:
            continue
        df = _prices_to_series(json.dumps(prices_dict[t]))
        all_returns[t] = df["close"].pct_change().dropna()

    if len(all_returns) < 2:
        return json.dumps({"error": "Need data for at least 2 tickers"})

    ret_df = pd.DataFrame(all_returns).dropna()
    valid_tickers = list(ret_df.columns)

    raw = optimize_max_diversification_raw(ret_df.values, valid_tickers)
    if "error" in raw:
        return json.dumps({"error": raw["error"]})
    return json.dumps(raw)


@function_tool
async def optimize_entropy_weighted(
    ctx: RunContextWrapper[AppContext],
    tickers: str,
    prices_json: str,
) -> str:
    """Maximum entropy portfolio — information-theoretic diversification. Maximizes Shannon entropy H = -sum(w*ln(w)) subject to full investment. prices_json: {ticker: [bars]}."""
    import pandas as pd

    from portfolio_advisor.tools.technical_indicators import _prices_to_series

    ticker_list = [t.strip() for t in tickers.split(",")]
    prices_dict = json.loads(prices_json)

    all_returns = {}
    for t in ticker_list:
        if t not in prices_dict:
            continue
        df = _prices_to_series(json.dumps(prices_dict[t]))
        all_returns[t] = df["close"].pct_change().dropna()

    if len(all_returns) < 2:
        return json.dumps({"error": "Need data for at least 2 tickers"})

    ret_df = pd.DataFrame(all_returns).dropna()
    valid_tickers = list(ret_df.columns)

    raw = optimize_entropy_raw(ret_df.values, valid_tickers)
    if "error" in raw:
        return json.dumps({"error": raw["error"]})
    return json.dumps(raw)


# Ticker-specific transaction cost table (Concern #18)
_TICKER_COST_BPS: dict[str, float] = {
    # Large-cap ETFs — very liquid
    "SPY": 2, "QQQ": 2, "IWM": 3, "EFA": 5, "EEM": 10,
    "VNQ": 5, "XLE": 5, "XLK": 5, "XLF": 5,
    "TLT": 3, "IEF": 3, "HYG": 5, "AGG": 3, "BND": 3, "LQD": 5,
    "GLD": 3, "SLV": 5,
    # Large-cap stocks
    "AAPL": 5, "MSFT": 5, "NVDA": 5, "AMZN": 5, "GOOGL": 5, "META": 5, "TSLA": 5,
    # Crypto — higher spread
    "BTC": 25, "ETH": 25, "SOL": 35, "AVAX": 50,
}
_DEFAULT_COST_BPS = 10.0


def _get_ticker_cost_bps(ticker: str) -> float:
    """Get ticker-specific transaction cost in bps."""
    return _TICKER_COST_BPS.get(ticker.upper(), _DEFAULT_COST_BPS)


@function_tool
async def compute_transaction_costs(
    ctx: RunContextWrapper[AppContext],
    current_weights_json: str,
    target_weights_json: str,
    portfolio_value: float = 100000.0,
    cost_bps: float = 0.0,
) -> str:
    """Compute transaction costs for rebalancing with ticker-specific costs. Uses per-ticker cost model (ETFs 2-5bps, stocks 5bps, EM 10bps, crypto 25-50bps). Set cost_bps > 0 to override with flat rate. current/target_weights_json: {ticker: weight_pct}."""
    current = json.loads(current_weights_json)
    target = json.loads(target_weights_json)

    all_tickers = set(list(current.keys()) + list(target.keys()))
    total_turnover = 0.0
    trades = []
    use_flat = cost_bps > 0

    for ticker in sorted(all_tickers):
        cur_w = current.get(ticker, 0.0)
        tgt_w = target.get(ticker, 0.0)
        delta = tgt_w - cur_w

        if abs(delta) > 0.1:  # ignore sub-0.1% changes
            trade_value = abs(delta) / 100.0 * portfolio_value
            ticker_bps = cost_bps if use_flat else _get_ticker_cost_bps(ticker)
            cost = trade_value * ticker_bps / 10000.0
            total_turnover += abs(delta)

            trades.append({
                "ticker": ticker,
                "current_pct": round(cur_w, 2),
                "target_pct": round(tgt_w, 2),
                "delta_pct": round(delta, 2),
                "action": "buy" if delta > 0 else "sell",
                "trade_value": round(trade_value, 2),
                "cost_bps": ticker_bps,
                "estimated_cost": round(cost, 2),
            })

    total_cost = sum(t["estimated_cost"] for t in trades)
    cost_as_pct = total_cost / portfolio_value * 100 if portfolio_value > 0 else 0

    return json.dumps({
        "trades": trades,
        "total_turnover_pct": round(total_turnover, 2),
        "one_way_turnover_pct": round(total_turnover / 2, 2),
        "total_cost": round(total_cost, 2),
        "cost_as_portfolio_pct": round(cost_as_pct, 4),
        "cost_bps": cost_bps,
        "portfolio_value": portfolio_value,
        "n_trades": len(trades),
        "interpretation": (
            f"{len(trades)} trades, {total_turnover:.1f}% total turnover. "
            f"Estimated cost: ${total_cost:.2f} ({cost_as_pct:.3f}% of portfolio). "
            f"{'Low turnover — efficient rebalance.' if total_turnover < 20 else 'Moderate turnover.' if total_turnover < 50 else 'High turnover — consider phasing.'}"
        ),
    })
