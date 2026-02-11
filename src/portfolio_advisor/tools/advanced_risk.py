"""Advanced risk metrics — Cornish-Fisher VaR, EVT, Monte Carlo VaR, stress testing, tail dependence."""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
from agents import RunContextWrapper, function_tool

from portfolio_advisor.agents.context import AppContext
from portfolio_advisor.tools.risk_metrics import _build_portfolio_returns
from portfolio_advisor.tools.technical_indicators import _prices_to_series

logger = logging.getLogger(__name__)


# ── Pure computation functions ────────────────────────────────────────────────


def compute_cornish_fisher_raw(
    returns: np.ndarray,
    confidence: float = 0.95,
) -> dict:
    """Cornish-Fisher VaR: skewness/kurtosis-adjusted quantile.

    z_cf = z + (z^2 - 1)*S/6 + (z^3 - 3z)*K/24 - (2z^3 - 5z)*S^2/36

    More accurate than Gaussian VaR when returns have fat tails or skew.
    """
    from scipy.stats import norm

    n = len(returns)
    if n < 30:
        return {"error": "Insufficient data (need 30+ observations)"}

    mu = float(np.mean(returns))
    sigma = float(np.std(returns))
    skew = float(pd.Series(returns).skew())
    kurt = float(pd.Series(returns).kurtosis())  # excess kurtosis

    z = norm.ppf(1 - confidence)  # e.g., -1.645 for 95%

    # Cornish-Fisher expansion
    z_cf = (
        z
        + (z**2 - 1) * skew / 6
        + (z**3 - 3 * z) * kurt / 24
        - (2 * z**3 - 5 * z) * skew**2 / 36
    )

    cf_var = mu + sigma * z_cf
    gaussian_var = mu + sigma * z

    # Historical VaR for comparison
    hist_var = float(np.percentile(returns, (1 - confidence) * 100))

    return {
        "confidence": confidence,
        "cornish_fisher_var_pct": round(cf_var * 100, 4),
        "gaussian_var_pct": round(gaussian_var * 100, 4),
        "historical_var_pct": round(hist_var * 100, 4),
        "cf_adjustment_pct": round((cf_var - gaussian_var) * 100, 4),
        "skewness": round(skew, 3),
        "excess_kurtosis": round(kurt, 3),
        "z_gaussian": round(float(z), 4),
        "z_cornish_fisher": round(float(z_cf), 4),
        "interpretation": (
            f"CF VaR ({confidence*100:.0f}%): {cf_var*100:.3f}% daily "
            f"(Gaussian: {gaussian_var*100:.3f}%, Historical: {hist_var*100:.3f}%). "
            f"{'CF more conservative (accounts for left skew/fat tails).' if cf_var < gaussian_var else 'CF less conservative than Gaussian.'} "
            f"Skew={skew:.2f}, Excess Kurtosis={kurt:.2f}."
        ),
        "observations": n,
    }


def compute_evt_var_raw(
    returns: np.ndarray,
    confidence: float = 0.99,
    threshold_pct: float = 10.0,
) -> dict:
    """Extreme Value Theory VaR using Generalized Pareto Distribution.

    Fit GPD to tail losses exceeding a high threshold.
    Shape xi > 0 = heavy tails (Frechet). xi = 0 = exponential tails.
    """
    from scipy.stats import genpareto

    n = len(returns)
    if n < 100:
        return {"error": "Insufficient data (need 100+ observations for EVT)"}

    # Use losses (negative returns)
    losses = -returns

    # Threshold: top threshold_pct% of losses
    threshold = float(np.percentile(losses, 100 - threshold_pct))
    exceedances = losses[losses > threshold] - threshold
    n_exceed = len(exceedances)

    if n_exceed < 20:
        return {"error": f"Too few exceedances ({n_exceed}) — lower threshold or use more data"}

    # Fit GPD
    try:
        shape, loc, scale = genpareto.fit(exceedances, floc=0)
    except Exception as e:
        return {"error": f"GPD fitting failed: {str(e)[:200]}"}

    # EVT VaR
    p = 1 - confidence
    n_ratio = n_exceed / n

    if abs(shape) < 1e-8:
        # Exponential case
        evt_var = threshold + scale * np.log(n_ratio / p)
    else:
        evt_var = threshold + (scale / shape) * ((n_ratio / p) ** shape - 1)

    # EVT ES (Expected Shortfall)
    if shape < 1:
        evt_es = evt_var / (1 - shape) + (scale - shape * threshold) / (1 - shape)
    else:
        evt_es = evt_var * 1.5  # rough approximation when shape >= 1

    # Historical comparison
    hist_var = float(np.percentile(losses, confidence * 100))

    return {
        "confidence": confidence,
        "evt_var_pct": round(float(evt_var) * 100, 4),
        "evt_es_pct": round(float(evt_es) * 100, 4),
        "historical_var_pct": round(hist_var * 100, 4),
        "gpd_shape": round(float(shape), 4),
        "gpd_scale": round(float(scale), 6),
        "threshold_pct": round(threshold * 100, 4),
        "n_exceedances": n_exceed,
        "tail_type": "heavy" if shape > 0.1 else "light" if shape < -0.1 else "exponential",
        "interpretation": (
            f"EVT VaR ({confidence*100:.0f}%): {evt_var*100:.3f}% daily "
            f"(Historical: {hist_var*100:.3f}%). "
            f"GPD shape={shape:.3f} — "
            f"{'heavy tails (risk higher than normal)' if shape > 0.1 else 'light tails' if shape < -0.1 else 'near-exponential tails'}. "
            f"EVT ES: {evt_es*100:.3f}%."
        ),
        "observations": n,
    }


def compute_mc_var_raw(
    returns: np.ndarray,
    confidence: float = 0.95,
    n_sims: int = 10000,
    horizon_days: int = 1,
    distribution: str = "normal",
) -> dict:
    """Monte Carlo VaR via simulated return paths.

    Supports normal and Student-t distributions.
    """
    from scipy.stats import t as t_dist

    n = len(returns)
    if n < 30:
        return {"error": "Insufficient data (need 30+ observations)"}

    mu = float(np.mean(returns))
    sigma = float(np.std(returns))

    rng = np.random.default_rng(42)

    if distribution == "student_t":
        # Estimate degrees of freedom from excess kurtosis
        kurt = float(pd.Series(returns).kurtosis())
        # kurtosis = 6/(nu-4) for t-dist with nu > 4
        if kurt > 0:
            nu = max(5, 6 / kurt + 4)
        else:
            nu = 30  # near-normal
        sims = t_dist.rvs(nu, loc=mu, scale=sigma * np.sqrt((nu - 2) / nu), size=(n_sims, horizon_days), random_state=rng)
    else:
        sims = rng.normal(mu, sigma, size=(n_sims, horizon_days))

    # Cumulative returns over horizon
    if horizon_days > 1:
        cum_returns = np.prod(1 + sims, axis=1) - 1
    else:
        cum_returns = sims[:, 0]

    mc_var = float(np.percentile(cum_returns, (1 - confidence) * 100))
    mc_es = float(cum_returns[cum_returns <= mc_var].mean()) if np.any(cum_returns <= mc_var) else mc_var

    # Historical comparison
    if horizon_days == 1:
        hist_var = float(np.percentile(returns, (1 - confidence) * 100))
    else:
        # Rolling multi-day returns
        rolling = np.array([
            np.prod(1 + returns[i : i + horizon_days]) - 1
            for i in range(len(returns) - horizon_days)
        ])
        hist_var = float(np.percentile(rolling, (1 - confidence) * 100)) if len(rolling) > 0 else mc_var

    return {
        "confidence": confidence,
        "horizon_days": horizon_days,
        "distribution": distribution,
        "mc_var_pct": round(mc_var * 100, 4),
        "mc_es_pct": round(mc_es * 100, 4),
        "historical_var_pct": round(hist_var * 100, 4),
        "n_simulations": n_sims,
        "sim_mean_pct": round(float(np.mean(cum_returns)) * 100, 4),
        "sim_std_pct": round(float(np.std(cum_returns)) * 100, 4),
        "worst_sim_pct": round(float(np.min(cum_returns)) * 100, 4),
        "best_sim_pct": round(float(np.max(cum_returns)) * 100, 4),
        "interpretation": (
            f"MC VaR ({confidence*100:.0f}%, {horizon_days}d, {distribution}): "
            f"{mc_var*100:.3f}%. ES: {mc_es*100:.3f}%. "
            f"Historical VaR: {hist_var*100:.3f}%. "
            f"{'MC more conservative.' if mc_var < hist_var else 'Historical more conservative.'}"
        ),
        "observations": n,
    }


def run_stress_test_raw(
    returns_matrix: np.ndarray,
    tickers: list[str],
    weights: np.ndarray,
    spy_returns: np.ndarray | None = None,
) -> dict:
    """Historical stress scenarios applied via factor shocks.

    Scenarios: 2008 GFC, 2020 COVID, 2022 Rate Hike, Custom.
    """
    # Define scenarios as factor shocks (market, bond proxy, commodity proxy)
    scenarios = {
        "2008_gfc": {
            "description": "2008 Global Financial Crisis (6-month drawdown)",
            "market_shock": -0.38,
            "bond_shock": 0.20,
            "vol_multiplier": 3.0,
        },
        "2020_covid": {
            "description": "COVID crash (1-month drawdown, Mar 2020)",
            "market_shock": -0.34,
            "bond_shock": 0.08,
            "vol_multiplier": 4.0,
        },
        "2022_rates": {
            "description": "2022 Rate hiking cycle (YTD drawdown)",
            "market_shock": -0.19,
            "bond_shock": -0.31,
            "vol_multiplier": 1.5,
        },
        "flash_crash": {
            "description": "Single-day flash crash",
            "market_shock": -0.07,
            "bond_shock": 0.02,
            "vol_multiplier": 5.0,
        },
    }

    # Estimate asset betas (to market proxy)
    betas = np.ones(len(tickers))
    if spy_returns is not None and len(spy_returns) >= 30:
        min_len = min(len(spy_returns), returns_matrix.shape[0])
        for i in range(len(tickers)):
            asset_r = returns_matrix[-min_len:, i]
            spy_r = spy_returns[-min_len:]
            cov = np.cov(asset_r, spy_r)
            if cov[1, 1] > 0:
                betas[i] = cov[0, 1] / cov[1, 1]

    # Classify tickers (simple heuristic)
    bond_set = {"TLT", "IEF", "HYG", "AGG", "BND", "LQD"}
    is_bond = np.array([1.0 if t in bond_set else 0.0 for t in tickers])

    results = {}
    for scenario_name, params in scenarios.items():
        # Compute per-asset impact
        asset_impacts = []
        for i, ticker in enumerate(tickers):
            if is_bond[i]:
                impact = params["bond_shock"]
            else:
                impact = params["market_shock"] * betas[i]
            asset_impacts.append(impact)

        asset_impacts = np.array(asset_impacts)

        # Portfolio impact
        portfolio_impact = float(weights @ asset_impacts)

        results[scenario_name] = {
            "description": params["description"],
            "portfolio_impact_pct": round(portfolio_impact * 100, 2),
            "asset_impacts": {
                t: round(float(asset_impacts[i]) * 100, 2) for i, t in enumerate(tickers)
            },
            "vol_multiplier": params["vol_multiplier"],
            "severity": (
                "severe" if abs(portfolio_impact) > 0.20 else
                "significant" if abs(portfolio_impact) > 0.10 else
                "moderate" if abs(portfolio_impact) > 0.05 else
                "mild"
            ),
        }

    # Worst-case scenario
    worst = min(results.values(), key=lambda x: x["portfolio_impact_pct"])

    return {
        "scenarios": results,
        "worst_case": {
            "scenario": [k for k, v in results.items() if v == worst][0],
            "impact_pct": worst["portfolio_impact_pct"],
        },
        "portfolio_weights_pct": {t: round(float(w) * 100, 2) for t, w in zip(tickers, weights)},
        "asset_betas": {t: round(float(b), 3) for t, b in zip(tickers, betas)},
    }


def compute_tail_dependence_raw(
    returns_matrix: np.ndarray,
    tickers: list[str],
    quantile: float = 0.05,
) -> dict:
    """Empirical tail dependence: probability of joint extreme losses.

    lambda_L = P(U < q | V < q) as q -> 0.
    Measures co-crash risk beyond what correlation captures.
    """
    n_obs, n_assets = returns_matrix.shape
    if n_obs < 100:
        return {"error": "Insufficient data (need 100+ observations)"}

    pairs = []
    for i in range(n_assets):
        for j in range(i + 1, n_assets):
            x = returns_matrix[:, i]
            y = returns_matrix[:, j]

            # Lower tail dependence (both assets crash)
            x_thresh = np.percentile(x, quantile * 100)
            y_thresh = np.percentile(y, quantile * 100)

            both_crash = np.sum((x <= x_thresh) & (y <= y_thresh))
            x_crash = np.sum(x <= x_thresh)

            tail_dep = both_crash / x_crash if x_crash > 0 else 0.0

            # Upper tail dependence (both assets rally)
            x_thresh_up = np.percentile(x, (1 - quantile) * 100)
            y_thresh_up = np.percentile(y, (1 - quantile) * 100)

            both_rally = np.sum((x >= x_thresh_up) & (y >= y_thresh_up))
            x_rally = np.sum(x >= x_thresh_up)

            upper_dep = both_rally / x_rally if x_rally > 0 else 0.0

            # Linear correlation for comparison
            corr = float(np.corrcoef(x, y)[0, 1])

            # Under independence, expected joint = quantile
            is_significant = tail_dep > quantile * 2

            pairs.append({
                "pair": f"{tickers[i]}/{tickers[j]}",
                "lower_tail_dependence": round(float(tail_dep), 4),
                "upper_tail_dependence": round(float(upper_dep), 4),
                "linear_correlation": round(corr, 3),
                "joint_crash_count": int(both_crash),
                "is_significant": is_significant,
                "co_crash_risk": (
                    "high" if tail_dep > 0.5 else
                    "moderate" if tail_dep > 0.2 else
                    "low"
                ),
            })

    # Sort by lower tail dependence
    pairs.sort(key=lambda x: x["lower_tail_dependence"], reverse=True)

    return {
        "quantile": quantile,
        "pairs": pairs,
        "high_co_crash_pairs": [p["pair"] for p in pairs if p["co_crash_risk"] in ("high", "moderate")],
        "interpretation": (
            f"{sum(1 for p in pairs if p['is_significant'])} pair(s) with significant "
            f"tail dependence (co-crash risk). "
            f"{'Diversification may fail in crashes.' if any(p['lower_tail_dependence'] > 0.3 for p in pairs) else 'Tail diversification appears adequate.'}"
        ),
        "observations": n_obs,
    }


# ── @function_tool wrappers ──────────────────────────────────────────────────


@function_tool
async def compute_cornish_fisher_var(
    ctx: RunContextWrapper[AppContext],
    portfolio_weights_json: str,
    prices_json: str,
    confidence: float = 0.95,
) -> str:
    """Cornish-Fisher VaR: skewness and kurtosis-adjusted Value-at-Risk. More accurate than Gaussian VaR for non-normal returns. weights_json: {ticker: weight}, prices_json: {ticker: [bars]}."""
    weights = json.loads(portfolio_weights_json)
    prices_dict = json.loads(prices_json)

    port_returns = _build_portfolio_returns(weights, prices_dict)
    if len(port_returns) < 30:
        return json.dumps({"error": "Insufficient data"})

    raw = compute_cornish_fisher_raw(port_returns.values, confidence)
    if "error" in raw:
        return json.dumps({"error": raw["error"]})
    return json.dumps(raw)


@function_tool
async def compute_evt_var(
    ctx: RunContextWrapper[AppContext],
    portfolio_weights_json: str,
    prices_json: str,
    confidence: float = 0.99,
) -> str:
    """Extreme Value Theory VaR using Generalized Pareto Distribution. Fits GPD to tail losses for extreme quantile estimation. Best for high confidence levels (99%+). weights_json: {ticker: weight}, prices_json: {ticker: [bars]}."""
    weights = json.loads(portfolio_weights_json)
    prices_dict = json.loads(prices_json)

    port_returns = _build_portfolio_returns(weights, prices_dict)
    if len(port_returns) < 100:
        return json.dumps({"error": "Insufficient data (need 100+ observations for EVT)"})

    raw = compute_evt_var_raw(port_returns.values, confidence)
    if "error" in raw:
        return json.dumps({"error": raw["error"]})
    return json.dumps(raw)


@function_tool
async def compute_monte_carlo_var(
    ctx: RunContextWrapper[AppContext],
    portfolio_weights_json: str,
    prices_json: str,
    confidence: float = 0.95,
    n_sims: int = 10000,
    horizon_days: int = 1,
    distribution: str = "normal",
) -> str:
    """Monte Carlo VaR via simulated return paths. Supports normal and Student-t distributions. Forward-looking risk estimate. weights_json: {ticker: weight}, prices_json: {ticker: [bars]}."""
    weights = json.loads(portfolio_weights_json)
    prices_dict = json.loads(prices_json)

    port_returns = _build_portfolio_returns(weights, prices_dict)
    if len(port_returns) < 30:
        return json.dumps({"error": "Insufficient data"})

    raw = compute_mc_var_raw(port_returns.values, confidence, n_sims, horizon_days, distribution)
    if "error" in raw:
        return json.dumps({"error": raw["error"]})
    return json.dumps(raw)


@function_tool
async def run_stress_test(
    ctx: RunContextWrapper[AppContext],
    portfolio_weights_json: str,
    prices_json: str,
) -> str:
    """Run historical stress scenarios (2008 GFC, COVID, 2022 rates, flash crash). Estimates portfolio impact using factor betas. weights_json: {ticker: weight_pct}, prices_json: {ticker: [bars]}."""
    import yfinance as yf

    weights_dict = json.loads(portfolio_weights_json)
    prices_dict = json.loads(prices_json)

    tickers = list(weights_dict.keys())

    # Build returns matrix
    all_returns = {}
    for t in tickers:
        if t not in prices_dict:
            continue
        df = _prices_to_series(json.dumps(prices_dict[t]))
        all_returns[t] = df["close"].pct_change().dropna()

    if not all_returns:
        return json.dumps({"error": "No price data available"})

    ret_df = pd.DataFrame(all_returns).dropna()
    valid_tickers = list(ret_df.columns)
    valid_w = np.array([weights_dict.get(t, 0) / 100.0 for t in valid_tickers])
    if valid_w.sum() > 0:
        valid_w = valid_w / valid_w.sum()

    # Get SPY for beta calculation
    spy_returns = None
    try:
        spy = yf.download("SPY", period="1y", progress=False)
        if not spy.empty:
            spy_ret = spy["Close"].pct_change().dropna()
            common = ret_df.index.intersection(spy_ret.index)
            if len(common) >= 30:
                spy_returns = spy_ret.loc[common].values
                ret_df = ret_df.loc[common]
    except Exception:
        pass

    raw = run_stress_test_raw(ret_df.values, valid_tickers, valid_w, spy_returns)
    return json.dumps(raw)


@function_tool
async def compute_tail_dependence(
    ctx: RunContextWrapper[AppContext],
    tickers: str,
    prices_json: str,
    quantile: float = 0.05,
) -> str:
    """Compute empirical tail dependence (co-crash risk) between asset pairs. Measures probability of simultaneous extreme losses beyond what correlation captures. prices_json: {ticker: [bars]}."""
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

    raw = compute_tail_dependence_raw(ret_df.values, valid_tickers, quantile)
    if "error" in raw:
        return json.dumps({"error": raw["error"]})
    return json.dumps(raw)
