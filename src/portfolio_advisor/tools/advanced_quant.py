"""Advanced quantitative models — GARCH, HMM regime detection, Kalman filter, Fama-French."""

from __future__ import annotations

import json
import logging

import numpy as np
from agents import RunContextWrapper, function_tool

from portfolio_advisor.agents.context import AppContext
from portfolio_advisor.tools.technical_indicators import _prices_to_series

logger = logging.getLogger(__name__)


# ── Pure computation functions (no ctx, return dicts) ────────────────────────


def compute_garch_raw(
    returns: np.ndarray,
    model_type: str = "GARCH",
) -> dict:
    """Fit GARCH(1,1) or EGARCH(1,1) to return series — pure function.

    Uses the `arch` library.

    Returns conditional vol forecast (1d/5d/21d), persistence (alpha+beta),
    and half-life of volatility shocks.
    """
    import pandas as pd
    from arch import arch_model

    if len(returns) < 100:
        return {"error": "Insufficient data (need 100+ observations for GARCH)"}

    # Scale returns to percentage for better numerical stability
    # arch library works best with pandas Series
    scaled = pd.Series(returns * 100)

    try:
        if model_type.upper() == "EGARCH":
            am = arch_model(scaled, vol="EGARCH", p=1, o=1, q=1, mean="Constant", rescale=False)
        else:
            am = arch_model(scaled, vol="Garch", p=1, q=1, mean="Constant", rescale=False)

        res = am.fit(disp="off", show_warning=False)

        # Extract parameters
        params = dict(res.params)

        if model_type.upper() == "EGARCH":
            alpha = abs(float(params.get("alpha[1]", 0)))
            beta = float(params.get("beta[1]", 0))
            persistence = alpha + beta
        else:
            alpha = float(params.get("alpha[1]", 0))
            beta = float(params.get("beta[1]", 0))
            persistence = alpha + beta

        # Conditional volatility (last fitted value)
        cond_vol_pct = float(res.conditional_volatility.iloc[-1])
        cond_vol_daily = cond_vol_pct / 100  # Convert back to decimal

        # Forecast — use 1-step analytic, then scale for multi-step
        try:
            forecasts = res.forecast(horizon=1)
            var_1d = float(forecasts.variance.iloc[-1, 0])
        except Exception:
            # Fallback to last conditional variance
            var_1d = cond_vol_pct ** 2

        vol_1d = np.sqrt(var_1d) / 100
        # Scale by sqrt(T) for multi-day forecasts (approximation)
        vol_5d = vol_1d * np.sqrt(5) / np.sqrt(1)
        vol_21d = vol_1d * np.sqrt(21) / np.sqrt(1)

        # Half-life of vol shocks: ln(2) / -ln(persistence)
        half_life = None
        if 0 < persistence < 1:
            half_life = round(np.log(2) / (-np.log(persistence)), 1)

        return {
            "model_type": model_type.upper(),
            "alpha": round(alpha, 4),
            "beta": round(beta, 4),
            "persistence": round(persistence, 4),
            "half_life_days": half_life,
            "current_cond_vol_daily": round(cond_vol_daily, 6),
            "current_cond_vol_annualized": round(cond_vol_daily * np.sqrt(252) * 100, 2),
            "forecast_vol_1d": round(vol_1d * 100, 4),
            "forecast_vol_5d_avg": round(vol_5d * 100, 4),
            "forecast_vol_21d_avg": round(vol_21d * 100, 4),
            "forecast_vol_annualized": round(vol_21d * np.sqrt(252) * 100, 2),
            "aic": round(float(res.aic), 2),
            "bic": round(float(res.bic), 2),
        }

    except Exception as e:
        logger.warning(f"GARCH fitting failed: {e}")
        return {"error": f"GARCH fitting failed: {str(e)[:200]}"}


def detect_regime_hmm_raw(
    returns: np.ndarray,
    n_states: int = 3,
) -> dict:
    """Fit Gaussian HMM to return series for regime detection — pure function.

    Uses `hmmlearn`. 3 states: bull (high mean, low vol), bear (low mean, high vol),
    transition (moderate).

    Returns current state, state probabilities, transition matrix, expected durations.
    """
    try:
        from hmmlearn.hmm import GaussianHMM
    except ImportError:
        return {"error": "hmmlearn not installed (requires Python <3.14 or pre-built wheel)"}

    if len(returns) < 100:
        return {"error": "Insufficient data (need 100+ observations for HMM)"}

    try:
        X = returns.reshape(-1, 1)

        model = GaussianHMM(
            n_components=n_states,
            covariance_type="full",
            n_iter=200,
            random_state=42,
            tol=0.01,
        )
        model.fit(X)

        # Get state sequence and probabilities
        hidden_states = model.predict(X)
        state_probs = model.predict_proba(X)

        # Current state
        current_state = int(hidden_states[-1])
        current_probs = state_probs[-1].tolist()

        # State characteristics (mean and vol)
        state_means = model.means_.flatten()
        raw_vars = model.covars_.flatten()
        # Clamp degenerate variances (numerical issue with few observations in a state)
        max_reasonable_var = np.var(returns) * 100
        clamped_vars = np.clip(raw_vars, 1e-12, max_reasonable_var)
        state_vols = np.sqrt(clamped_vars)

        # Label states by mean return: highest mean = bull, lowest = bear
        state_order = np.argsort(state_means)
        labels = [""] * n_states
        labels[state_order[0]] = "bear"
        labels[state_order[-1]] = "bull"
        for i in state_order[1:-1]:
            labels[i] = "transition"

        # Transition matrix
        trans_matrix = model.transmat_.tolist()

        # Expected duration in each state: 1 / (1 - p_ii)
        expected_durations = {}
        for i in range(n_states):
            p_stay = model.transmat_[i, i]
            duration = 1 / (1 - p_stay) if p_stay < 1 else float("inf")
            expected_durations[labels[i]] = round(duration, 1)

        # State statistics
        state_stats = {}
        for i in range(n_states):
            label = labels[i]
            state_stats[label] = {
                "mean_daily_return_pct": round(float(state_means[i]) * 100, 4),
                "daily_vol_pct": round(float(state_vols[i]) * 100, 4),
                "annualized_return_pct": round(float(state_means[i]) * 252 * 100, 2),
                "annualized_vol_pct": round(float(state_vols[i]) * np.sqrt(252) * 100, 2),
            }

        current_label = labels[current_state]

        return {
            "n_states": n_states,
            "current_state": current_label,
            "current_state_probability": round(float(current_probs[current_state]), 3),
            "state_probabilities": {
                labels[i]: round(float(current_probs[i]), 3) for i in range(n_states)
            },
            "state_stats": state_stats,
            "transition_matrix": {
                labels[i]: {labels[j]: round(trans_matrix[i][j], 3) for j in range(n_states)}
                for i in range(n_states)
            },
            "expected_durations_days": expected_durations,
            "log_likelihood": round(float(model.score(X)), 2),
        }

    except Exception as e:
        logger.warning(f"HMM fitting failed: {e}")
        return {"error": f"HMM fitting failed: {str(e)[:200]}"}


def compute_kalman_beta_raw(
    asset_returns: np.ndarray,
    benchmark_returns: np.ndarray,
) -> dict:
    """Kalman filter for time-varying beta estimation — pure function.

    2-state model: [alpha, beta] follow a random walk.
    Observation: r_asset = alpha + beta * r_benchmark + noise.

    Returns time-varying beta/alpha with confidence intervals and trend.
    """
    n = len(asset_returns)
    if n < 60:
        return {"error": "Insufficient data (need 60+ overlapping observations)"}

    # State: [alpha, beta]
    # Transition: state_t = state_{t-1} + process_noise
    # Observation: y_t = [1, x_t] @ state_t + obs_noise

    # Initialize
    state = np.array([0.0, 1.0])  # alpha=0, beta=1
    P = np.eye(2) * 1.0  # state covariance

    # Noise parameters (tuned for daily financial data)
    Q = np.eye(2) * 1e-5  # process noise (random walk volatility)
    Q[0, 0] = 1e-7  # alpha changes slowly
    Q[1, 1] = 1e-5  # beta changes faster

    obs_var = np.var(asset_returns) * 0.5  # observation noise

    # Store results
    betas = np.zeros(n)
    alphas = np.zeros(n)
    beta_vars = np.zeros(n)
    alpha_vars = np.zeros(n)

    for t in range(n):
        # Predict
        state_pred = state  # random walk: no change
        P_pred = P + Q

        # Update
        H = np.array([1.0, benchmark_returns[t]])  # observation matrix row
        y = asset_returns[t]
        y_pred = H @ state_pred
        innovation = y - y_pred
        S = H @ P_pred @ H + obs_var  # innovation variance
        K = P_pred @ H / S  # Kalman gain

        state = state_pred + K * innovation
        P = (np.eye(2) - np.outer(K, H)) @ P_pred

        betas[t] = state[1]
        alphas[t] = state[0]
        beta_vars[t] = P[1, 1]
        alpha_vars[t] = P[0, 0]

    # Current estimates with confidence intervals
    current_beta = float(betas[-1])
    current_alpha = float(alphas[-1])
    beta_std = float(np.sqrt(beta_vars[-1]))
    alpha_std = float(np.sqrt(alpha_vars[-1]))

    # Beta trend over last 30 days
    lookback = min(30, n)
    beta_30d_start = float(betas[-lookback])
    beta_30d_end = float(betas[-1])
    beta_trend = "stable"
    if beta_30d_end - beta_30d_start > 0.05:
        beta_trend = "increasing"
    elif beta_30d_end - beta_30d_start < -0.05:
        beta_trend = "decreasing"

    return {
        "current_beta": round(current_beta, 4),
        "current_alpha_daily": round(current_alpha, 6),
        "current_alpha_annualized_pct": round(current_alpha * 252 * 100, 2),
        "beta_ci_95": [
            round(current_beta - 1.96 * beta_std, 4),
            round(current_beta + 1.96 * beta_std, 4),
        ],
        "alpha_ci_95": [
            round((current_alpha - 1.96 * alpha_std) * 252 * 100, 2),
            round((current_alpha + 1.96 * alpha_std) * 252 * 100, 2),
        ],
        "beta_30d_start": round(beta_30d_start, 4),
        "beta_30d_end": round(beta_30d_end, 4),
        "beta_trend": beta_trend,
        "beta_range_1y": [round(float(betas.min()), 4), round(float(betas.max()), 4)],
    }


def compute_fama_french_raw(
    asset_returns: np.ndarray,
    market_returns: np.ndarray,
    smb_returns: np.ndarray,
    hml_returns: np.ndarray,
) -> dict:
    """Fama-French 3-factor model regression — pure function.

    r_i - r_f = alpha + beta_mkt * (r_m - r_f) + beta_smb * SMB + beta_hml * HML + epsilon

    Uses ETF proxies: SPY (market), IWM-SPY (size), IWD-IWF (value).
    Risk-free rate approximated as 0 for daily returns.
    """
    n = len(asset_returns)
    if n < 60:
        return {"error": "Insufficient data (need 60+ observations)"}

    # Build design matrix: [1, market, smb, hml]
    X = np.column_stack([
        np.ones(n),
        market_returns,
        smb_returns,
        hml_returns,
    ])

    try:
        coeffs, residuals, rank, sv = np.linalg.lstsq(X, asset_returns, rcond=None)

        alpha = float(coeffs[0])
        beta_mkt = float(coeffs[1])
        beta_smb = float(coeffs[2])
        beta_hml = float(coeffs[3])

        # R-squared
        y_pred = X @ coeffs
        ss_res = np.sum((asset_returns - y_pred) ** 2)
        ss_tot = np.sum((asset_returns - asset_returns.mean()) ** 2)
        r_squared = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

        # Residual standard error
        dof = n - 4
        resid_se = np.sqrt(ss_res / dof) if dof > 0 else 0.0

        # Standard errors of coefficients
        try:
            XtX_inv = np.linalg.inv(X.T @ X)
            coeff_se = np.sqrt(np.diag(XtX_inv) * (ss_res / dof))
            t_stats = coeffs / coeff_se
        except np.linalg.LinAlgError:
            coeff_se = np.zeros(4)
            t_stats = np.zeros(4)

        # Interpret factor exposures
        style = []
        if beta_smb > 0.2:
            style.append("small_cap_tilt")
        elif beta_smb < -0.2:
            style.append("large_cap_tilt")
        if beta_hml > 0.2:
            style.append("value_tilt")
        elif beta_hml < -0.2:
            style.append("growth_tilt")
        if not style:
            style.append("blend")

        return {
            "alpha_daily": round(alpha, 6),
            "alpha_annualized_pct": round(alpha * 252 * 100, 2),
            "beta_market": round(beta_mkt, 4),
            "beta_smb": round(beta_smb, 4),
            "beta_hml": round(beta_hml, 4),
            "r_squared": round(r_squared, 4),
            "adj_r_squared": round(1 - (1 - r_squared) * (n - 1) / (n - 4), 4),
            "t_stats": {
                "alpha": round(float(t_stats[0]), 2),
                "market": round(float(t_stats[1]), 2),
                "smb": round(float(t_stats[2]), 2),
                "hml": round(float(t_stats[3]), 2),
            },
            "residual_std_daily": round(float(resid_se), 6),
            "style": style,
        }

    except Exception as e:
        logger.warning(f"Fama-French regression failed: {e}")
        return {"error": f"Fama-French regression failed: {str(e)[:200]}"}


# ── @function_tool wrappers ──────────────────────────────────────────────────


@function_tool
async def compute_garch_volatility(
    ctx: RunContextWrapper[AppContext],
    ticker: str,
    prices_json: str,
    model_type: str = "GARCH",
) -> str:
    """Fit GARCH(1,1) or EGARCH(1,1) volatility model. Returns conditional vol forecast (1d/5d/21d), persistence (alpha+beta), and half-life of vol shocks."""
    df = _prices_to_series(prices_json)
    returns = df["close"].pct_change().dropna().values
    raw = compute_garch_raw(returns, model_type)
    if "error" in raw:
        return json.dumps({"ticker": ticker, "error": raw["error"]})
    raw["ticker"] = ticker
    return json.dumps(raw)


@function_tool
async def detect_regime_hmm(
    ctx: RunContextWrapper[AppContext],
    ticker: str,
    prices_json: str,
    n_states: int = 3,
) -> str:
    """Detect market regime using Hidden Markov Model (3 states: bull/bear/transition). Returns current state, probabilities, transition matrix, expected durations."""
    df = _prices_to_series(prices_json)
    returns = df["close"].pct_change().dropna().values
    raw = detect_regime_hmm_raw(returns, n_states)
    if "error" in raw:
        return json.dumps({"ticker": ticker, "error": raw["error"]})
    raw["ticker"] = ticker
    return json.dumps(raw)


@function_tool
async def compute_kalman_filter(
    ctx: RunContextWrapper[AppContext],
    ticker: str,
    prices_json: str,
    benchmark: str = "SPY",
) -> str:
    """Compute time-varying beta via Kalman filter (adaptive beta that tracks regime changes). Returns current beta/alpha with 95% CIs and 30-day trend."""
    import yfinance as yf

    df = _prices_to_series(prices_json)
    asset_returns = df["close"].pct_change().dropna()

    bench_df = yf.download(benchmark, period="1y", progress=False)
    if bench_df.empty:
        return json.dumps({"ticker": ticker, "error": f"Could not fetch {benchmark} data"})

    bench_returns = bench_df["Close"].pct_change().dropna()

    # Align dates
    common = asset_returns.index.intersection(bench_returns.index)
    if len(common) < 60:
        return json.dumps({"ticker": ticker, "error": "Insufficient overlapping data"})

    raw = compute_kalman_beta_raw(
        asset_returns.loc[common].values,
        bench_returns.loc[common].values,
    )
    if "error" in raw:
        return json.dumps({"ticker": ticker, "error": raw["error"]})
    raw["ticker"] = ticker
    raw["benchmark"] = benchmark
    return json.dumps(raw)


@function_tool
async def compute_fama_french_3factor(
    ctx: RunContextWrapper[AppContext],
    ticker: str,
    prices_json: str,
) -> str:
    """Fama-French 3-factor model: market, size (SMB), value (HML) factor betas. Uses ETF proxies (SPY, IWM, IWD, IWF). Returns factor betas, alpha, R-squared, and style classification."""
    import yfinance as yf

    df = _prices_to_series(prices_json)
    asset_returns = df["close"].pct_change().dropna()

    # Fetch factor proxy ETFs
    etfs = ["SPY", "IWM", "IWD", "IWF"]
    etf_df = yf.download(etfs, period="1y", group_by="ticker", progress=False)
    if etf_df.empty:
        return json.dumps({"ticker": ticker, "error": "Could not fetch factor proxy ETFs"})

    try:
        spy_ret = etf_df["SPY"]["Close"].pct_change().dropna()
        iwm_ret = etf_df["IWM"]["Close"].pct_change().dropna()
        iwd_ret = etf_df["IWD"]["Close"].pct_change().dropna()
        iwf_ret = etf_df["IWF"]["Close"].pct_change().dropna()
    except KeyError as e:
        return json.dumps({"ticker": ticker, "error": f"Missing ETF data: {e}"})

    # Construct factor returns
    # SMB = IWM - SPY (small minus big proxy)
    # HML = IWD - IWF (value minus growth proxy)
    common = (
        asset_returns.index
        .intersection(spy_ret.index)
        .intersection(iwm_ret.index)
        .intersection(iwd_ret.index)
        .intersection(iwf_ret.index)
    )

    if len(common) < 60:
        return json.dumps({"ticker": ticker, "error": "Insufficient overlapping data"})

    market = spy_ret.loc[common].values
    smb = (iwm_ret.loc[common] - spy_ret.loc[common]).values
    hml = (iwd_ret.loc[common] - iwf_ret.loc[common]).values
    asset = asset_returns.loc[common].values

    raw = compute_fama_french_raw(asset, market, smb, hml)
    if "error" in raw:
        return json.dumps({"ticker": ticker, "error": raw["error"]})
    raw["ticker"] = ticker
    raw["factor_proxies"] = {"market": "SPY", "smb": "IWM-SPY", "hml": "IWD-IWF"}
    return json.dumps(raw)
