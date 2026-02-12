"""Live data validation tests for portfolio optimization models.

These tests fetch real market data from yfinance and validate that
the portfolio optimization functions produce mathematically valid results.
"""

from __future__ import annotations

import numpy as np
import pytest
import yfinance as yf

from portfolio_advisor.tools.advanced_portfolio import (
    _get_ticker_cost_bps,
    optimize_cvar_raw,
    optimize_entropy_raw,
    optimize_hrp_raw,
    optimize_max_diversification_raw,
    compute_kelly_raw,
)


@pytest.fixture(scope="module")
def live_returns():
    """Fetch 1Y returns for 5 assets."""
    tickers = ["SPY", "AAPL", "MSFT", "GLD", "TLT"]
    df = yf.download(tickers, period="1y", progress=False)
    # yfinance multi-ticker download uses MultiIndex columns
    close = df["Close"]
    returns = close.pct_change().dropna()
    return returns.values, tickers


# ── CVaR Optimization ────────────────────────────────────────────────────────


@pytest.mark.live
def test_cvar_optimization_live(live_returns):
    """CVaR optimization produces valid weights and negative CVaR."""
    returns_matrix, tickers = live_returns
    result = optimize_cvar_raw(returns_matrix, tickers)

    assert "error" not in result, f"CVaR optimization failed: {result}"
    assert result["method"] == "cvar_optimization"

    # Weights sum to ~1.0
    alloc = result["allocations_pct"]
    weight_sum = sum(alloc.values())
    assert abs(weight_sum - 100.0) < 1.0, f"Weights sum to {weight_sum}, expected ~100"

    # All weights >= 0
    for ticker, w in alloc.items():
        assert w >= 0, f"Negative weight for {ticker}: {w}"

    # CVaR should be negative (it's a loss metric)
    assert result["cvar_daily_pct"] < 0, f"CVaR should be negative, got {result['cvar_daily_pct']}"

    # VaR should also be negative
    assert result["var_daily_pct"] < 0, f"VaR should be negative, got {result['var_daily_pct']}"

    # CVaR <= VaR (CVaR is the expected loss beyond VaR, so more negative)
    assert result["cvar_daily_pct"] <= result["var_daily_pct"], (
        f"CVaR ({result['cvar_daily_pct']}) should be <= VaR ({result['var_daily_pct']})"
    )


# ── HRP Optimization ─────────────────────────────────────────────────────────


@pytest.mark.live
def test_hrp_optimization_live(live_returns):
    """HRP optimization produces diversified positive weights."""
    returns_matrix, tickers = live_returns
    result = optimize_hrp_raw(returns_matrix, tickers)

    assert "error" not in result, f"HRP optimization failed: {result}"
    assert result["method"] == "hierarchical_risk_parity"

    alloc = result["allocations_pct"]
    weight_sum = sum(alloc.values())
    assert abs(weight_sum - 100.0) < 1.0, f"Weights sum to {weight_sum}, expected ~100"

    # HRP allocates to all assets (diversified) - all should be > 0
    for ticker, w in alloc.items():
        assert w > 0, f"HRP should allocate positively to all assets, {ticker} got {w}"

    # Cluster order should contain all tickers
    assert set(result["cluster_order"]) == set(tickers)

    # Portfolio vol should be positive and finite
    assert result["portfolio_vol_annualized_pct"] > 0
    assert np.isfinite(result["portfolio_vol_annualized_pct"])


# ── Kelly Criterion ───────────────────────────────────────────────────────────


@pytest.mark.live
def test_kelly_criterion_live(live_returns):
    """Kelly criterion produces valid full and half Kelly allocations."""
    returns_matrix, tickers = live_returns

    # Compute mean returns and covariance from returns matrix
    mu = returns_matrix.mean(axis=0)
    cov = np.cov(returns_matrix.T)

    result = compute_kelly_raw(mu, cov, tickers)

    assert "error" not in result, f"Kelly criterion failed: {result}"
    assert result["method"] == "kelly_criterion"

    # Full Kelly allocations should be present for all tickers
    full_kelly = result["full_kelly_pct"]
    assert len(full_kelly) == len(tickers)

    # Half Kelly allocations should be present for all tickers
    half_kelly = result["half_kelly_pct"]
    assert len(half_kelly) == len(tickers)

    # Full Kelly raw values should be floats
    full_raw = result["full_kelly_raw"]
    for ticker, val in full_raw.items():
        assert isinstance(val, float), f"full_kelly_raw for {ticker} is not float: {type(val)}"

    # Normalized allocations should sum to ~100%
    full_sum = sum(full_kelly.values())
    assert abs(full_sum - 100.0) < 1.0, f"Full Kelly weights sum to {full_sum}, expected ~100"

    half_sum = sum(half_kelly.values())
    assert abs(half_sum - 100.0) < 1.0, f"Half Kelly weights sum to {half_sum}, expected ~100"

    # Recommendation should be half_kelly
    assert result["recommendation"] == "half_kelly"


# ── Max Diversification ──────────────────────────────────────────────────────


@pytest.mark.live
def test_max_diversification_live(live_returns):
    """Max diversification produces valid weights with diversification_ratio > 1."""
    returns_matrix, tickers = live_returns
    result = optimize_max_diversification_raw(returns_matrix, tickers)

    assert "error" not in result, f"Max diversification failed: {result}"
    assert result["method"] == "max_diversification"

    alloc = result["allocations_pct"]
    weight_sum = sum(alloc.values())
    assert abs(weight_sum - 100.0) < 1.0, f"Weights sum to {weight_sum}, expected ~100"

    # All weights >= 0
    for ticker, w in alloc.items():
        assert w >= 0, f"Negative weight for {ticker}: {w}"

    # Diversification ratio should be >= 1
    # (weighted avg vol / portfolio vol -- portfolio effect always helps)
    assert result["diversification_ratio"] >= 1.0, (
        f"Diversification ratio {result['diversification_ratio']} < 1.0"
    )

    # Portfolio vol and weighted avg vol should be positive
    assert result["portfolio_vol_annualized_pct"] > 0
    assert result["weighted_avg_vol_pct"] > 0

    # Diversification benefit should be >= 0
    assert result["diversification_benefit_pct"] >= 0


# ── Max Entropy ───────────────────────────────────────────────────────────────


@pytest.mark.live
def test_max_entropy_live(live_returns):
    """Max entropy optimization produces valid weights with positive entropy."""
    returns_matrix, tickers = live_returns
    result = optimize_entropy_raw(returns_matrix, tickers)

    assert "error" not in result, f"Max entropy failed: {result}"
    assert result["method"] == "max_entropy"

    alloc = result["allocations_pct"]
    weight_sum = sum(alloc.values())
    assert abs(weight_sum - 100.0) < 1.0, f"Weights sum to {weight_sum}, expected ~100"

    # All weights >= 0
    for ticker, w in alloc.items():
        assert w >= 0, f"Negative weight for {ticker}: {w}"

    # Shannon entropy should be positive
    assert result["entropy"] > 0, f"Entropy should be positive, got {result['entropy']}"

    # Max entropy = ln(n_assets)
    expected_max = float(np.log(len(tickers)))
    assert abs(result["max_entropy"] - expected_max) < 0.01

    # Normalized entropy should be between 0 and 1
    assert 0 < result["normalized_entropy"] <= 1.0

    # Portfolio vol should be positive
    assert result["portfolio_vol_annualized_pct"] > 0


# ── Transaction Cost Model ───────────────────────────────────────────────────


@pytest.mark.live
def test_transaction_cost_model():
    """Transaction cost model returns positive costs for known tickers."""
    # Test that the cost lookup works for all 5 tickers
    tickers = ["SPY", "AAPL", "MSFT", "GLD", "TLT"]
    for ticker in tickers:
        cost = _get_ticker_cost_bps(ticker)
        assert cost > 0, f"Cost for {ticker} should be > 0, got {cost}"
        assert cost < 100, f"Cost for {ticker} should be < 100 bps, got {cost}"

    # SPY (most liquid ETF) should have lower cost than AAPL (stock)
    assert _get_ticker_cost_bps("SPY") <= _get_ticker_cost_bps("AAPL")


# ── Two-Asset Edge Case ──────────────────────────────────────────────────────


@pytest.mark.live
def test_portfolio_optimizers_2_assets():
    """All optimizers produce valid results with just 2 assets (SPY + GLD)."""
    tickers = ["SPY", "GLD"]
    df = yf.download(tickers, period="1y", progress=False)
    close = df["Close"]
    returns = close.pct_change().dropna()
    returns_matrix = returns.values

    # CVaR
    cvar_result = optimize_cvar_raw(returns_matrix, tickers)
    assert "error" not in cvar_result, f"CVaR failed for 2 assets: {cvar_result}"
    cvar_sum = sum(cvar_result["allocations_pct"].values())
    assert abs(cvar_sum - 100.0) < 1.0

    # HRP
    hrp_result = optimize_hrp_raw(returns_matrix, tickers)
    assert "error" not in hrp_result, f"HRP failed for 2 assets: {hrp_result}"
    hrp_sum = sum(hrp_result["allocations_pct"].values())
    assert abs(hrp_sum - 100.0) < 1.0

    # Max Diversification
    div_result = optimize_max_diversification_raw(returns_matrix, tickers)
    assert "error" not in div_result, f"Max div failed for 2 assets: {div_result}"
    div_sum = sum(div_result["allocations_pct"].values())
    assert abs(div_sum - 100.0) < 1.0

    # Max Entropy
    ent_result = optimize_entropy_raw(returns_matrix, tickers)
    assert "error" not in ent_result, f"Max entropy failed for 2 assets: {ent_result}"
    ent_sum = sum(ent_result["allocations_pct"].values())
    assert abs(ent_sum - 100.0) < 1.0

    # Kelly
    mu = returns_matrix.mean(axis=0)
    cov = np.cov(returns_matrix.T)
    kelly_result = compute_kelly_raw(mu, cov, tickers)
    assert "error" not in kelly_result, f"Kelly failed for 2 assets: {kelly_result}"
    kelly_sum = sum(kelly_result["full_kelly_pct"].values())
    assert abs(kelly_sum - 100.0) < 1.0


# ── Singular / Highly Correlated Covariance ──────────────────────────────────


@pytest.mark.live
def test_portfolio_optimizers_singular_cov():
    """Optimizers handle nearly-perfectly-correlated series gracefully."""
    np.random.seed(42)
    n_obs = 252

    # Create two nearly identical series (correlation ~1.0)
    base_returns = np.random.normal(0.0005, 0.01, n_obs)
    noise = np.random.normal(0, 0.0001, n_obs)
    returns_matrix = np.column_stack([base_returns, base_returns + noise])
    tickers = ["A", "B"]

    # CVaR should either succeed or return an error dict (no crash)
    cvar_result = optimize_cvar_raw(returns_matrix, tickers)
    if "error" not in cvar_result:
        cvar_sum = sum(cvar_result["allocations_pct"].values())
        assert abs(cvar_sum - 100.0) < 1.0
    # If error, it should be a clean error message
    else:
        assert isinstance(cvar_result["error"], str)

    # HRP should handle this (it's robust to singular cov)
    hrp_result = optimize_hrp_raw(returns_matrix, tickers)
    if "error" not in hrp_result:
        hrp_sum = sum(hrp_result["allocations_pct"].values())
        assert abs(hrp_sum - 100.0) < 1.0

    # Max Diversification - diversification_ratio close to 1 for correlated assets
    div_result = optimize_max_diversification_raw(returns_matrix, tickers)
    if "error" not in div_result:
        div_sum = sum(div_result["allocations_pct"].values())
        assert abs(div_sum - 100.0) < 1.0
        # Nearly perfect correlation => diversification ratio ~1.0
        assert div_result["diversification_ratio"] < 1.1, (
            f"Diversification ratio should be ~1.0 for correlated assets, "
            f"got {div_result['diversification_ratio']}"
        )

    # Max Entropy
    ent_result = optimize_entropy_raw(returns_matrix, tickers)
    if "error" not in ent_result:
        ent_sum = sum(ent_result["allocations_pct"].values())
        assert abs(ent_sum - 100.0) < 1.0

    # Kelly - uses pinv for singular cov (should not crash)
    mu = returns_matrix.mean(axis=0)
    cov = np.cov(returns_matrix.T)
    kelly_result = compute_kelly_raw(mu, cov, tickers)
    if "error" not in kelly_result:
        kelly_sum = sum(kelly_result["full_kelly_pct"].values())
        assert abs(kelly_sum - 100.0) < 1.0
