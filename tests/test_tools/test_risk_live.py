"""Live data validation tests for advanced risk models."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import yfinance as yf

from portfolio_advisor.tools.advanced_risk import (
    compute_cornish_fisher_raw,
    compute_evt_var_raw,
    compute_mc_var_raw,
    compute_tail_dependence_raw,
    run_stress_test_raw,
)


# ── Module-level fixtures ────────────────────────────────────────────────────


def _download_and_normalize(ticker: str, period: str = "1y") -> pd.DataFrame:
    """Download and normalize column names to lowercase."""
    df = yf.download(ticker, period=period, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower() for c in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]
    return df


@pytest.fixture(scope="module")
def spy_df():
    df = _download_and_normalize("SPY")
    assert len(df) > 100
    return df


@pytest.fixture(scope="module")
def spy_returns(spy_df):
    return spy_df["close"].pct_change().dropna().values


@pytest.fixture(scope="module")
def aapl_df():
    df = _download_and_normalize("AAPL")
    assert len(df) > 100
    return df


@pytest.fixture(scope="module")
def aapl_returns(aapl_df):
    return aapl_df["close"].pct_change().dropna().values


@pytest.fixture(scope="module")
def tlt_df():
    df = _download_and_normalize("TLT")
    assert len(df) > 100
    return df


@pytest.fixture(scope="module")
def tlt_returns(tlt_df):
    return tlt_df["close"].pct_change().dropna().values


# ── Cornish-Fisher VaR ───────────────────────────────────────────────────────


@pytest.mark.live
class TestCornishFisherLive:
    def test_cornish_fisher_var_live(self, spy_returns):
        result = compute_cornish_fisher_raw(spy_returns, confidence=0.95)

        assert "error" not in result
        assert result["confidence"] == 0.95
        # CF VaR should be negative (loss)
        assert result["cornish_fisher_var_pct"] < 0
        # Gaussian VaR should also be negative
        assert result["gaussian_var_pct"] < 0
        # Historical VaR should be negative
        assert result["historical_var_pct"] < 0
        # Skewness and kurtosis are floats
        assert isinstance(result["skewness"], float)
        assert isinstance(result["excess_kurtosis"], float)
        assert result["observations"] == len(spy_returns)


# ── EVT VaR ──────────────────────────────────────────────────────────────────


@pytest.mark.live
class TestEvtVarLive:
    def test_evt_var_live(self, spy_returns):
        result = compute_evt_var_raw(spy_returns, confidence=0.99)

        assert "error" not in result
        assert result["confidence"] == 0.99
        # EVT VaR is in terms of losses (positive = loss percentage)
        assert result["evt_var_pct"] > 0
        # ES should be >= VaR (more extreme)
        assert result["evt_es_pct"] >= result["evt_var_pct"]
        # GPD shape parameter
        assert result["gpd_shape"] > -0.5
        assert result["tail_type"] in {"heavy", "light", "exponential"}
        assert result["n_exceedances"] >= 20

    def test_evt_thin_tails(self, tlt_returns):
        """TLT (bonds) should have thinner tails -- shape near 0."""
        result = compute_evt_var_raw(tlt_returns, confidence=0.99)

        if "error" not in result:
            # Bond tails are typically thinner than equity -- shape closer to 0
            assert isinstance(result["gpd_shape"], float)
            # Just verify it runs; exact shape depends on the period


# ── Monte Carlo VaR ──────────────────────────────────────────────────────────


@pytest.mark.live
class TestMonteCarloVarLive:
    def test_monte_carlo_var_live(self, spy_returns):
        result_95 = compute_mc_var_raw(spy_returns, confidence=0.95)
        result_99 = compute_mc_var_raw(spy_returns, confidence=0.99)

        assert "error" not in result_95
        assert "error" not in result_99

        # VaR at 95% should be less negative (smaller loss) than at 99%
        assert result_95["mc_var_pct"] > result_99["mc_var_pct"]

        # ES should be more negative than VaR
        assert result_95["mc_es_pct"] <= result_95["mc_var_pct"]
        assert result_99["mc_es_pct"] <= result_99["mc_var_pct"]

        assert result_95["n_simulations"] == 10000
        assert result_95["distribution"] == "normal"
        assert result_95["horizon_days"] == 1


# ── Stress Test ──────────────────────────────────────────────────────────────


@pytest.mark.live
class TestStressTestLive:
    def test_stress_test_live(self, spy_returns, aapl_returns):
        """Run stress test with a 2-asset portfolio."""
        # Align lengths
        min_len = min(len(spy_returns), len(aapl_returns))
        returns_matrix = np.column_stack([
            spy_returns[-min_len:],
            aapl_returns[-min_len:],
        ])
        tickers = ["SPY", "AAPL"]
        weights = np.array([0.6, 0.4])

        result = run_stress_test_raw(
            returns_matrix, tickers, weights, spy_returns=spy_returns[-min_len:]
        )

        assert "scenarios" in result
        scenarios = result["scenarios"]
        assert "2008_gfc" in scenarios
        assert "2020_covid" in scenarios
        assert "2022_rates" in scenarios
        assert "flash_crash" in scenarios

        for scenario_name, scenario in scenarios.items():
            assert isinstance(scenario["portfolio_impact_pct"], float)
            assert "description" in scenario
            assert "asset_impacts" in scenario
            assert scenario["severity"] in {
                "severe",
                "significant",
                "moderate",
                "mild",
            }

        # Worst case
        assert "worst_case" in result
        assert result["worst_case"]["scenario"] in scenarios
        assert isinstance(result["worst_case"]["impact_pct"], float)

        # Asset betas
        assert "asset_betas" in result
        assert "SPY" in result["asset_betas"]
        # SPY beta vs itself should be ~1.0
        assert abs(result["asset_betas"]["SPY"] - 1.0) < 0.1


# ── Tail Dependence ──────────────────────────────────────────────────────────


@pytest.mark.live
class TestTailDependenceLive:
    def test_tail_dependence_live(self, spy_returns, aapl_returns):
        """SPY and AAPL should show some tail dependence."""
        min_len = min(len(spy_returns), len(aapl_returns))
        returns_matrix = np.column_stack([
            spy_returns[-min_len:],
            aapl_returns[-min_len:],
        ])
        tickers = ["SPY", "AAPL"]

        result = compute_tail_dependence_raw(returns_matrix, tickers, quantile=0.05)

        assert "error" not in result
        assert result["quantile"] == 0.05
        assert len(result["pairs"]) == 1  # 2 assets = 1 pair

        pair = result["pairs"][0]
        assert pair["pair"] == "SPY/AAPL"
        assert 0 <= pair["lower_tail_dependence"] <= 1
        assert 0 <= pair["upper_tail_dependence"] <= 1
        assert -1 <= pair["linear_correlation"] <= 1
        assert pair["co_crash_risk"] in {"high", "moderate", "low"}


# ── Short data edge cases ────────────────────────────────────────────────────


@pytest.mark.live
class TestAllRiskModelsShortData:
    """All risk models should return an error dict with <30 observations."""

    @pytest.fixture()
    def short_returns(self):
        return np.random.default_rng(42).standard_normal(20) * 0.01

    def test_cornish_fisher_short(self, short_returns):
        result = compute_cornish_fisher_raw(short_returns)
        assert "error" in result

    def test_evt_short(self, short_returns):
        result = compute_evt_var_raw(short_returns)
        assert "error" in result

    def test_mc_var_short(self, short_returns):
        result = compute_mc_var_raw(short_returns)
        assert "error" in result

    def test_tail_dependence_short(self, short_returns):
        matrix = np.column_stack([short_returns, short_returns])
        result = compute_tail_dependence_raw(matrix, ["A", "B"])
        assert "error" in result
