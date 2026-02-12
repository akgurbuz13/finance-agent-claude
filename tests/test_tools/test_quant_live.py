"""Live data validation tests for quantitative models and advanced quant analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import yfinance as yf

from portfolio_advisor.tools.advanced_quant import (
    compute_fama_french_raw,
    compute_garch_raw,
    compute_kalman_beta_raw,
    detect_regime_hmm_raw,
)

try:
    import hmmlearn  # noqa: F401

    HAS_HMMLEARN = True
except ImportError:
    HAS_HMMLEARN = False
from portfolio_advisor.tools.quant_models import (
    compute_factor_exposures_raw,
    compute_return_forecast_raw,
    compute_vol_forecast_raw,
    detect_regime_raw,
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
    """Fetch 1Y SPY OHLCV data."""
    df = _download_and_normalize("SPY")
    assert len(df) > 100, "SPY download returned too few rows"
    return df


@pytest.fixture(scope="module")
def spy_returns(spy_df):
    """Daily returns for SPY as numpy array."""
    ret = spy_df["close"].pct_change().dropna().values
    assert len(ret) > 100
    return ret


@pytest.fixture(scope="module")
def aapl_df():
    """Fetch 1Y AAPL OHLCV data, with retry."""
    import time

    for attempt in range(3):
        df = _download_and_normalize("AAPL")
        if len(df) > 100:
            return df
        time.sleep(2)
    pytest.skip("AAPL download failed after 3 attempts")


@pytest.fixture(scope="module")
def aapl_returns(aapl_df):
    """Daily returns for AAPL as numpy array."""
    return aapl_df["close"].pct_change().dropna().values


@pytest.fixture(scope="module")
def btc_df():
    """Fetch 1Y BTC-USD OHLCV data."""
    df = _download_and_normalize("BTC-USD")
    assert len(df) > 100
    return df


@pytest.fixture(scope="module")
def btc_returns(btc_df):
    """Daily returns for BTC-USD as numpy array."""
    return btc_df["close"].pct_change().dropna().values


@pytest.fixture(scope="module")
def tlt_df():
    """Fetch 1Y TLT OHLCV data."""
    df = _download_and_normalize("TLT")
    assert len(df) > 100
    return df


@pytest.fixture(scope="module")
def tlt_returns(tlt_df):
    """Daily returns for TLT as numpy array."""
    return tlt_df["close"].pct_change().dropna().values


# ── GARCH tests (advanced_quant.py) ──────────────────────────────────────────


@pytest.mark.live
class TestGarchLive:
    def test_garch_live_spy(self, spy_returns):
        result = compute_garch_raw(spy_returns)

        assert "error" not in result
        assert result["model_type"] == "GARCH"
        assert result["forecast_vol_1d"] > 0
        assert 0 < result["persistence"] < 1
        assert result["half_life_days"] is not None
        assert result["half_life_days"] > 0
        assert result["forecast_vol_annualized"] > 0
        assert result["current_cond_vol_daily"] > 0

    def test_garch_live_btc(self, btc_returns):
        result = compute_garch_raw(btc_returns)

        assert "error" not in result
        assert result["forecast_vol_1d"] > 0
        # BTC is typically more volatile than SPY
        assert result["current_cond_vol_annualized"] > 0

    def test_egarch_live(self, spy_returns):
        result = compute_garch_raw(spy_returns, "EGARCH")

        assert "error" not in result
        assert result["model_type"] == "EGARCH"
        assert result["forecast_vol_1d"] > 0

    def test_garch_short_data(self):
        """Less than 100 observations should return an error dict."""
        short_returns = np.random.default_rng(42).standard_normal(50) * 0.01
        result = compute_garch_raw(short_returns)

        assert "error" in result
        assert "Insufficient" in result["error"]


# ── HMM regime detection tests (advanced_quant.py) ───────────────────────────


@pytest.mark.live
@pytest.mark.skipif(not HAS_HMMLEARN, reason="hmmlearn not installed")
class TestHmmRegimeLive:
    def test_hmm_regime_live(self, spy_returns):
        result = detect_regime_hmm_raw(spy_returns)

        assert "error" not in result
        assert result["current_state"] in {"bull", "bear", "transition"}
        assert result["n_states"] == 3

        # State probabilities should sum to ~1
        probs = result["state_probabilities"]
        assert len(probs) == 3
        prob_sum = sum(probs.values())
        assert abs(prob_sum - 1.0) < 0.01

        # State stats should have 3 entries
        assert len(result["state_stats"]) == 3

        # Transition matrix should be 3x3 (dict of dicts)
        trans = result["transition_matrix"]
        assert len(trans) == 3
        for row_label, row in trans.items():
            assert len(row) == 3
            # Each row should sum to ~1
            row_sum = sum(row.values())
            assert abs(row_sum - 1.0) < 0.05

        # Expected durations
        assert len(result["expected_durations_days"]) == 3

    def test_hmm_degenerate_low_vol(self, tlt_returns):
        """TLT (bonds) has low volatility -- HMM should not crash."""
        result = detect_regime_hmm_raw(tlt_returns)

        # Should either succeed or return a structured error, never crash
        assert isinstance(result, dict)
        if "error" not in result:
            assert result["current_state"] in {"bull", "bear", "transition"}


# ── Kalman beta tests (advanced_quant.py) ─────────────────────────────────────


@pytest.mark.live
class TestKalmanBetaLive:
    def test_kalman_beta_live(self, aapl_returns, spy_returns):
        # Align lengths
        min_len = min(len(aapl_returns), len(spy_returns))
        result = compute_kalman_beta_raw(
            aapl_returns[-min_len:],
            spy_returns[-min_len:],
        )

        assert "error" not in result
        assert -2 <= result["current_beta"] <= 5
        assert isinstance(result["current_alpha_daily"], float)
        assert isinstance(result["beta_ci_95"], list)
        assert len(result["beta_ci_95"]) == 2
        assert result["beta_ci_95"][0] < result["beta_ci_95"][1]
        assert result["beta_trend"] in {"increasing", "decreasing", "stable"}

    def test_kalman_beta_crypto(self, btc_returns, spy_returns):
        min_len = min(len(btc_returns), len(spy_returns))
        result = compute_kalman_beta_raw(
            btc_returns[-min_len:],
            spy_returns[-min_len:],
        )

        assert "error" not in result
        assert isinstance(result["current_beta"], float)


# ── Fama-French tests (advanced_quant.py) ─────────────────────────────────────


@pytest.mark.live
class TestFamaFrenchLive:
    def test_fama_french_live(self, spy_returns):
        """SPY regressed on itself should give market_beta near 1.0."""
        n = len(spy_returns)
        # Use spy as both asset and market; SMB and HML as small random noise
        rng = np.random.default_rng(42)
        smb = rng.standard_normal(n) * 0.001
        hml = rng.standard_normal(n) * 0.001

        result = compute_fama_french_raw(spy_returns, spy_returns, smb, hml)

        assert "error" not in result
        assert abs(result["beta_market"] - 1.0) < 0.05
        assert result["r_squared"] > 0.5
        assert isinstance(result["style"], list)
        assert isinstance(result["t_stats"], dict)

    def test_fama_french_insufficient_data(self):
        """Fewer than 60 observations should return an error dict."""
        rng = np.random.default_rng(42)
        short = rng.standard_normal(25) * 0.01
        result = compute_fama_french_raw(short, short, short, short)

        assert "error" in result
        assert "Insufficient" in result["error"]


# ── Return forecast tests (quant_models.py) ──────────────────────────────────


@pytest.mark.live
class TestReturnForecastLive:
    def test_return_forecast_live(self, spy_df):
        result = compute_return_forecast_raw(spy_df)

        assert "error" not in result
        assert "forecasts" in result
        forecasts = result["forecasts"]
        assert "1w" in forecasts
        assert "1m" in forecasts
        assert "3m" in forecasts

        for horizon in ("1w", "1m", "3m"):
            fc = forecasts[horizon]
            assert "expected_return_pct" in fc
            assert isinstance(fc["expected_return_pct"], float)
            assert fc["ci_low_pct"] < fc["ci_high_pct"]


# ── Volatility forecast tests (quant_models.py) ──────────────────────────────


@pytest.mark.live
class TestVolForecastLive:
    def test_vol_forecast_live(self, spy_df):
        result = compute_vol_forecast_raw(spy_df)

        assert "error" not in result
        assert result["ewma_vol_daily"] > 0
        assert result["ewma_vol_annualized"] > 0
        assert result["vol_regime"] in {"low", "normal", "high"}
        assert 0 <= result["vol_percentile_1y"] <= 100


# ── Regime detection tests (quant_models.py) ─────────────────────────────────


@pytest.mark.live
class TestRegimeDetectionLive:
    def test_regime_detection_live(self, spy_df):
        result = detect_regime_raw(spy_df)

        assert "error" not in result
        assert result["regime"] in {"trending", "mean_reverting", "volatile", "neutral"}
        assert 0 < result["hurst_exponent"] < 1
        assert result["hurst_interpretation"] in {
            "trending",
            "mean_reverting",
            "random_walk",
        }
        assert isinstance(result["vol_autocorrelation"], float)
        assert 0 < result["confidence"] <= 1.0


# ── Factor exposures tests (quant_models.py) ─────────────────────────────────


@pytest.mark.live
class TestFactorExposuresLive:
    def test_factor_exposures_live(self, spy_returns):
        """SPY vs SPY should give market_beta near 1.0 and R-squared near 1.0."""
        result = compute_factor_exposures_raw(spy_returns, spy_returns)

        assert "error" not in result
        assert isinstance(result["market_beta"], float)
        assert abs(result["market_beta"] - 1.0) < 0.01
        assert 0 <= result["r_squared"] <= 1.0
        assert result["r_squared"] > 0.99  # SPY vs itself
        assert result["interpretation"] in {
            "defensive",
            "market_neutral",
            "aggressive",
        }
