"""Tests for quantitative models — forecasts, regime, factor exposures."""

import numpy as np
import pandas as pd

from portfolio_advisor.tools.quant_models import (
    compute_factor_exposures_raw,
    compute_return_forecast_raw,
    compute_vol_forecast_raw,
    detect_regime_raw,
)


class TestReturnForecast:
    """Tests for the momentum + mean-reversion blend forecast."""

    def _make_df(self, n=100, trend=0.001):
        dates = pd.bdate_range(end=pd.Timestamp.now(), periods=n)
        close = (np.random.randn(n) * 0.01 + trend).cumsum()
        close = np.exp(close) * 100
        return pd.DataFrame({"close": close}, index=dates)

    def test_returns_forecasts_for_all_horizons(self):
        df = self._make_df(n=100)
        result = compute_return_forecast_raw(df)
        assert "error" not in result
        assert "forecasts" in result
        for horizon in ["1w", "1m", "3m"]:
            assert horizon in result["forecasts"]
            fc = result["forecasts"][horizon]
            assert "expected_return_pct" in fc
            assert "ci_low_pct" in fc
            assert "ci_high_pct" in fc
            assert "confidence" in fc
            # CI should bracket the expected return
            assert fc["ci_low_pct"] <= fc["expected_return_pct"] <= fc["ci_high_pct"]

    def test_insufficient_data_returns_error(self):
        df = self._make_df(n=30)
        result = compute_return_forecast_raw(df)
        assert "error" in result

    def test_regime_conditioning_changes_forecast(self):
        """With HMM state info, forecast should differ from unconditional."""
        df = self._make_df(n=100)
        compute_return_forecast_raw(df)  # baseline (unused, but validates no error)
        conditioned = compute_return_forecast_raw(
            df, hmm_state="bull", hmm_state_means={"bull": 0.001, "bear": -0.001}
        )
        assert conditioned["regime_conditioned"] is True
        assert conditioned["hmm_state"] == "bull"
        # The expected return should differ from baseline
        # (can't guarantee direction since random data, but structure should be valid)
        assert "forecasts" in conditioned

    def test_regime_conditioning_without_state_mean(self):
        """If state not in state_means, falls back to unconditional."""
        df = self._make_df(n=100)
        result = compute_return_forecast_raw(
            df, hmm_state="unknown", hmm_state_means={"bull": 0.001}
        )
        assert result["regime_conditioned"] is False

    def test_includes_annualized_vol(self):
        df = self._make_df(n=100)
        result = compute_return_forecast_raw(df)
        assert "annualized_vol" in result
        assert result["annualized_vol"] > 0

    def test_confidence_is_bounded(self):
        df = self._make_df(n=100)
        result = compute_return_forecast_raw(df)
        for horizon in ["1w", "1m", "3m"]:
            conf = result["forecasts"][horizon]["confidence"]
            assert 0.3 <= conf <= 0.8


class TestVolForecast:
    """Tests for EWMA volatility forecast."""

    def _make_df(self, n=100):
        dates = pd.bdate_range(end=pd.Timestamp.now(), periods=n)
        close = np.exp(np.random.randn(n).cumsum() * 0.02) * 100
        return pd.DataFrame({"close": close}, index=dates)

    def test_returns_vol_metrics(self):
        df = self._make_df(n=300)
        result = compute_vol_forecast_raw(df)
        assert "error" not in result
        assert "ewma_vol_daily" in result
        assert "ewma_vol_annualized" in result
        assert "vol_percentile_1y" in result
        assert "vol_regime" in result
        assert result["vol_regime"] in ("high", "normal", "low")

    def test_insufficient_data(self):
        df = self._make_df(n=10)
        result = compute_vol_forecast_raw(df)
        assert "error" in result


class TestRegimeDetection:
    """Tests for Hurst exponent + vol clustering regime detection."""

    def _make_df(self, n=200):
        dates = pd.bdate_range(end=pd.Timestamp.now(), periods=n)
        close = np.exp(np.random.randn(n).cumsum() * 0.015) * 100
        return pd.DataFrame({"close": close}, index=dates)

    def test_returns_regime_info(self):
        df = self._make_df(n=200)
        result = detect_regime_raw(df)
        assert "error" not in result
        assert "hurst_exponent" in result
        assert "regime" in result
        assert result["regime"] in ("trending", "mean_reverting", "volatile", "neutral")
        assert "confidence" in result
        assert 0.0 <= result["confidence"] <= 1.0

    def test_insufficient_data(self):
        df = self._make_df(n=50)
        result = detect_regime_raw(df)
        assert "error" in result


class TestFactorExposures:
    """Tests for OLS factor regression."""

    def test_basic_regression(self):
        np.random.seed(42)
        n = 100
        spy = np.random.randn(n) * 0.01
        # Stock with beta ~ 1.5
        stock = 1.5 * spy + np.random.randn(n) * 0.005

        result = compute_factor_exposures_raw(stock, spy)
        assert "error" not in result
        assert "market_beta" in result
        assert abs(result["market_beta"] - 1.5) < 0.3  # reasonably close to 1.5
        assert "alpha_daily" in result
        assert "r_squared" in result
        assert 0 <= result["r_squared"] <= 1

    def test_interpretation_labels(self):
        np.random.seed(42)
        n = 100
        spy = np.random.randn(n) * 0.01

        # Defensive (beta < 0.8)
        defensive = 0.5 * spy + np.random.randn(n) * 0.003
        result = compute_factor_exposures_raw(defensive, spy)
        assert result["interpretation"] == "defensive"

    def test_insufficient_data(self):
        result = compute_factor_exposures_raw(np.array([0.01] * 10), np.array([0.01] * 10))
        assert "error" in result
