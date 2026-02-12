"""Tests for technical indicator computations."""

import numpy as np
import pandas as pd

from portfolio_advisor.tools.technical_indicators import (
    compute_atr_bollinger_raw,
    compute_macd_raw,
    compute_rsi_raw,
    compute_sma_ema_raw,
)


class TestSMAEMA:
    """Tests for SMA/EMA calculations."""

    def _make_df(self, n=250):
        dates = pd.bdate_range(end=pd.Timestamp.now(), periods=n)
        n_actual = len(dates)
        close = np.exp(np.random.randn(n_actual).cumsum() * 0.015) * 100
        return pd.DataFrame({"close": close, "high": close * 1.01, "low": close * 0.99}, index=dates)

    def test_sma50_sma200_computed(self):
        df = self._make_df(n=250)
        result = compute_sma_ema_raw(df)
        assert "sma50" in result
        assert "sma200" in result
        assert result["sma50"] > 0
        assert result["sma200"] > 0

    def test_ema12_ema26_computed(self):
        df = self._make_df(n=100)
        result = compute_sma_ema_raw(df)
        assert "ema12" in result
        assert "ema26" in result


class TestRSI:
    """Tests for RSI calculation."""

    def _make_df(self, n=100):
        dates = pd.bdate_range(end=pd.Timestamp.now(), periods=n)
        close = np.exp(np.random.randn(n).cumsum() * 0.015) * 100
        return pd.DataFrame({"close": close}, index=dates)

    def test_rsi_in_bounds(self):
        df = self._make_df(n=100)
        result = compute_rsi_raw(df)
        if "error" not in result:
            assert 0 <= result["rsi"] <= 100

    def test_rsi_interpretation_present(self):
        df = self._make_df(n=100)
        result = compute_rsi_raw(df)
        if "error" not in result:
            assert "interpretation" in result
            assert result["interpretation"] in ("overbought", "oversold", "neutral")


class TestMACD:
    """Tests for MACD calculation."""

    def _make_df(self, n=100):
        dates = pd.bdate_range(end=pd.Timestamp.now(), periods=n)
        close = np.exp(np.random.randn(n).cumsum() * 0.015) * 100
        return pd.DataFrame({"close": close}, index=dates)

    def test_macd_components_present(self):
        df = self._make_df(n=100)
        result = compute_macd_raw(df)
        assert "macd_line" in result
        assert "signal_line" in result
        assert "histogram" in result


class TestATRBollinger:
    """Tests for ATR + Bollinger Bands."""

    def _make_df(self, n=100):
        dates = pd.bdate_range(end=pd.Timestamp.now(), periods=n)
        n_actual = len(dates)
        close = np.exp(np.random.randn(n_actual).cumsum() * 0.015) * 100
        high = close * (1 + np.random.rand(n_actual) * 0.02)
        low = close * (1 - np.random.rand(n_actual) * 0.02)
        return pd.DataFrame({"close": close, "high": high, "low": low}, index=dates)

    def test_bollinger_bands_ordered(self):
        df = self._make_df(n=100)
        result = compute_atr_bollinger_raw(df)
        if "error" not in result:
            assert result["bb_lower"] <= result["bb_upper"]
            assert result["atr_14"] > 0
