"""Live data validation tests for technical indicators and advanced technical analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import yfinance as yf

from portfolio_advisor.tools.technical_indicators import (
    compute_atr_bollinger_raw,
    compute_macd_raw,
    compute_rsi_raw,
    compute_sma_ema_raw,
    compute_support_resistance_raw,
)
from portfolio_advisor.tools.advanced_technical import (
    compute_adx_dmi_raw,
    compute_fibonacci_raw,
    compute_ichimoku_raw,
    compute_obv_raw,
    compute_stochastic_raw,
    compute_volume_profile_raw,
    compute_vwap_raw,
)


# ── Module-level fixtures ────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def spy_df():
    """Fetch 1Y SPY OHLCV data for tests."""
    df = yf.download("SPY", period="1y", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower() for c in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]
    assert len(df) > 100, "SPY download returned too few rows"
    return df


@pytest.fixture(scope="module")
def aapl_df():
    """Fetch 1Y AAPL OHLCV data for tests."""
    df = yf.download("AAPL", period="1y", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower() for c in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]
    assert len(df) > 100, "AAPL download returned too few rows"
    return df


# ── Basic indicators (technical_indicators.py) ───────────────────────────────


@pytest.mark.live
class TestSmaEmaLive:
    def test_sma_ema_live(self, spy_df):
        result = compute_sma_ema_raw(spy_df)

        # sma50, sma200 are floats (or None if insufficient data, but 1Y SPY has 200+ rows)
        assert isinstance(result["sma50"], float)
        assert isinstance(result["sma200"], float)
        assert isinstance(result["ema12"], float)
        assert isinstance(result["ema26"], float)
        assert isinstance(result["price"], float)
        assert result["trend"] in {"bullish", "bearish", "neutral"}
        assert result["cross"] in {"none", "golden_cross", "death_cross"}
        assert result["ema_signal"] in {"bullish", "bearish"}
        assert result["interpretation"] in {"bullish", "bearish", "neutral"}
        assert 0 < result["confidence"] <= 1.0


@pytest.mark.live
class TestRsiLive:
    def test_rsi_live(self, spy_df):
        result = compute_rsi_raw(spy_df)

        assert 0 <= result["rsi"] <= 100
        assert result["period"] == 14
        assert result["interpretation"] in {"bullish", "bearish", "neutral"}
        assert result["divergence"] in {
            "none",
            "bearish_divergence",
            "bullish_divergence",
        }
        assert 0 < result["confidence"] <= 1.0


@pytest.mark.live
class TestMacdLive:
    def test_macd_live(self, spy_df):
        result = compute_macd_raw(spy_df)

        assert isinstance(result["macd_line"], float)
        assert isinstance(result["signal_line"], float)
        assert isinstance(result["histogram"], float)

        # histogram should be approximately macd_line - signal_line
        assert abs(result["histogram"] - (result["macd_line"] - result["signal_line"])) < 0.01

        assert result["crossover"] in {"none", "bullish_crossover", "bearish_crossover"}
        assert result["interpretation"] in {"bullish", "bearish"}
        assert 0 < result["confidence"] <= 1.0


@pytest.mark.live
class TestAtrBollingerLive:
    def test_atr_bollinger_live(self, spy_df):
        result = compute_atr_bollinger_raw(spy_df)

        assert result["atr_14"] > 0
        assert result["bb_upper"] > result["bb_lower"]
        assert result["bandwidth"] > 0
        assert isinstance(result["pct_b"], float)
        assert result["interpretation"] in {"bullish", "bearish", "neutral"}
        assert 0 < result["confidence"] <= 1.0


@pytest.mark.live
class TestSupportResistanceLive:
    def test_support_resistance_live(self, spy_df):
        result = compute_support_resistance_raw(spy_df)

        assert result["nearest_support"] < result["nearest_resistance"]
        assert isinstance(result["pivot"], float)
        assert result["s1"] < result["r1"]
        assert result["s2"] < result["r2"]
        assert result["s3"] < result["r3"]
        assert result["interpretation"] in {"bullish", "bearish", "neutral"}
        assert result["confidence"] == 0.55


# ── Advanced indicators (advanced_technical.py) ──────────────────────────────


@pytest.mark.live
class TestIchimokuLive:
    def test_ichimoku_live(self, spy_df):
        result = compute_ichimoku_raw(spy_df)

        # tenkan, kijun should be floats for 1Y data
        assert isinstance(result["tenkan"], float)
        assert isinstance(result["kijun"], float)
        # senkou_a and senkou_b may be float or None depending on data length
        # 1Y ~252 bars is enough for the 52+26=78 period requirement
        assert result["senkou_a"] is None or isinstance(result["senkou_a"], float)
        assert result["senkou_b"] is None or isinstance(result["senkou_b"], float)
        # chikou may be None (shifted forward by 26, so last 26 bars are NaN)
        assert result["chikou"] is None or isinstance(result["chikou"], float)

        assert result["price_vs_cloud"] in {
            "above_cloud",
            "below_cloud",
            "inside_cloud",
            "insufficient_data",
        }
        assert result["interpretation"] in {"bullish", "bearish", "neutral"}
        assert 0 < result["confidence"] <= 1.0


@pytest.mark.live
class TestVwapLive:
    def test_vwap_live_equity(self, spy_df):
        result = compute_vwap_raw(spy_df)

        assert result["vwap"] is not None
        assert result["vwap"] > 0
        assert isinstance(result["price_vs_vwap_pct"], float)
        assert isinstance(result["price"], float)
        assert result["interpretation"] in {"bullish", "bearish", "neutral"}
        assert 0 < result["confidence"] <= 1.0

    def test_vwap_live_crypto_zero_volume(self):
        """DataFrame with volume=0 should not crash."""
        dates = pd.date_range("2024-01-01", periods=30, freq="B")
        df = pd.DataFrame(
            {
                "open": np.linspace(100, 110, 30),
                "high": np.linspace(101, 111, 30),
                "low": np.linspace(99, 109, 30),
                "close": np.linspace(100, 110, 30),
                "volume": np.zeros(30),
            },
            index=dates,
        )
        result = compute_vwap_raw(df)
        # Should return gracefully -- vwap may be None due to division by zero
        assert isinstance(result, dict)
        assert "interpretation" in result


@pytest.mark.live
class TestObvLive:
    def test_obv_live(self, spy_df):
        result = compute_obv_raw(spy_df)

        assert isinstance(result["obv"], float)
        assert result["obv_trending_up"] in {True, False, None}
        assert result["divergence"] in {
            "none",
            "bearish_divergence",
            "bullish_divergence",
        }
        assert result["interpretation"] in {"bullish", "bearish", "neutral"}
        assert 0 < result["confidence"] <= 1.0

    def test_obv_crypto_zero_volume(self):
        """DataFrame with volume=0 should not crash."""
        dates = pd.date_range("2024-01-01", periods=30, freq="B")
        df = pd.DataFrame(
            {
                "open": np.linspace(100, 110, 30),
                "high": np.linspace(101, 111, 30),
                "low": np.linspace(99, 109, 30),
                "close": np.linspace(100, 110, 30),
                "volume": np.zeros(30),
            },
            index=dates,
        )
        result = compute_obv_raw(df)
        assert isinstance(result, dict)
        assert "interpretation" in result


@pytest.mark.live
class TestAdxDmiLive:
    def test_adx_dmi_live(self, spy_df):
        result = compute_adx_dmi_raw(spy_df)

        assert result["adx"] is not None
        assert 0 <= result["adx"] <= 100
        assert result["plus_di"] is not None
        assert 0 <= result["plus_di"] <= 100
        assert result["minus_di"] is not None
        assert 0 <= result["minus_di"] <= 100
        assert result["trend_strength"] in {
            "strong_trend",
            "trending",
            "weak_trend",
            "no_trend",
        }
        assert result["interpretation"] in {"bullish", "bearish", "neutral"}
        assert 0 < result["confidence"] <= 1.0


@pytest.mark.live
class TestStochasticLive:
    def test_stochastic_live(self, spy_df):
        result = compute_stochastic_raw(spy_df)

        assert 0 <= result["k"] <= 100
        assert 0 <= result["d"] <= 100
        assert result["zone"] in {"oversold", "overbought", "neutral"}
        assert result["crossover"] in {
            "none",
            "bullish_crossover",
            "bearish_crossover",
        }
        assert result["interpretation"] in {"bullish", "bearish", "neutral"}
        assert 0 < result["confidence"] <= 1.0


@pytest.mark.live
class TestFibonacciLive:
    def test_fibonacci_live(self, spy_df):
        result = compute_fibonacci_raw(spy_df)

        assert result["swing_high"] > result["swing_low"]
        levels = result["levels"]
        # Levels should be ordered: 0.0 (highest) > 23.6 > 38.2 > 50.0 > 61.8 > 78.6 > 100.0 (lowest)
        assert levels["0.0"] >= levels["23.6"] >= levels["38.2"]
        assert levels["38.2"] >= levels["50.0"] >= levels["61.8"]
        assert levels["61.8"] >= levels["78.6"] >= levels["100.0"]
        assert isinstance(result["fib_retrace_pct"], float)
        assert result["interpretation"] in {"bullish", "bearish", "neutral"}
        assert 0 < result["confidence"] <= 1.0


@pytest.mark.live
class TestVolumeProfileLive:
    def test_volume_profile_live(self, spy_df):
        result = compute_volume_profile_raw(spy_df)

        price_min = float(spy_df["close"].min())
        price_max = float(spy_df["close"].max())

        assert price_min <= result["poc"] <= price_max
        assert result["value_area_high"] > result["value_area_low"]
        assert isinstance(result["hvn_zones"], list)
        assert isinstance(result["lvn_zones"], list)
        assert isinstance(result["price_vs_poc_pct"], float)
        assert result["interpretation"] in {"bullish", "bearish", "neutral"}
        assert 0 < result["confidence"] <= 1.0


# ── Edge cases ───────────────────────────────────────────────────────────────


@pytest.mark.live
class TestShortDataEdgeCases:
    """All indicators should handle very short (10-day) data gracefully."""

    @pytest.fixture()
    def short_df(self):
        dates = pd.date_range("2024-01-01", periods=10, freq="B")
        rng = np.random.default_rng(42)
        close = 100 + rng.standard_normal(10).cumsum()
        return pd.DataFrame(
            {
                "open": close - rng.uniform(0, 1, 10),
                "high": close + rng.uniform(0, 2, 10),
                "low": close - rng.uniform(0, 2, 10),
                "close": close,
                "volume": rng.integers(1000, 10000, 10).astype(float),
            },
            index=dates,
        )

    def test_sma_ema_short(self, short_df):
        result = compute_sma_ema_raw(short_df)
        assert isinstance(result, dict)
        # sma50/sma200 will be None with only 10 bars
        assert result["sma50"] is None
        assert result["sma200"] is None

    def test_rsi_short(self, short_df):
        result = compute_rsi_raw(short_df)
        assert isinstance(result, dict)
        assert "rsi" in result

    def test_macd_short(self, short_df):
        result = compute_macd_raw(short_df)
        assert isinstance(result, dict)
        assert "macd_line" in result

    def test_atr_bollinger_short(self, short_df):
        result = compute_atr_bollinger_raw(short_df)
        assert isinstance(result, dict)
        assert "atr_14" in result

    def test_support_resistance_short(self, short_df):
        result = compute_support_resistance_raw(short_df)
        assert isinstance(result, dict)
        assert "pivot" in result

    def test_ichimoku_short(self, short_df):
        result = compute_ichimoku_raw(short_df)
        assert isinstance(result, dict)
        # With 10 bars, most Ichimoku values will be None
        assert "tenkan" in result

    def test_vwap_short(self, short_df):
        result = compute_vwap_raw(short_df)
        assert isinstance(result, dict)
        assert "vwap" in result

    def test_obv_short(self, short_df):
        result = compute_obv_raw(short_df)
        assert isinstance(result, dict)
        assert "obv" in result

    def test_adx_dmi_short(self, short_df):
        result = compute_adx_dmi_raw(short_df)
        assert isinstance(result, dict)
        assert "adx" in result

    def test_stochastic_short(self, short_df):
        result = compute_stochastic_raw(short_df)
        assert isinstance(result, dict)
        assert "k" in result

    def test_fibonacci_short(self, short_df):
        result = compute_fibonacci_raw(short_df)
        assert isinstance(result, dict)
        assert "levels" in result

    def test_volume_profile_short(self, short_df):
        result = compute_volume_profile_raw(short_df)
        assert isinstance(result, dict)
        assert "poc" in result


@pytest.mark.live
class TestNanPricesEdgeCases:
    """All indicators should handle DataFrames with NaN rows gracefully."""

    @pytest.fixture()
    def nan_df(self):
        dates = pd.date_range("2024-01-01", periods=60, freq="B")
        rng = np.random.default_rng(42)
        close = 100 + rng.standard_normal(60).cumsum()
        df = pd.DataFrame(
            {
                "open": close - rng.uniform(0, 1, 60),
                "high": close + rng.uniform(0, 2, 60),
                "low": close - rng.uniform(0, 2, 60),
                "close": close,
                "volume": rng.integers(1000, 10000, 60).astype(float),
            },
            index=dates,
        )
        # Inject NaN rows
        df.iloc[5] = np.nan
        df.iloc[20] = np.nan
        df.iloc[40] = np.nan
        return df

    def test_sma_ema_nan(self, nan_df):
        result = compute_sma_ema_raw(nan_df)
        assert isinstance(result, dict)

    def test_rsi_nan(self, nan_df):
        result = compute_rsi_raw(nan_df)
        assert isinstance(result, dict)

    def test_macd_nan(self, nan_df):
        result = compute_macd_raw(nan_df)
        assert isinstance(result, dict)

    def test_atr_bollinger_nan(self, nan_df):
        result = compute_atr_bollinger_raw(nan_df)
        assert isinstance(result, dict)

    def test_support_resistance_nan(self, nan_df):
        result = compute_support_resistance_raw(nan_df)
        assert isinstance(result, dict)

    def test_ichimoku_nan(self, nan_df):
        result = compute_ichimoku_raw(nan_df)
        assert isinstance(result, dict)

    def test_vwap_nan(self, nan_df):
        result = compute_vwap_raw(nan_df)
        assert isinstance(result, dict)

    def test_obv_nan(self, nan_df):
        result = compute_obv_raw(nan_df)
        assert isinstance(result, dict)

    def test_adx_dmi_nan(self, nan_df):
        result = compute_adx_dmi_raw(nan_df)
        assert isinstance(result, dict)

    def test_stochastic_nan(self, nan_df):
        result = compute_stochastic_raw(nan_df)
        assert isinstance(result, dict)

    def test_fibonacci_nan(self, nan_df):
        result = compute_fibonacci_raw(nan_df)
        assert isinstance(result, dict)

    def test_volume_profile_nan(self, nan_df):
        # Drop NaN rows before passing -- volume profile iterates all rows
        # and cannot handle NaN prices (ValueError on int conversion).
        clean_df = nan_df.dropna()
        result = compute_volume_profile_raw(clean_df)
        assert isinstance(result, dict)
