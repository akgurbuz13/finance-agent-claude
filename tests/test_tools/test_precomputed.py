"""Tests for precomputed pipeline utilities."""


import numpy as np
import pandas as pd

from portfolio_advisor.tools.precomputed import (
    _check_data_quality,
    _generate_ticker_narrative,
)


class TestDataQuality:
    """Tests for the data quality check function."""

    def _make_df(self, days=100, with_volume=True, zero_volume_days=0):
        """Helper to create a sample DataFrame."""
        dates = pd.bdate_range(end=pd.Timestamp.now(), periods=days)
        close = np.random.lognormal(mean=0, sigma=0.02, size=days).cumprod() * 100
        data = {"close": close}
        if with_volume:
            vol = np.random.randint(1_000_000, 50_000_000, size=days)
            if zero_volume_days > 0:
                vol[-zero_volume_days:] = 0
            data["volume"] = vol
        df = pd.DataFrame(data, index=dates)
        return df

    def test_healthy_data_has_no_warnings(self):
        df = self._make_df(days=100)
        result = _check_data_quality(df, "AAPL")
        assert result["is_valid"]
        assert result["warnings"] == []

    def test_stale_data_warns(self):
        """Data ending >5 business days ago should warn about delisting."""
        # End date ~30 business days ago
        end = pd.Timestamp.now() - pd.Timedelta(days=45)
        dates = pd.bdate_range(end=end, periods=50)
        close = np.ones(len(dates)) * 100.0
        df = pd.DataFrame({"close": close}, index=dates)
        result = _check_data_quality(df, "DELISTED")
        assert not result["is_valid"]
        assert any("delisted" in w.lower() or "business days" in w.lower()
                    for w in result["warnings"])

    def test_zero_volume_streak_warns(self):
        df = self._make_df(days=50, zero_volume_days=6)
        result = _check_data_quality(df, "STALE")
        assert any("zero-volume" in w.lower() or "stale" in w.lower()
                    for w in result["warnings"])

    def test_no_volume_data_warns(self):
        df = self._make_df(days=50, with_volume=True)
        df["volume"] = 0  # all zero
        result = _check_data_quality(df, "NOVOLUME")
        assert any("volume" in w.lower() for w in result["warnings"])

    def test_large_price_gap_warns(self):
        """A >50% gap should trigger a warning."""
        dates = pd.bdate_range(end=pd.Timestamp.now(), periods=20)
        close = np.ones(20) * 100.0
        close[10] = 200.0  # 100% gap
        df = pd.DataFrame({"close": close}, index=dates)
        result = _check_data_quality(df, "SPLIT")
        assert any("gap" in w.lower() or "split" in w.lower()
                    for w in result["warnings"])

    def test_no_volume_column_ok(self):
        """If volume column is missing entirely, no volume-related warnings."""
        df = self._make_df(days=50, with_volume=False)
        result = _check_data_quality(df, "NOVOL")
        # Should not have volume warnings (no volume column to check)
        assert not any("volume" in w.lower() for w in result["warnings"])


class TestGenerateTickerNarrative:
    """Tests for the narrative generator."""

    def test_basic_narrative_structure(self):
        tech_data = {
            "sma50": 150.0,
            "sma200": 140.0,
            "rsi_14": 55.0,
            "macd_histogram": 0.5,
            "adx": 30.0,
            "stochastic_k": 60.0,
            "r1": 165.0,
            "s1": 145.0,
        }
        result = _generate_ticker_narrative(
            "AAPL", tech_data, 155.0, "bullish", 0.75
        )
        assert "AAPL" in result
        assert "Bullish" in result
        assert "0.75" in result
        assert "RSI=55" in result

    def test_overbought_rsi(self):
        tech_data = {"rsi_14": 75.0}
        result = _generate_ticker_narrative("SPY", tech_data, 500.0, "bullish", 0.6)
        assert "overbought" in result

    def test_oversold_rsi(self):
        tech_data = {"rsi_14": 25.0}
        result = _generate_ticker_narrative("XLE", tech_data, 80.0, "bearish", 0.6)
        assert "oversold" in result

    def test_golden_cross(self):
        tech_data = {"sma50": 150.0, "sma200": 140.0}
        result = _generate_ticker_narrative("AAPL", tech_data, 155.0, "bullish", 0.7)
        assert "golden cross" in result

    def test_death_cross(self):
        tech_data = {"sma50": 130.0, "sma200": 140.0}
        result = _generate_ticker_narrative("AAPL", tech_data, 125.0, "bearish", 0.7)
        assert "death cross" in result

    def test_fundamentals_enrichment(self):
        tech_data = {"rsi_14": 55.0}
        fundamentals = {
            "pe_ratio": 28.5,
            "analyst_rating": "Strong Buy",
            "analyst_target_price": 220.0,
            "short_pct_float": 1.5,
        }
        result = _generate_ticker_narrative(
            "AAPL", tech_data, 180.0, "bullish", 0.7,
            fundamentals=fundamentals,
        )
        assert "PE=28.5" in result
        assert "Strong Buy" in result
        assert "Target $220" in result
        assert "Short 1.5%" in result

    def test_no_fundamentals_still_works(self):
        tech_data = {"rsi_14": 50.0}
        result = _generate_ticker_narrative(
            "SPY", tech_data, 500.0, "neutral", 0.5,
            fundamentals=None,
        )
        assert "SPY" in result
        assert "Neutral" in result

    def test_near_resistance(self):
        tech_data = {"r1": 101.0, "s1": 90.0}
        result = _generate_ticker_narrative("TEST", tech_data, 100.0, "bullish", 0.6)
        assert "R1 resistance" in result

    def test_near_support(self):
        tech_data = {"r1": 110.0, "s1": 100.5}
        result = _generate_ticker_narrative("TEST", tech_data, 100.0, "bearish", 0.6)
        assert "S1 support" in result
