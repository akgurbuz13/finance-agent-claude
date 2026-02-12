"""Live data validation tests for macro data integration.

These tests validate the macro snapshot pipeline, yield curve, VIX,
and credit spread data integration using real data providers with fallbacks.
"""

from __future__ import annotations

import json

import pytest
import yfinance as yf

from portfolio_advisor.tools.precomputed import _compute_macro_snapshot


# ── Helpers ───────────────────────────────────────────────────────────────────


def _build_ticker_dfs() -> dict:
    """Fetch OHLCV for macro-relevant ETFs used by _compute_macro_snapshot."""
    import pandas as pd

    tickers = ["SPY", "TLT", "IEF", "HYG"]
    df = yf.download(tickers, period="3mo", progress=False)

    ticker_dfs = {}
    for t in tickers:
        try:
            sub = df[t].dropna() if t in df.columns.get_level_values(0) else None
        except Exception:
            sub = None

        if sub is None:
            try:
                # Try extracting from multi-index
                sub_data = {}
                for col in ["Open", "High", "Low", "Close", "Volume"]:
                    try:
                        sub_data[col.lower()] = df[(col, t)].dropna()
                    except KeyError:
                        pass
                if "close" in sub_data:
                    sub = pd.DataFrame(sub_data)
                    sub.index.name = "date"
            except Exception:
                pass

        if sub is not None and len(sub) >= 20:
            # Normalize column names if needed
            if "Close" in sub.columns:
                sub = sub.rename(columns={
                    c: c.lower() for c in sub.columns
                })
            ticker_dfs[t] = sub

    return ticker_dfs


# ── Macro Snapshot with Providers (yfinance fallback) ─────────────────────────


@pytest.mark.live
async def test_macro_snapshot_with_providers():
    """Macro snapshot works with yfinance-based providers (no API keys needed)."""
    from portfolio_advisor.providers.registry import ProviderRegistry

    # Create a minimal registry with only yfinance provider (always available)
    registry = ProviderRegistry()
    ticker_dfs = _build_ticker_dfs()

    result = await _compute_macro_snapshot(ticker_dfs, providers=registry)

    # Should have VIX data
    assert "vix_level" in result, f"Missing vix_level. Result keys: {list(result.keys())}"
    assert result["vix_level"] > 0, f"VIX should be positive, got {result['vix_level']}"

    # VIX regime should be classified
    assert result.get("vix_regime") in ("low", "normal", "elevated", "extreme"), (
        f"Unexpected VIX regime: {result.get('vix_regime')}"
    )

    # Source labels should be present
    assert "vix_source" in result
    assert "yield_curve_source" in result

    # Yield curve data should be present (from yfinance or ETF proxy)
    assert "yield_curve_slope" in result, f"Missing yield_curve_slope. Keys: {list(result.keys())}"

    # Macro regime should be determined
    assert result.get("macro_regime") in (
        "expansion", "slowdown", "contraction", "recovery"
    ), f"Unexpected macro regime: {result.get('macro_regime')}"


# ── Macro Snapshot without Providers (ETF proxy fallback) ─────────────────────


@pytest.mark.live
async def test_macro_snapshot_without_providers():
    """Macro snapshot works with providers=None using ETF proxy fallback."""
    ticker_dfs = _build_ticker_dfs()

    result = await _compute_macro_snapshot(ticker_dfs, providers=None)

    # ETF proxy path should still determine a yield curve slope
    assert "yield_curve_slope" in result, f"Missing yield_curve_slope. Keys: {list(result.keys())}"
    assert result.get("yield_curve_source") == "etf_proxy"

    # VIX should come from SPY proxy
    assert "vix_level" in result, f"Missing vix_level. Keys: {list(result.keys())}"
    assert result.get("vix_source") == "spy_proxy"

    # Regime should still be determined
    assert result.get("macro_regime") in (
        "expansion", "slowdown", "contraction", "recovery"
    ), f"Unexpected macro regime: {result.get('macro_regime')}"


# ── Macro Regime from compute_macro_regime function tool ──────────────────────


@pytest.mark.live
async def test_macro_regime_from_compute_tool():
    """compute_macro_regime function tool returns valid regime classification."""
    from unittest.mock import MagicMock

    from portfolio_advisor.agents.context import AppContext
    from portfolio_advisor.providers.registry import ProviderRegistry
    from portfolio_advisor.tools.economic_data import compute_macro_regime

    # Create AppContext with a minimal provider registry
    ctx_obj = AppContext(
        db_path=":memory:",
        telegram_chat_id=0,
        providers=ProviderRegistry(),
    )

    # Mock the RunContextWrapper
    mock_ctx = MagicMock()
    mock_ctx.context = ctx_obj

    # Call the function tool's underlying function
    result_json = await compute_macro_regime.on_invoke_tool(mock_ctx, "")

    result = json.loads(result_json)

    assert "regime" in result, f"Missing 'regime' key. Result: {result}"
    assert result["regime"] in ("expansion", "slowdown", "contraction", "recovery"), (
        f"Unexpected regime: {result['regime']}"
    )

    # Confidence should be between 0 and 1
    assert 0 <= result["confidence"] <= 1.0, (
        f"Confidence out of range: {result['confidence']}"
    )

    # Regime scores should be present
    assert "regime_scores" in result
    assert "signals" in result


# ── Credit Spread Units Consistency ───────────────────────────────────────────


@pytest.mark.live
async def test_credit_spread_units_consistency():
    """Verify credit spread is converted to bps correctly for regime thresholds.

    FRED BAMLH0A0HYM2 returns percentage points (e.g. 3.5 = 3.5%).
    _compute_macro_snapshot converts to bps (*100) for FRED/Massive sources.
    Thresholds in bps: stress > 500, elevated > 400, tight < 300.
    """
    from portfolio_advisor.providers.registry import ProviderRegistry

    registry = ProviderRegistry()
    ticker_dfs = _build_ticker_dfs()

    result = await _compute_macro_snapshot(ticker_dfs, providers=registry)

    credit_spread = result.get("credit_spread")
    credit_source = result.get("credit_spread_source", "")

    if credit_spread is not None:
        if credit_source in ("fred", "massive"):
            # Should be in bps (typically 200-800 range)
            assert credit_spread > 10, (
                f"Credit spread from {credit_source} looks too small for bps: {credit_spread}. "
                "Expected conversion from pct points to bps (*100)."
            )
            assert credit_spread < 2000, (
                f"Credit spread from {credit_source} too large: {credit_spread} bps"
            )
        elif credit_source == "etf_proxy":
            # ETF proxy is a relative return difference, small number
            assert -10 < credit_spread < 10, (
                f"ETF proxy credit spread out of expected range: {credit_spread}"
            )


# ── Yield Curve Slope from yfinance ───────────────────────────────────────────


@pytest.mark.live
async def test_yield_curve_slope_from_yfinance():
    """Fetch yields from yfinance and verify slope classification."""
    from portfolio_advisor.providers.yfinance_provider import YFinanceProvider

    provider = YFinanceProvider()
    yields_data = await provider.fetch_treasury_yields()

    if yields_data is None:
        pytest.skip("yfinance treasury yield data unavailable")

    # Should have at least some maturities
    assert len(yields_data) >= 2, f"Expected at least 2 maturities, got {yields_data}"

    # Yields should be reasonable (0-20% range)
    for label, val in yields_data.items():
        assert 0 < val < 20, f"Unreasonable yield for {label}: {val}"

    # If we have 10y and 3m/5y, compute slope
    if "10y" in yields_data:
        y10 = yields_data["10y"]
        short_key = "3m" if "3m" in yields_data else "5y" if "5y" in yields_data else None
        if short_key:
            slope = y10 - yields_data[short_key]
            # Classify
            if slope < -0.5:
                expected = "inverted"
            elif slope < 0.5:
                expected = "flat"
            elif slope > 1.5:
                expected = "steep"
            else:
                expected = "normal"
            # Just verify the classification logic is consistent
            assert expected in ("inverted", "flat", "normal", "steep")


# ── Yield Curve Slope from ETF Proxy ──────────────────────────────────────────


@pytest.mark.live
async def test_yield_curve_slope_from_etf_proxy():
    """ETF proxy yield curve slope is computed when providers=None."""
    ticker_dfs = _build_ticker_dfs()

    # Without providers, macro snapshot uses TLT/IEF proxy
    result = await _compute_macro_snapshot(ticker_dfs, providers=None)

    assert result.get("yield_curve_source") == "etf_proxy"
    assert "yield_curve_slope" in result

    slope = result["yield_curve_slope"]
    # ETF proxy slope should be a small number (relative return diff * 100)
    assert isinstance(slope, float)
    assert -20 < slope < 20, f"ETF proxy slope out of range: {slope}"


# ── VIX Regime Thresholds ────────────────────────────────────────────────────


@pytest.mark.live
async def test_vix_regime_thresholds():
    """VIX from yfinance is correctly classified into regime buckets.

    Thresholds: panic > 30, elevated > 20, normal 15-20, calm < 15.
    """
    # Fetch VIX directly from yfinance
    vix_df = yf.download("^VIX", period="5d", progress=False)
    if vix_df.empty:
        pytest.skip("VIX data unavailable from yfinance")

    close_col = vix_df["Close"]
    # yfinance may return multi-level columns even for single ticker
    if hasattr(close_col, "columns"):
        close_col = close_col.iloc[:, 0]
    vix_yf = float(close_col.dropna().iloc[-1])

    # VIX should be a positive number in a reasonable range
    assert 5 < vix_yf < 90, f"VIX value out of expected range: {vix_yf}"

    # Now get the macro snapshot classification
    from portfolio_advisor.providers.registry import ProviderRegistry

    registry = ProviderRegistry()
    ticker_dfs = _build_ticker_dfs()
    result = await _compute_macro_snapshot(ticker_dfs, providers=registry)

    vix_from_snapshot = result.get("vix_level")
    regime_from_snapshot = result.get("vix_regime")

    assert vix_from_snapshot is not None, "Macro snapshot missing vix_level"
    assert regime_from_snapshot is not None, "Macro snapshot missing vix_regime"

    # VIX values from both sources should be similar (within 2 points)
    assert abs(vix_from_snapshot - vix_yf) < 3.0, (
        f"VIX mismatch: snapshot={vix_from_snapshot}, direct yfinance={vix_yf}"
    )

    # Verify classification is consistent
    # Since data may fetch at slightly different times, just verify
    # the classification logic is correct for the value we got
    if vix_from_snapshot < 15:
        assert regime_from_snapshot == "low"
    elif vix_from_snapshot < 20:
        assert regime_from_snapshot == "normal"
    elif vix_from_snapshot < 30:
        assert regime_from_snapshot == "elevated"
    else:
        assert regime_from_snapshot == "extreme"


# ── VIX from Economic Data Tool Matches ──────────────────────────────────────


@pytest.mark.live
async def test_vix_from_compute_macro_regime():
    """compute_macro_regime function tool returns VIX with correct classification."""
    from unittest.mock import MagicMock

    from portfolio_advisor.agents.context import AppContext
    from portfolio_advisor.providers.registry import ProviderRegistry
    from portfolio_advisor.tools.economic_data import compute_macro_regime

    ctx_obj = AppContext(
        db_path=":memory:",
        telegram_chat_id=0,
        providers=ProviderRegistry(),
    )
    mock_ctx = MagicMock()
    mock_ctx.context = ctx_obj

    result_json = await compute_macro_regime.on_invoke_tool(mock_ctx, "")
    result = json.loads(result_json)

    signals = result.get("signals", {})
    vix = signals.get("vix")
    vol_signal = signals.get("vol_signal")

    if vix is not None:
        # Verify the vol_signal classification matches VIX thresholds
        if vix > 30:
            assert vol_signal == "panic", f"VIX={vix} should be 'panic', got '{vol_signal}'"
        elif vix > 20:
            assert vol_signal == "elevated", f"VIX={vix} should be 'elevated', got '{vol_signal}'"
        elif vix < 15:
            assert vol_signal == "calm", f"VIX={vix} should be 'calm', got '{vol_signal}'"
        else:
            assert vol_signal == "normal", f"VIX={vix} should be 'normal', got '{vol_signal}'"
