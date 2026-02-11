"""Pre-computed analysis pipeline and cache query tools.

The pipeline runs 2-3x daily (via scheduler) and batch-computes all
technical indicators + quant metrics for the entire watchlist, storing
results in the `technical_indicators` and `quant_metrics` tables.

The cache query tools are used by the chat agent to retrieve pre-computed
data instantly instead of running live analysis every time.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import date, datetime

import numpy as np
import pandas as pd
import yfinance as yf
from agents import RunContextWrapper, function_tool

from portfolio_advisor.agents.context import AppContext
from portfolio_advisor.config import get_settings
from portfolio_advisor.db import queries
from portfolio_advisor.db.connection import get_db
from portfolio_advisor.tools.market_data import CRYPTO_MAP, _fetch_crypto_ohlcv
from portfolio_advisor.tools.quant_models import (
    compute_factor_exposures_raw,
    compute_return_forecast_raw,
    compute_vol_forecast_raw,
    detect_regime_raw,
)
from portfolio_advisor.tools.advanced_quant import (
    compute_garch_raw,
    compute_kalman_beta_raw,
    detect_regime_hmm_raw,
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
from portfolio_advisor.tools.technical_indicators import (
    _prices_to_series,
    compute_atr_bollinger_raw,
    compute_macd_raw,
    compute_rsi_raw,
    compute_sma_ema_raw,
    compute_support_resistance_raw,
)

logger = logging.getLogger(__name__)


# ── Helper: build DataFrame from yfinance multi-ticker download ──────────────


def _extract_ticker_df(
    bulk_df: pd.DataFrame,
    ticker: str,
    is_single: bool,
) -> pd.DataFrame | None:
    """Extract a single-ticker OHLCV DataFrame from a yfinance bulk download."""
    try:
        if is_single:
            sub = bulk_df.dropna()
        else:
            sub = bulk_df[ticker].dropna()
        if sub.empty:
            return None
        # Normalize column names to lowercase
        df = pd.DataFrame({
            "date": [idx.strftime("%Y-%m-%d") for idx in sub.index],
            "open": sub["Open"].values.astype(float),
            "high": sub["High"].values.astype(float),
            "low": sub["Low"].values.astype(float),
            "close": sub["Close"].values.astype(float),
            "volume": sub["Volume"].values.astype(float) if "Volume" in sub.columns else 0.0,
        })
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        return df
    except (KeyError, AttributeError):
        return None


def _synthesize_bias(tech_signals: list[dict]) -> tuple[str, float, str]:
    """Synthesize an overall bias from multiple indicator signals.

    Returns (bias, confidence, narrative).
    """
    bullish = 0
    bearish = 0
    total_conf = 0.0

    for sig in tech_signals:
        interp = sig.get("interpretation", "neutral")
        conf = sig.get("confidence", 0.5)
        total_conf += conf
        if interp == "bullish":
            bullish += 1
        elif interp == "bearish":
            bearish += 1

    n = len(tech_signals)
    avg_conf = round(total_conf / n, 2) if n > 0 else 0.5

    if bullish > bearish and bullish >= n * 0.6:
        bias = "bullish"
        confidence = min(0.95, avg_conf + 0.05)
    elif bearish > bullish and bearish >= n * 0.6:
        bias = "bearish"
        confidence = min(0.95, avg_conf + 0.05)
    elif bullish > bearish:
        bias = "slightly_bullish"
        confidence = avg_conf
    elif bearish > bullish:
        bias = "slightly_bearish"
        confidence = avg_conf
    else:
        bias = "neutral"
        confidence = max(0.4, avg_conf - 0.1)

    parts = []
    for sig in tech_signals:
        name = sig.get("_indicator", "indicator")
        interp = sig.get("interpretation", "neutral")
        parts.append(f"{name}={interp}")
    narrative = f"Signals: {', '.join(parts)}. Overall: {bias} (conf={confidence:.2f})."

    return bias, round(confidence, 2), narrative


# ── Core pipeline (called by scheduler, not a tool) ─────────────────────────


async def run_precompute_pipeline(ctx: AppContext) -> dict:
    """Batch-compute all indicators for the entire watchlist.

    1. Fetch OHLCV for all tickers + SPY benchmark
    2. For each ticker: compute technical indicators + quant metrics
    3. Compute portfolio-level risk metrics
    4. Track via analysis_runs table

    Returns a summary dict with processed/failed counts.
    """
    run_id = str(uuid.uuid4())
    settings = get_settings()
    today = date.today().isoformat()

    # Get watchlist and portfolio
    async with get_db(ctx.db_path) as db:
        prefs = await queries.get_user_preferences(db)
        positions = await queries.get_portfolio_state(db)

    watchlist = prefs.get("watchlist", settings.default_watchlist)
    if not watchlist:
        watchlist = settings.default_watchlist

    # Track the run
    async with get_db(ctx.db_path) as db:
        await queries.create_analysis_run(db, run_id, "precompute", watchlist)

    results = {"run_id": run_id, "processed": [], "failed": [], "skipped": []}

    # ── Step 1: Batch fetch OHLCV ────────────────────────────────────────
    equity_tickers = [t for t in watchlist if t.upper() not in CRYPTO_MAP]
    crypto_tickers = [t for t in watchlist if t.upper() in CRYPTO_MAP]

    # Always include SPY for factor exposure calculations
    all_equity = list(set(equity_tickers + ["SPY"]))

    ticker_dfs: dict[str, pd.DataFrame] = {}

    # Fetch equity data in batch
    if all_equity:
        try:
            bulk_df = yf.download(
                all_equity,
                period="1y",
                interval="1d",
                group_by="ticker" if len(all_equity) > 1 else None,
                progress=False,
            )
            is_single = len(all_equity) == 1
            for t in all_equity:
                df = _extract_ticker_df(bulk_df, t, is_single)
                if df is not None and len(df) >= 30:
                    ticker_dfs[t] = df
                else:
                    logger.warning(f"Insufficient data for {t}, skipping")
                    if t != "SPY":
                        results["skipped"].append(t)
        except Exception as e:
            logger.exception(f"Bulk equity download failed: {e}")

    # Fetch crypto data individually
    for t in crypto_tickers:
        try:
            coin_id = CRYPTO_MAP.get(t.upper())
            if not coin_id:
                results["skipped"].append(t)
                continue
            bars = await _fetch_crypto_ohlcv(coin_id, 365)
            if bars and len(bars) >= 30:
                bars_json = json.dumps(bars)
                ticker_dfs[t] = _prices_to_series(bars_json)
            else:
                results["skipped"].append(t)
        except Exception as e:
            logger.warning(f"Crypto fetch failed for {t}: {e}")
            results["skipped"].append(t)

    # Get SPY returns for factor exposures
    spy_df = ticker_dfs.get("SPY")
    spy_returns = None
    if spy_df is not None:
        spy_returns = spy_df["close"].pct_change().dropna()

    # ── Step 2: Compute per-ticker indicators ────────────────────────────
    for ticker in watchlist:
        df = ticker_dfs.get(ticker)
        if df is None or len(df) < 30:
            results["failed"].append(ticker)
            continue

        try:
            # ── Technical indicators ──
            sma_ema = compute_sma_ema_raw(df)
            sma_ema["_indicator"] = "sma_ema"
            rsi = compute_rsi_raw(df)
            rsi["_indicator"] = "rsi"
            macd = compute_macd_raw(df)
            macd["_indicator"] = "macd"
            atr_bb = compute_atr_bollinger_raw(df)
            atr_bb["_indicator"] = "atr_bollinger"
            sr = compute_support_resistance_raw(df)
            sr["_indicator"] = "support_resistance"

            # ── Advanced technical indicators ──
            ichimoku = compute_ichimoku_raw(df)
            ichimoku["_indicator"] = "ichimoku"
            vwap_data = compute_vwap_raw(df)
            vwap_data["_indicator"] = "vwap"
            obv_data = compute_obv_raw(df)
            obv_data["_indicator"] = "obv"
            adx_data = compute_adx_dmi_raw(df)
            adx_data["_indicator"] = "adx_dmi"
            stoch = compute_stochastic_raw(df)
            stoch["_indicator"] = "stochastic"
            fib = compute_fibonacci_raw(df)
            fib["_indicator"] = "fibonacci"
            vol_profile = compute_volume_profile_raw(df)
            vol_profile["_indicator"] = "volume_profile"

            # Synthesize overall bias from ALL 12 indicators
            all_signals = [
                sma_ema, rsi, macd, atr_bb, sr,
                ichimoku, vwap_data, obv_data, adx_data, stoch, fib, vol_profile,
            ]
            bias, confidence, narrative = _synthesize_bias(all_signals)

            tech_data = {
                "ticker": ticker,
                "indicator_date": today,
                "run_id": run_id,
                # Core indicators
                "sma50": sma_ema["sma50"],
                "sma200": sma_ema["sma200"],
                "ema12": sma_ema["ema12"],
                "ema26": sma_ema["ema26"],
                "rsi_14": rsi["rsi"],
                "macd_line": macd["macd_line"],
                "macd_signal": macd["signal_line"],
                "macd_histogram": macd["histogram"],
                "atr_14": atr_bb["atr_14"],
                "bb_upper": atr_bb["bb_upper"],
                "bb_lower": atr_bb["bb_lower"],
                "bb_bandwidth": atr_bb["bandwidth"],
                "bb_pct_b": atr_bb["pct_b"],
                "pivot": sr["pivot"],
                "r1": sr["r1"],
                "r2": sr["r2"],
                "s1": sr["s1"],
                "s2": sr["s2"],
                # Advanced indicators
                "ichimoku_tenkan": ichimoku["tenkan"],
                "ichimoku_kijun": ichimoku["kijun"],
                "ichimoku_senkou_a": ichimoku["senkou_a"],
                "ichimoku_senkou_b": ichimoku["senkou_b"],
                "ichimoku_chikou": ichimoku["chikou"],
                "vwap": vwap_data["vwap"],
                "obv": obv_data["obv"],
                "adx": adx_data["adx"],
                "stochastic_k": stoch["k"],
                "stochastic_d": stoch["d"],
                "fib_levels": json.dumps(fib["levels"]),
                # Synthesis
                "overall_bias": bias,
                "overall_confidence": confidence,
                "narrative": narrative,
            }

            async with get_db(ctx.db_path) as db:
                await queries.store_technical_indicators(db, tech_data)

            # ── Quant metrics ──
            returns = df["close"].pct_change().dropna()

            forecast = compute_return_forecast_raw(df)
            vol = compute_vol_forecast_raw(df)
            regime = detect_regime_raw(df)

            # Factor exposures (needs SPY)
            beta_val = alpha_val = r_sq_val = None
            if spy_returns is not None:
                common = returns.index.intersection(spy_returns.index)
                if len(common) >= 30:
                    factor_raw = compute_factor_exposures_raw(
                        returns.loc[common].values,
                        spy_returns.loc[common].values,
                    )
                    if "error" not in factor_raw:
                        beta_val = factor_raw["market_beta"]
                        alpha_val = factor_raw["alpha_daily"]
                        r_sq_val = factor_raw["r_squared"]

            # ── Advanced quant models ──
            garch_vol = None
            if len(returns) >= 100:
                garch_result = compute_garch_raw(returns.values)
                if "error" not in garch_result:
                    garch_vol = garch_result.get("current_cond_vol_annualized")

            hmm_state = None
            if len(returns) >= 100:
                hmm_result = detect_regime_hmm_raw(returns.values)
                if "error" not in hmm_result:
                    hmm_state = hmm_result.get("current_state")

            kalman_beta = None
            if spy_returns is not None:
                common_k = returns.index.intersection(spy_returns.index)
                if len(common_k) >= 60:
                    kalman_result = compute_kalman_beta_raw(
                        returns.loc[common_k].values,
                        spy_returns.loc[common_k].values,
                    )
                    if "error" not in kalman_result:
                        kalman_beta = kalman_result.get("current_beta")

            ff3_betas = None
            if spy_returns is not None and ticker_dfs.get("SPY") is not None:
                # We need IWM, IWD, IWF for factor proxies — skip if not available
                # (These are fetched only if in watchlist; store None otherwise)
                pass  # FF3 requires additional ETF data not in batch download

            # Distribution metrics (inline — simple stats)
            skewness = float(returns.skew()) if len(returns) >= 30 else None
            kurtosis = float(returns.kurtosis()) if len(returns) >= 30 else None

            # Performance metrics (inline)
            ann_return = float(returns.mean()) * 252
            ann_vol = float(returns.std()) * np.sqrt(252)
            sharpe = (ann_return - 0.05) / ann_vol if ann_vol > 0 else 0.0

            downside = returns[returns < 0]
            downside_std = float(downside.std()) * np.sqrt(252) if len(downside) > 0 else ann_vol
            sortino = (ann_return - 0.05) / downside_std if downside_std > 0 else 0.0

            cum_ret = (1 + returns).cumprod()
            max_dd = float(((cum_ret / cum_ret.cummax()) - 1).min())
            calmar = ann_return / abs(max_dd) if max_dd < 0 else 0.0

            # Build quant data
            quant_data = {
                "ticker": ticker,
                "metric_date": today,
                "run_id": run_id,
                "ewma_vol": vol.get("ewma_vol_annualized") if "error" not in vol else None,
                "vol_regime": vol.get("vol_regime") if "error" not in vol else None,
                "vol_percentile": vol.get("vol_percentile_1y") if "error" not in vol else None,
                "hurst": regime.get("hurst_exponent") if "error" not in regime else None,
                "regime": regime.get("regime") if "error" not in regime else None,
                "regime_confidence": regime.get("confidence") if "error" not in regime else None,
                "beta": beta_val,
                "alpha": alpha_val,
                "r_squared": r_sq_val,
                "skewness": round(skewness, 3) if skewness is not None else None,
                "kurtosis": round(kurtosis, 3) if kurtosis is not None else None,
                "sharpe": round(sharpe, 3),
                "sortino": round(sortino, 3),
                "calmar": round(calmar, 3),
                # Advanced quant
                "garch_vol": round(garch_vol, 2) if garch_vol is not None else None,
                "hmm_state": hmm_state,
                "kalman_beta": round(kalman_beta, 4) if kalman_beta is not None else None,
                "ff3_betas": ff3_betas,
            }

            # Forecasts (may have error)
            if "error" not in forecast:
                fc = forecast["forecasts"]
                quant_data["return_1w_pct"] = fc["1w"]["expected_return_pct"]
                quant_data["return_1w_ci_low"] = fc["1w"]["ci_low_pct"]
                quant_data["return_1w_ci_high"] = fc["1w"]["ci_high_pct"]
                quant_data["return_1m_pct"] = fc["1m"]["expected_return_pct"]
                quant_data["return_1m_ci_low"] = fc["1m"]["ci_low_pct"]
                quant_data["return_1m_ci_high"] = fc["1m"]["ci_high_pct"]
                quant_data["return_3m_pct"] = fc["3m"]["expected_return_pct"]
                quant_data["return_3m_ci_low"] = fc["3m"]["ci_low_pct"]
                quant_data["return_3m_ci_high"] = fc["3m"]["ci_high_pct"]

            async with get_db(ctx.db_path) as db:
                await queries.store_quant_metrics(db, quant_data)

            results["processed"].append(ticker)

        except Exception as e:
            logger.exception(f"Pre-compute failed for {ticker}: {e}")
            results["failed"].append(ticker)

    # ── Step 3: Portfolio-level risk metrics ──────────────────────────────
    if positions:
        try:
            weights = {p["ticker"]: p["weight_pct"] / 100.0 for p in positions}
            total_weight = sum(weights.values())

            # Weighted portfolio returns
            port_returns = pd.Series(dtype=float)
            for t, w in weights.items():
                if t in ticker_dfs:
                    t_ret = ticker_dfs[t]["close"].pct_change().dropna()
                    if port_returns.empty:
                        port_returns = t_ret * w
                    else:
                        common_idx = port_returns.index.intersection(t_ret.index)
                        port_returns = port_returns.loc[common_idx] + t_ret.loc[common_idx] * w

            if len(port_returns) >= 30:
                var_95 = float(np.percentile(port_returns, 5))
                tail = port_returns[port_returns <= var_95]
                es_95 = float(tail.mean()) if len(tail) > 0 else var_95

                cum_port = (1 + port_returns).cumprod()
                max_dd_port = float(((cum_port / cum_port.cummax()) - 1).min())
                current_dd = float(((cum_port / cum_port.cummax()) - 1).iloc[-1])

                # Portfolio beta
                portfolio_beta = None
                if spy_returns is not None:
                    common_idx = port_returns.index.intersection(spy_returns.index)
                    if len(common_idx) >= 30:
                        cov = np.cov(
                            port_returns.loc[common_idx].values,
                            spy_returns.loc[common_idx].values,
                        )
                        portfolio_beta = round(float(cov[0, 1] / cov[1, 1]), 3)

                # Asset class breakdown
                bond_set = {"TLT", "IEF", "HYG", "AGG", "BND", "LQD"}
                commodity_set = {"GLD", "SLV", "USO", "DBA"}
                crypto_set = set(CRYPTO_MAP.keys())
                asset_class_pcts = {
                    "equity": sum(w for t, w in weights.items()
                                  if t not in bond_set | commodity_set | crypto_set) * 100,
                    "bond": sum(w for t, w in weights.items() if t in bond_set) * 100,
                    "commodity": sum(w for t, w in weights.items() if t in commodity_set) * 100,
                    "crypto": sum(w for t, w in weights.items() if t in crypto_set) * 100,
                    "cash": round((1 - total_weight) * 100, 2),
                }

                risk_data = {
                    "risk_date": today,
                    "run_id": run_id,
                    "var_95": round(var_95 * 100, 3),
                    "es_95": round(es_95 * 100, 3),
                    "max_drawdown": round(max_dd_port * 100, 2),
                    "current_drawdown": round(current_dd * 100, 2),
                    "portfolio_beta": portfolio_beta,
                    "asset_class_pcts": asset_class_pcts,
                }

                async with get_db(ctx.db_path) as db:
                    await queries.store_daily_risk_metrics(db, risk_data)

        except Exception as e:
            logger.exception(f"Portfolio risk computation failed: {e}")

    # ── Step 4: Complete the run ─────────────────────────────────────────
    status = "completed" if not results["failed"] else "completed"
    error_msg = None
    if results["failed"]:
        error_msg = f"Failed tickers: {', '.join(results['failed'])}"

    async with get_db(ctx.db_path) as db:
        await queries.complete_analysis_run(db, run_id, status, error_msg)

    results["summary"] = (
        f"Processed {len(results['processed'])}/{len(watchlist)} tickers. "
        f"Failed: {len(results['failed'])}. Skipped: {len(results['skipped'])}."
    )
    logger.info(results["summary"])
    return results


# ── @function_tool cache query tools (used by chat agent) ───────────────────


@function_tool
async def get_cached_technical(
    ctx: RunContextWrapper[AppContext],
    ticker: str,
) -> str:
    """Get the latest pre-computed technical indicators for a ticker.

    Returns cached SMA, EMA, RSI, MACD, Bollinger Bands, support/resistance,
    plus an overall bias/confidence and freshness indicator.
    Use this before running live technical analysis — if data is fresh, no need to recompute.
    """
    settings = get_settings()
    async with get_db(ctx.context.db_path) as db:
        data = await queries.get_technical_indicators(db, ticker)

    if data is None:
        return json.dumps({"ticker": ticker, "cached": False, "message": "No cached data"})

    # Check freshness
    indicator_date = data.get("indicator_date", "")
    try:
        dt = datetime.strptime(indicator_date, "%Y-%m-%d")
        age_hours = (datetime.utcnow() - dt).total_seconds() / 3600
        is_fresh = age_hours < settings.precompute_stale_hours
    except (ValueError, TypeError):
        age_hours = 999
        is_fresh = False

    data["cached"] = True
    data["is_fresh"] = is_fresh
    data["age_hours"] = round(age_hours, 1)
    return json.dumps(data, default=str)


@function_tool
async def get_cached_quant(
    ctx: RunContextWrapper[AppContext],
    ticker: str,
) -> str:
    """Get the latest pre-computed quant metrics for a ticker.

    Returns cached return forecasts, volatility, regime, factor exposures,
    risk-adjusted returns, plus freshness indicator.
    Use this before running live quant analysis — if data is fresh, no need to recompute.
    """
    settings = get_settings()
    async with get_db(ctx.context.db_path) as db:
        data = await queries.get_quant_metrics(db, ticker)

    if data is None:
        return json.dumps({"ticker": ticker, "cached": False, "message": "No cached data"})

    metric_date = data.get("metric_date", "")
    try:
        dt = datetime.strptime(metric_date, "%Y-%m-%d")
        age_hours = (datetime.utcnow() - dt).total_seconds() / 3600
        is_fresh = age_hours < settings.precompute_stale_hours
    except (ValueError, TypeError):
        age_hours = 999
        is_fresh = False

    data["cached"] = True
    data["is_fresh"] = is_fresh
    data["age_hours"] = round(age_hours, 1)
    return json.dumps(data, default=str)


@function_tool
async def get_cached_bulk_summary(
    ctx: RunContextWrapper[AppContext],
    tickers: str,
) -> str:
    """Get a condensed summary of pre-computed data for multiple tickers.

    Returns a compact view: signal, confidence, RSI, regime, vol for each ticker.
    Ideal for quick multi-ticker overview. Pass tickers as comma-separated string.
    """
    ticker_list = [t.strip().upper() for t in tickers.split(",")]
    settings = get_settings()

    summaries = []
    async with get_db(ctx.context.db_path) as db:
        for ticker in ticker_list:
            tech = await queries.get_technical_indicators(db, ticker)
            quant = await queries.get_quant_metrics(db, ticker)

            summary = {"ticker": ticker}

            if tech:
                summary["signal"] = tech.get("overall_bias")
                summary["confidence"] = tech.get("overall_confidence")
                summary["rsi"] = tech.get("rsi_14")
                summary["sma50"] = tech.get("sma50")
                summary["sma200"] = tech.get("sma200")
                summary["indicator_date"] = tech.get("indicator_date")
            else:
                summary["signal"] = None
                summary["message"] = "No technical data"

            if quant:
                summary["regime"] = quant.get("regime")
                summary["vol_regime"] = quant.get("vol_regime")
                summary["ewma_vol"] = quant.get("ewma_vol")
                summary["return_1w_pct"] = quant.get("return_1w_pct")
                summary["sharpe"] = quant.get("sharpe")
                summary["beta"] = quant.get("beta")

            summaries.append(summary)

    # Check freshness based on most recent indicator date
    dates = [s.get("indicator_date") for s in summaries if s.get("indicator_date")]
    is_fresh = False
    if dates:
        try:
            latest = max(datetime.strptime(d, "%Y-%m-%d") for d in dates)
            age_hours = (datetime.utcnow() - latest).total_seconds() / 3600
            is_fresh = age_hours < settings.precompute_stale_hours
        except (ValueError, TypeError):
            pass

    return json.dumps({
        "tickers_requested": len(ticker_list),
        "tickers_with_data": sum(1 for s in summaries if s.get("signal") is not None),
        "is_fresh": is_fresh,
        "summaries": summaries,
    })


@function_tool
async def check_data_freshness(
    ctx: RunContextWrapper[AppContext],
) -> str:
    """Check when the last pre-compute pipeline ran and whether data is fresh.

    Returns the latest analysis run status, age, and whether a re-compute is needed.
    Call this before deciding whether to use cached data or run live analysis.
    """
    settings = get_settings()
    async with get_db(ctx.context.db_path) as db:
        run = await queries.get_latest_analysis_run(db, "precompute")

    if run is None:
        return json.dumps({
            "has_run": False,
            "is_fresh": False,
            "message": "No pre-compute runs found. Data must be computed live.",
        })

    completed_at = run.get("completed_at")
    status = run.get("status")
    is_fresh = False
    age_hours = None

    if completed_at:
        try:
            dt = datetime.strptime(completed_at, "%Y-%m-%d %H:%M:%S")
            age_hours = (datetime.utcnow() - dt).total_seconds() / 3600
            is_fresh = age_hours < settings.precompute_stale_hours
        except (ValueError, TypeError):
            pass

    return json.dumps({
        "has_run": True,
        "is_fresh": is_fresh,
        "last_run_id": run.get("run_id"),
        "last_status": status,
        "completed_at": completed_at,
        "age_hours": round(age_hours, 1) if age_hours is not None else None,
        "stale_threshold_hours": settings.precompute_stale_hours,
        "tickers_processed": run.get("tickers_processed"),
        "duration_seconds": run.get("duration_seconds"),
    })


@function_tool
async def get_signal_history(
    ctx: RunContextWrapper[AppContext],
    ticker: str,
    days: int = 7,
) -> str:
    """Get the signal trend for a ticker over the last N days.

    Shows how the overall bias and confidence have changed over time.
    Useful for detecting signal reversals or strengthening trends.
    """
    async with get_db(ctx.context.db_path) as db:
        trend = await queries.get_signal_trend(db, ticker, days)

    if not trend:
        return json.dumps({
            "ticker": ticker,
            "days": days,
            "has_history": False,
            "message": "No signal history found",
        })

    # Analyze trend direction
    biases = [t["overall_bias"] for t in trend if t.get("overall_bias")]
    confidences = [t["overall_confidence"] for t in trend if t.get("overall_confidence")]

    # Determine if signal is consistent, reversing, or strengthening
    consistency = "insufficient_data"
    if len(biases) >= 3:
        bullish_count = sum(1 for b in biases if b in ("bullish", "slightly_bullish"))
        bearish_count = sum(1 for b in biases if b in ("bearish", "slightly_bearish"))
        if bullish_count >= len(biases) * 0.7:
            consistency = "consistently_bullish"
        elif bearish_count >= len(biases) * 0.7:
            consistency = "consistently_bearish"
        elif biases[-1] != biases[0]:
            consistency = "signal_reversal"
        else:
            consistency = "mixed"

    # Confidence trend
    conf_trend = "stable"
    if len(confidences) >= 3:
        if confidences[-1] > confidences[0] + 0.1:
            conf_trend = "strengthening"
        elif confidences[-1] < confidences[0] - 0.1:
            conf_trend = "weakening"

    return json.dumps({
        "ticker": ticker,
        "days": days,
        "has_history": True,
        "data_points": len(trend),
        "history": trend,
        "analysis": {
            "consistency": consistency,
            "confidence_trend": conf_trend,
            "latest_bias": biases[-1] if biases else None,
            "latest_confidence": confidences[-1] if confidences else None,
        },
    })
