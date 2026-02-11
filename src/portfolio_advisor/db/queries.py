"""Typed CRUD functions for all tables."""

from __future__ import annotations

import json
from datetime import datetime

import aiosqlite


# ── User Preferences ──────────────────────────────────────────────────────────

async def get_user_preferences(db: aiosqlite.Connection) -> dict:
    cursor = await db.execute("SELECT * FROM user_preferences WHERE id = 1")
    row = await cursor.fetchone()
    if row is None:
        return {}
    d = dict(row)
    for key in ("excluded_assets", "allowed_regions", "watchlist", "preferred_sectors"):
        if isinstance(d.get(key), str):
            try:
                d[key] = json.loads(d[key])
            except (json.JSONDecodeError, TypeError):
                d[key] = []
    # Convert SQLite integers to booleans for boolean fields
    for key in ("esg_filter", "tax_aware"):
        if key in d:
            d[key] = bool(d[key])
    return d


async def update_user_preference(db: aiosqlite.Connection, key: str, value) -> None:
    if key in ("excluded_assets", "allowed_regions", "watchlist", "preferred_sectors"):
        value = json.dumps(value)
    elif key in ("esg_filter", "tax_aware"):
        value = int(bool(value))
    await db.execute(
        f"UPDATE user_preferences SET {key} = ?, updated_at = ? WHERE id = 1",
        (value, datetime.utcnow().isoformat()),
    )
    await db.commit()


# ── Portfolio State ───────────────────────────────────────────────────────────

async def get_portfolio_state(db: aiosqlite.Connection) -> list[dict]:
    cursor = await db.execute("SELECT * FROM portfolio_state ORDER BY weight_pct DESC")
    rows = await cursor.fetchall()
    return [dict(r) for r in rows]


async def update_portfolio_position(
    db: aiosqlite.Connection, ticker: str, weight_pct: float, asset_class: str = "equity"
) -> None:
    now = datetime.utcnow().isoformat()
    await db.execute(
        """INSERT INTO portfolio_state (ticker, weight_pct, asset_class, updated_at)
           VALUES (?, ?, ?, ?)
           ON CONFLICT(ticker) DO UPDATE SET
               weight_pct = excluded.weight_pct,
               asset_class = excluded.asset_class,
               updated_at = excluded.updated_at""",
        (ticker, weight_pct, asset_class, now),
    )
    await db.commit()


async def remove_portfolio_position(db: aiosqlite.Connection, ticker: str) -> None:
    await db.execute("DELETE FROM portfolio_state WHERE ticker = ?", (ticker,))
    await db.commit()


async def snapshot_portfolio(
    db: aiosqlite.Connection, trigger: str = "manual"
) -> None:
    positions = await get_portfolio_state(db)
    await db.execute(
        "INSERT INTO portfolio_history (snapshot_date, state_json, trigger) VALUES (?, ?, ?)",
        (datetime.utcnow().strftime("%Y-%m-%d"), json.dumps(positions), trigger),
    )
    await db.commit()


async def get_portfolio_history(db: aiosqlite.Connection, days: int = 30) -> list[dict]:
    cursor = await db.execute(
        """SELECT * FROM portfolio_history
           WHERE snapshot_date >= date('now', ?)
           ORDER BY snapshot_date DESC""",
        (f"-{days} days",),
    )
    rows = await cursor.fetchall()
    return [dict(r) for r in rows]


# ── Daily Briefs ──────────────────────────────────────────────────────────────

async def store_daily_brief(db: aiosqlite.Connection, brief: dict) -> None:
    brief_date = brief["brief_date"]
    content_json = json.dumps(brief)

    await db.execute(
        """INSERT INTO daily_briefs (brief_date, content_json, market_summary, telegram_summary)
           VALUES (?, ?, ?, ?)
           ON CONFLICT(brief_date) DO UPDATE SET
               content_json = excluded.content_json,
               market_summary = excluded.market_summary,
               telegram_summary = excluded.telegram_summary""",
        (brief_date, content_json, brief.get("market_summary"), brief.get("telegram_summary")),
    )

    # Denormalize instrument briefs
    for inst in brief.get("instruments", []):
        await db.execute(
            """INSERT INTO instrument_briefs
               (brief_date, ticker, signal, confidence, what_happened,
                why_it_matters, technical_json, quant_json, sources)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(brief_date, ticker) DO UPDATE SET
                   signal = excluded.signal,
                   confidence = excluded.confidence,
                   what_happened = excluded.what_happened,
                   why_it_matters = excluded.why_it_matters,
                   technical_json = excluded.technical_json,
                   quant_json = excluded.quant_json,
                   sources = excluded.sources""",
            (
                brief_date,
                inst["ticker"],
                inst.get("signal"),
                inst.get("confidence"),
                inst.get("what_happened"),
                inst.get("why_it_matters"),
                json.dumps(inst.get("technical_json", {})),
                json.dumps(inst.get("quant_json", {})),
                json.dumps(inst.get("sources", [])),
            ),
        )
    await db.commit()


async def retrieve_daily_briefs(
    db: aiosqlite.Connection,
    start_date: str,
    end_date: str,
    ticker: str | None = None,
) -> list[dict]:
    if ticker:
        cursor = await db.execute(
            """SELECT * FROM instrument_briefs
               WHERE brief_date BETWEEN ? AND ? AND ticker = ?
               ORDER BY brief_date DESC""",
            (start_date, end_date, ticker),
        )
    else:
        cursor = await db.execute(
            """SELECT * FROM daily_briefs
               WHERE brief_date BETWEEN ? AND ?
               ORDER BY brief_date DESC""",
            (start_date, end_date),
        )
    rows = await cursor.fetchall()
    return [dict(r) for r in rows]


# ── Weekly Reports ────────────────────────────────────────────────────────────

async def store_weekly_report(db: aiosqlite.Connection, report: dict) -> None:
    week_ending = report["week_ending"]
    await db.execute(
        """INSERT INTO weekly_reports
           (week_ending, content_json, executive_summary, allocations_json)
           VALUES (?, ?, ?, ?)
           ON CONFLICT(week_ending) DO UPDATE SET
               content_json = excluded.content_json,
               executive_summary = excluded.executive_summary,
               allocations_json = excluded.allocations_json""",
        (
            week_ending,
            json.dumps(report),
            report.get("executive_summary"),
            json.dumps(report.get("allocations", [])),
        ),
    )
    await db.commit()


async def retrieve_weekly_reports(db: aiosqlite.Connection, count: int = 4) -> list[dict]:
    cursor = await db.execute(
        "SELECT * FROM weekly_reports ORDER BY week_ending DESC LIMIT ?",
        (count,),
    )
    rows = await cursor.fetchall()
    return [dict(r) for r in rows]


# ── Price Cache ───────────────────────────────────────────────────────────────

async def cache_prices(db: aiosqlite.Connection, ticker: str, bars: list[dict], source: str = "yfinance") -> None:
    for bar in bars:
        await db.execute(
            """INSERT INTO price_cache (ticker, bar_date, open, high, low, close, volume, source)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(ticker, bar_date) DO NOTHING""",
            (ticker, bar["date"], bar["open"], bar["high"], bar["low"], bar["close"], bar["volume"], source),
        )
    await db.commit()


async def get_cached_prices(
    db: aiosqlite.Connection, ticker: str, start_date: str, end_date: str
) -> list[dict]:
    cursor = await db.execute(
        """SELECT * FROM price_cache
           WHERE ticker = ? AND bar_date BETWEEN ? AND ?
           ORDER BY bar_date""",
        (ticker, start_date, end_date),
    )
    rows = await cursor.fetchall()
    return [dict(r) for r in rows]


# ── Forecasts Log ────────────────────────────────────────────────────────────

async def store_forecast(
    db: aiosqlite.Connection,
    ticker: str,
    forecast_type: str,
    horizon: str,
    predicted_value: dict,
) -> None:
    await db.execute(
        """INSERT INTO forecasts_log (forecast_date, ticker, forecast_type, horizon, predicted_value)
           VALUES (date('now'), ?, ?, ?, ?)""",
        (ticker, forecast_type, horizon, json.dumps(predicted_value)),
    )
    await db.commit()


async def query_forecasts_log(
    db: aiosqlite.Connection,
    ticker: str | None = None,
    forecast_type: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> list[dict]:
    query = "SELECT * FROM forecasts_log WHERE 1=1"
    params: list = []
    if ticker:
        query += " AND ticker = ?"
        params.append(ticker)
    if forecast_type:
        query += " AND forecast_type = ?"
        params.append(forecast_type)
    if start_date:
        query += " AND forecast_date >= ?"
        params.append(start_date)
    if end_date:
        query += " AND forecast_date <= ?"
        params.append(end_date)
    query += " ORDER BY forecast_date DESC"

    cursor = await db.execute(query, params)
    rows = await cursor.fetchall()
    return [dict(r) for r in rows]


# ── Token Usage ───────────────────────────────────────────────────────────────

async def log_token_usage(
    db: aiosqlite.Connection,
    model: str,
    run_type: str,
    input_tokens: int,
    output_tokens: int,
    estimated_cost_usd: float,
) -> None:
    await db.execute(
        """INSERT INTO token_usage (usage_date, model, run_type, input_tokens, output_tokens, estimated_cost_usd)
           VALUES (date('now'), ?, ?, ?, ?, ?)""",
        (model, run_type, input_tokens, output_tokens, estimated_cost_usd),
    )
    await db.commit()


async def get_token_usage_summary(
    db: aiosqlite.Connection, days: int = 30
) -> dict:
    cursor = await db.execute(
        """SELECT
               SUM(input_tokens) as total_input,
               SUM(output_tokens) as total_output,
               SUM(estimated_cost_usd) as total_cost,
               COUNT(*) as call_count
           FROM token_usage
           WHERE usage_date >= date('now', ?)""",
        (f"-{days} days",),
    )
    row = await cursor.fetchone()
    return dict(row) if row else {}


async def get_daily_token_usage(db: aiosqlite.Connection) -> int:
    cursor = await db.execute(
        "SELECT COALESCE(SUM(input_tokens + output_tokens), 0) FROM token_usage WHERE usage_date = date('now')"
    )
    (total,) = await cursor.fetchone()
    return total


# ── Technical Indicators (pre-computed) ──────────────────────────────────────

async def store_technical_indicators(db: aiosqlite.Connection, data: dict) -> None:
    """Upsert a row into technical_indicators."""
    await db.execute(
        """INSERT INTO technical_indicators (
               ticker, indicator_date, snapshot_hour, run_id,
               sma50, sma200, ema12, ema26, rsi_14,
               macd_line, macd_signal, macd_histogram,
               atr_14, bb_upper, bb_lower, bb_bandwidth, bb_pct_b,
               pivot, r1, r2, s1, s2,
               ichimoku_tenkan, ichimoku_kijun, ichimoku_senkou_a,
               ichimoku_senkou_b, ichimoku_chikou,
               vwap, obv, adx, stochastic_k, stochastic_d,
               fib_levels, overall_bias, overall_confidence, narrative
           ) VALUES (
               ?, ?, ?, ?,
               ?, ?, ?, ?, ?,
               ?, ?, ?,
               ?, ?, ?, ?, ?,
               ?, ?, ?, ?, ?,
               ?, ?, ?,
               ?, ?,
               ?, ?, ?, ?, ?,
               ?, ?, ?, ?
           )
           ON CONFLICT(ticker, indicator_date, snapshot_hour) DO UPDATE SET
               run_id = excluded.run_id,
               sma50 = excluded.sma50, sma200 = excluded.sma200,
               ema12 = excluded.ema12, ema26 = excluded.ema26,
               rsi_14 = excluded.rsi_14,
               macd_line = excluded.macd_line, macd_signal = excluded.macd_signal,
               macd_histogram = excluded.macd_histogram,
               atr_14 = excluded.atr_14,
               bb_upper = excluded.bb_upper, bb_lower = excluded.bb_lower,
               bb_bandwidth = excluded.bb_bandwidth, bb_pct_b = excluded.bb_pct_b,
               pivot = excluded.pivot, r1 = excluded.r1, r2 = excluded.r2,
               s1 = excluded.s1, s2 = excluded.s2,
               ichimoku_tenkan = excluded.ichimoku_tenkan,
               ichimoku_kijun = excluded.ichimoku_kijun,
               ichimoku_senkou_a = excluded.ichimoku_senkou_a,
               ichimoku_senkou_b = excluded.ichimoku_senkou_b,
               ichimoku_chikou = excluded.ichimoku_chikou,
               vwap = excluded.vwap, obv = excluded.obv, adx = excluded.adx,
               stochastic_k = excluded.stochastic_k, stochastic_d = excluded.stochastic_d,
               fib_levels = excluded.fib_levels,
               overall_bias = excluded.overall_bias,
               overall_confidence = excluded.overall_confidence,
               narrative = excluded.narrative""",
        (
            data["ticker"], data["indicator_date"], data.get("snapshot_hour", 6),
            data.get("run_id"),
            data.get("sma50"), data.get("sma200"),
            data.get("ema12"), data.get("ema26"), data.get("rsi_14"),
            data.get("macd_line"), data.get("macd_signal"), data.get("macd_histogram"),
            data.get("atr_14"),
            data.get("bb_upper"), data.get("bb_lower"),
            data.get("bb_bandwidth"), data.get("bb_pct_b"),
            data.get("pivot"), data.get("r1"), data.get("r2"),
            data.get("s1"), data.get("s2"),
            data.get("ichimoku_tenkan"), data.get("ichimoku_kijun"),
            data.get("ichimoku_senkou_a"), data.get("ichimoku_senkou_b"),
            data.get("ichimoku_chikou"),
            data.get("vwap"), data.get("obv"), data.get("adx"),
            data.get("stochastic_k"), data.get("stochastic_d"),
            json.dumps(data["fib_levels"]) if data.get("fib_levels") else None,
            data.get("overall_bias"), data.get("overall_confidence"),
            data.get("narrative"),
        ),
    )
    await db.commit()


async def get_technical_indicators(
    db: aiosqlite.Connection, ticker: str, date: str | None = None,
    snapshot_hour: int | None = None,
) -> dict | None:
    """Get the latest (or date-specific) technical indicators for a ticker.

    If snapshot_hour is None, returns the latest snapshot for the given date.
    """
    if date and snapshot_hour is not None:
        cursor = await db.execute(
            """SELECT * FROM technical_indicators
               WHERE ticker = ? AND indicator_date = ? AND snapshot_hour = ?""",
            (ticker, date, snapshot_hour),
        )
    elif date:
        cursor = await db.execute(
            """SELECT * FROM technical_indicators
               WHERE ticker = ? AND indicator_date = ?
               ORDER BY snapshot_hour DESC LIMIT 1""",
            (ticker, date),
        )
    else:
        cursor = await db.execute(
            """SELECT * FROM technical_indicators
               WHERE ticker = ?
               ORDER BY indicator_date DESC, snapshot_hour DESC LIMIT 1""",
            (ticker,),
        )
    row = await cursor.fetchone()
    if row is None:
        return None
    d = dict(row)
    if d.get("fib_levels") and isinstance(d["fib_levels"], str):
        try:
            d["fib_levels"] = json.loads(d["fib_levels"])
        except json.JSONDecodeError:
            pass
    return d


async def get_bulk_technical_indicators(
    db: aiosqlite.Connection, tickers: list[str], date: str | None = None
) -> list[dict]:
    """Get latest technical indicators for multiple tickers."""
    results = []
    for ticker in tickers:
        row = await get_technical_indicators(db, ticker, date)
        if row:
            results.append(row)
    return results


# ── Quant Metrics (pre-computed) ─────────────────────────────────────────────

async def store_quant_metrics(db: aiosqlite.Connection, data: dict) -> None:
    """Upsert a row into quant_metrics."""
    await db.execute(
        """INSERT INTO quant_metrics (
               ticker, metric_date, snapshot_hour, run_id,
               return_1w_pct, return_1w_ci_low, return_1w_ci_high,
               return_1m_pct, return_1m_ci_low, return_1m_ci_high,
               return_3m_pct, return_3m_ci_low, return_3m_ci_high,
               ewma_vol, vol_regime, vol_percentile,
               hurst, regime, regime_confidence,
               beta, alpha, r_squared,
               skewness, kurtosis, sharpe, sortino, calmar,
               garch_vol, hmm_state, kalman_beta, ff3_betas,
               cornish_fisher_var, evt_var
           ) VALUES (
               ?, ?, ?, ?,
               ?, ?, ?,
               ?, ?, ?,
               ?, ?, ?,
               ?, ?, ?,
               ?, ?, ?,
               ?, ?, ?,
               ?, ?, ?, ?, ?,
               ?, ?, ?, ?,
               ?, ?
           )
           ON CONFLICT(ticker, metric_date, snapshot_hour) DO UPDATE SET
               run_id = excluded.run_id,
               return_1w_pct = excluded.return_1w_pct,
               return_1w_ci_low = excluded.return_1w_ci_low,
               return_1w_ci_high = excluded.return_1w_ci_high,
               return_1m_pct = excluded.return_1m_pct,
               return_1m_ci_low = excluded.return_1m_ci_low,
               return_1m_ci_high = excluded.return_1m_ci_high,
               return_3m_pct = excluded.return_3m_pct,
               return_3m_ci_low = excluded.return_3m_ci_low,
               return_3m_ci_high = excluded.return_3m_ci_high,
               ewma_vol = excluded.ewma_vol,
               vol_regime = excluded.vol_regime,
               vol_percentile = excluded.vol_percentile,
               hurst = excluded.hurst,
               regime = excluded.regime,
               regime_confidence = excluded.regime_confidence,
               beta = excluded.beta, alpha = excluded.alpha,
               r_squared = excluded.r_squared,
               skewness = excluded.skewness, kurtosis = excluded.kurtosis,
               sharpe = excluded.sharpe, sortino = excluded.sortino,
               calmar = excluded.calmar,
               garch_vol = excluded.garch_vol,
               hmm_state = excluded.hmm_state,
               kalman_beta = excluded.kalman_beta,
               ff3_betas = excluded.ff3_betas,
               cornish_fisher_var = excluded.cornish_fisher_var,
               evt_var = excluded.evt_var""",
        (
            data["ticker"], data["metric_date"], data.get("snapshot_hour", 6),
            data.get("run_id"),
            data.get("return_1w_pct"), data.get("return_1w_ci_low"),
            data.get("return_1w_ci_high"),
            data.get("return_1m_pct"), data.get("return_1m_ci_low"),
            data.get("return_1m_ci_high"),
            data.get("return_3m_pct"), data.get("return_3m_ci_low"),
            data.get("return_3m_ci_high"),
            data.get("ewma_vol"), data.get("vol_regime"), data.get("vol_percentile"),
            data.get("hurst"), data.get("regime"), data.get("regime_confidence"),
            data.get("beta"), data.get("alpha"), data.get("r_squared"),
            data.get("skewness"), data.get("kurtosis"),
            data.get("sharpe"), data.get("sortino"), data.get("calmar"),
            data.get("garch_vol"), data.get("hmm_state"), data.get("kalman_beta"),
            json.dumps(data["ff3_betas"]) if data.get("ff3_betas") else None,
            data.get("cornish_fisher_var"), data.get("evt_var"),
        ),
    )
    await db.commit()


async def get_quant_metrics(
    db: aiosqlite.Connection, ticker: str, date: str | None = None,
    snapshot_hour: int | None = None,
) -> dict | None:
    """Get latest (or date-specific) quant metrics for a ticker."""
    if date and snapshot_hour is not None:
        cursor = await db.execute(
            """SELECT * FROM quant_metrics
               WHERE ticker = ? AND metric_date = ? AND snapshot_hour = ?""",
            (ticker, date, snapshot_hour),
        )
    elif date:
        cursor = await db.execute(
            """SELECT * FROM quant_metrics
               WHERE ticker = ? AND metric_date = ?
               ORDER BY snapshot_hour DESC LIMIT 1""",
            (ticker, date),
        )
    else:
        cursor = await db.execute(
            """SELECT * FROM quant_metrics
               WHERE ticker = ?
               ORDER BY metric_date DESC, snapshot_hour DESC LIMIT 1""",
            (ticker,),
        )
    row = await cursor.fetchone()
    if row is None:
        return None
    d = dict(row)
    if d.get("ff3_betas") and isinstance(d["ff3_betas"], str):
        try:
            d["ff3_betas"] = json.loads(d["ff3_betas"])
        except json.JSONDecodeError:
            pass
    return d


async def get_bulk_quant_metrics(
    db: aiosqlite.Connection, tickers: list[str], date: str | None = None
) -> list[dict]:
    """Get latest quant metrics for multiple tickers."""
    results = []
    for ticker in tickers:
        row = await get_quant_metrics(db, ticker, date)
        if row:
            results.append(row)
    return results


# ── Analysis Runs ────────────────────────────────────────────────────────────

async def create_analysis_run(
    db: aiosqlite.Connection, run_id: str, run_type: str, tickers: list[str]
) -> None:
    """Record the start of an analysis run."""
    await db.execute(
        """INSERT INTO analysis_runs (run_id, run_type, tickers_processed, status)
           VALUES (?, ?, ?, 'running')""",
        (run_id, run_type, json.dumps(tickers)),
    )
    await db.commit()


async def complete_analysis_run(
    db: aiosqlite.Connection,
    run_id: str,
    status: str = "completed",
    error: str | None = None,
) -> None:
    """Mark an analysis run as completed or failed with duration."""
    await db.execute(
        """UPDATE analysis_runs
           SET status = ?,
               completed_at = datetime('now'),
               duration_seconds = (
                   julianday(datetime('now')) - julianday(started_at)
               ) * 86400,
               error_message = ?
           WHERE run_id = ?""",
        (status, error, run_id),
    )
    await db.commit()


async def get_latest_analysis_run(
    db: aiosqlite.Connection, run_type: str
) -> dict | None:
    """Get the most recent analysis run of a given type."""
    cursor = await db.execute(
        """SELECT * FROM analysis_runs
           WHERE run_type = ?
           ORDER BY started_at DESC LIMIT 1""",
        (run_type,),
    )
    row = await cursor.fetchone()
    if row is None:
        return None
    d = dict(row)
    if d.get("tickers_processed") and isinstance(d["tickers_processed"], str):
        try:
            d["tickers_processed"] = json.loads(d["tickers_processed"])
        except json.JSONDecodeError:
            pass
    return d


# ── Chat History ─────────────────────────────────────────────────────────────

async def store_chat_message(
    db: aiosqlite.Connection, chat_id: int, role: str, content: str
) -> None:
    """Store a single chat message."""
    await db.execute(
        "INSERT INTO chat_history (chat_id, role, content) VALUES (?, ?, ?)",
        (chat_id, role, content),
    )
    await db.commit()


async def get_chat_history(
    db: aiosqlite.Connection, chat_id: int, limit: int = 20
) -> list[dict]:
    """Get the most recent chat messages for a given chat_id."""
    cursor = await db.execute(
        """SELECT role, content, created_at FROM chat_history
           WHERE chat_id = ?
           ORDER BY created_at DESC LIMIT ?""",
        (chat_id, limit),
    )
    rows = await cursor.fetchall()
    # Return in chronological order (oldest first)
    return [dict(r) for r in reversed(rows)]


# ── Forecast Accuracy ────────────────────────────────────────────────────────

async def evaluate_forecast(
    db: aiosqlite.Connection,
    forecast_id: int,
    actual_return_pct: float,
) -> None:
    """Evaluate a forecast by comparing predicted vs actual return and direction."""
    cursor = await db.execute(
        "SELECT * FROM forecasts_log WHERE id = ?", (forecast_id,)
    )
    row = await cursor.fetchone()
    if row is None:
        return

    forecast = dict(row)
    predicted_value = forecast.get("predicted_value")
    if predicted_value:
        try:
            pred_data = json.loads(predicted_value)
            predicted_return = pred_data.get("expected_return_pct", 0.0)
        except (json.JSONDecodeError, TypeError):
            predicted_return = 0.0
    else:
        predicted_return = 0.0

    predicted_dir = "up" if predicted_return > 0 else "down" if predicted_return < 0 else "flat"
    actual_dir = "up" if actual_return_pct > 0 else "down" if actual_return_pct < 0 else "flat"
    is_correct = 1 if predicted_dir == actual_dir else 0
    abs_error = abs(predicted_return - actual_return_pct)

    await db.execute(
        """INSERT INTO forecast_accuracy
           (forecast_id, evaluation_date, predicted_direction, actual_direction,
            predicted_return_pct, actual_return_pct, absolute_error, is_direction_correct)
           VALUES (?, date('now'), ?, ?, ?, ?, ?, ?)
           ON CONFLICT(forecast_id) DO UPDATE SET
               evaluation_date = excluded.evaluation_date,
               actual_direction = excluded.actual_direction,
               actual_return_pct = excluded.actual_return_pct,
               absolute_error = excluded.absolute_error,
               is_direction_correct = excluded.is_direction_correct""",
        (
            forecast_id, predicted_dir, actual_dir,
            predicted_return, actual_return_pct, abs_error, is_correct,
        ),
    )

    # Also backfill actual_value and was_correct on the forecasts_log itself
    await db.execute(
        """UPDATE forecasts_log
           SET actual_value = ?, was_correct = ?
           WHERE id = ?""",
        (str(actual_return_pct), is_correct, forecast_id),
    )
    await db.commit()


async def get_forecast_accuracy_summary(
    db: aiosqlite.Connection, days: int = 30
) -> dict:
    """Aggregate forecast accuracy: MAPE, directional accuracy, count."""
    cursor = await db.execute(
        """SELECT
               COUNT(*) as total_evaluated,
               AVG(absolute_error) as mean_abs_error,
               SUM(is_direction_correct) as correct_directions,
               AVG(CAST(is_direction_correct AS REAL)) as directional_accuracy
           FROM forecast_accuracy
           WHERE evaluation_date >= date('now', ?)""",
        (f"-{days} days",),
    )
    row = await cursor.fetchone()
    return dict(row) if row else {
        "total_evaluated": 0, "mean_abs_error": None,
        "correct_directions": 0, "directional_accuracy": None,
    }


# ── Research Themes ──────────────────────────────────────────────────────────

async def store_research_theme(db: aiosqlite.Connection, theme: dict) -> None:
    """Insert a new research theme."""
    await db.execute(
        """INSERT INTO research_themes
           (theme_date, theme, summary, impact, affected_tickers, sources, source_tier, is_active)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            theme["theme_date"],
            theme["theme"],
            theme.get("summary"),
            theme.get("impact", "medium"),
            json.dumps(theme.get("affected_tickers", [])),
            json.dumps(theme.get("sources", [])),
            theme.get("source_tier"),
            theme.get("is_active", 1),
        ),
    )
    await db.commit()


async def get_active_research_themes(
    db: aiosqlite.Connection, days: int = 7
) -> list[dict]:
    """Get active research themes from the last N days."""
    cursor = await db.execute(
        """SELECT * FROM research_themes
           WHERE is_active = 1 AND theme_date >= date('now', ?)
           ORDER BY theme_date DESC""",
        (f"-{days} days",),
    )
    rows = await cursor.fetchall()
    results = []
    for row in rows:
        d = dict(row)
        for key in ("affected_tickers", "sources"):
            if isinstance(d.get(key), str):
                try:
                    d[key] = json.loads(d[key])
                except json.JSONDecodeError:
                    d[key] = []
        results.append(d)
    return results


async def deactivate_old_themes(db: aiosqlite.Connection, days: int = 7) -> None:
    """Mark themes older than N days as inactive."""
    await db.execute(
        "UPDATE research_themes SET is_active = 0 WHERE theme_date < date('now', ?)",
        (f"-{days} days",),
    )
    await db.commit()


# ── Signal Trend ─────────────────────────────────────────────────────────────

async def get_signal_trend(
    db: aiosqlite.Connection, ticker: str, days: int = 7
) -> list[dict]:
    """Get the signal history for a ticker over the last N days."""
    cursor = await db.execute(
        """SELECT indicator_date, overall_bias, overall_confidence
           FROM technical_indicators
           WHERE ticker = ? AND indicator_date >= date('now', ?)
           ORDER BY indicator_date ASC""",
        (ticker, f"-{days} days"),
    )
    rows = await cursor.fetchall()
    return [dict(r) for r in rows]


# ── Daily Risk Metrics ───────────────────────────────────────────────────────

async def store_daily_risk_metrics(db: aiosqlite.Connection, data: dict) -> None:
    """Upsert daily risk metrics for the portfolio."""
    await db.execute(
        """INSERT INTO daily_risk_metrics (
               risk_date, snapshot_hour, run_id,
               var_95, es_95, max_drawdown, current_drawdown,
               portfolio_beta, asset_class_pcts, stress_test_results,
               diversification_ratio, entropy_score,
               yield_curve_slope, yield_curve_inverted, vix_level, vix_regime,
               credit_spread, macro_regime
           ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(risk_date, snapshot_hour) DO UPDATE SET
               run_id = excluded.run_id,
               var_95 = excluded.var_95, es_95 = excluded.es_95,
               max_drawdown = excluded.max_drawdown,
               current_drawdown = excluded.current_drawdown,
               portfolio_beta = excluded.portfolio_beta,
               asset_class_pcts = excluded.asset_class_pcts,
               stress_test_results = excluded.stress_test_results,
               diversification_ratio = excluded.diversification_ratio,
               entropy_score = excluded.entropy_score,
               yield_curve_slope = excluded.yield_curve_slope,
               yield_curve_inverted = excluded.yield_curve_inverted,
               vix_level = excluded.vix_level,
               vix_regime = excluded.vix_regime,
               credit_spread = excluded.credit_spread,
               macro_regime = excluded.macro_regime""",
        (
            data["risk_date"], data.get("snapshot_hour", 6), data.get("run_id"),
            data.get("var_95"), data.get("es_95"),
            data.get("max_drawdown"), data.get("current_drawdown"),
            data.get("portfolio_beta"),
            json.dumps(data["asset_class_pcts"]) if data.get("asset_class_pcts") else None,
            json.dumps(data["stress_test_results"]) if data.get("stress_test_results") else None,
            data.get("diversification_ratio"), data.get("entropy_score"),
            data.get("yield_curve_slope"), data.get("yield_curve_inverted"),
            data.get("vix_level"), data.get("vix_regime"),
            data.get("credit_spread"), data.get("macro_regime"),
        ),
    )
    await db.commit()


async def get_risk_metrics_history(
    db: aiosqlite.Connection, days: int = 30
) -> list[dict]:
    """Get portfolio risk metrics history."""
    cursor = await db.execute(
        """SELECT * FROM daily_risk_metrics
           WHERE risk_date >= date('now', ?)
           ORDER BY risk_date DESC""",
        (f"-{days} days",),
    )
    rows = await cursor.fetchall()
    results = []
    for row in rows:
        d = dict(row)
        for key in ("asset_class_pcts", "stress_test_results"):
            if isinstance(d.get(key), str):
                try:
                    d[key] = json.loads(d[key])
                except json.JSONDecodeError:
                    pass
        results.append(d)
    return results


async def get_latest_risk_metrics(db: aiosqlite.Connection) -> dict | None:
    """Get the most recent daily risk metrics (including macro snapshot)."""
    cursor = await db.execute(
        "SELECT * FROM daily_risk_metrics ORDER BY risk_date DESC, snapshot_hour DESC LIMIT 1"
    )
    row = await cursor.fetchone()
    if row is None:
        return None
    d = dict(row)
    for key in ("asset_class_pcts", "stress_test_results"):
        if isinstance(d.get(key), str):
            try:
                d[key] = json.loads(d[key])
            except json.JSONDecodeError:
                pass
    return d


# ── Intraday Comparison ──────────────────────────────────────────────────────

async def get_intraday_snapshots(
    db: aiosqlite.Connection, ticker: str, date: str,
) -> list[dict]:
    """Get all snapshot_hour entries for a ticker on a given date."""
    cursor = await db.execute(
        """SELECT snapshot_hour, overall_bias, overall_confidence, rsi_14,
                  macd_histogram, adx, stochastic_k, bb_pct_b, vwap
           FROM technical_indicators
           WHERE ticker = ? AND indicator_date = ?
           ORDER BY snapshot_hour ASC""",
        (ticker, date),
    )
    return [dict(r) for r in await cursor.fetchall()]


async def get_indicator_history(
    db: aiosqlite.Connection, ticker: str, indicator_name: str, days: int = 14,
) -> list[dict]:
    """Get a specific indicator's values over the last N days.

    indicator_name must be a valid column in technical_indicators.
    """
    valid_cols = {
        "rsi_14", "macd_line", "macd_histogram", "adx", "stochastic_k",
        "stochastic_d", "bb_pct_b", "bb_bandwidth", "atr_14", "obv", "vwap",
        "overall_bias", "overall_confidence", "sma50", "sma200",
    }
    if indicator_name not in valid_cols:
        return []
    cursor = await db.execute(
        f"""SELECT indicator_date, snapshot_hour, {indicator_name}
            FROM technical_indicators
            WHERE ticker = ? AND indicator_date >= date('now', ?)
            ORDER BY indicator_date ASC, snapshot_hour ASC""",
        (ticker, f"-{days} days"),
    )
    return [dict(r) for r in await cursor.fetchall()]


# ── Earnings Calendar ────────────────────────────────────────────────────────

async def upsert_earnings_entry(db: aiosqlite.Connection, data: dict) -> None:
    """Insert or update an earnings calendar entry."""
    await db.execute(
        """INSERT INTO earnings_calendar (
               ticker, earnings_date, earnings_time,
               eps_estimate, revenue_estimate,
               eps_actual, revenue_actual,
               eps_surprise_pct, revenue_surprise_pct,
               status, source, updated_at
           ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
           ON CONFLICT(ticker, earnings_date) DO UPDATE SET
               earnings_time = COALESCE(excluded.earnings_time, earnings_calendar.earnings_time),
               eps_estimate = COALESCE(excluded.eps_estimate, earnings_calendar.eps_estimate),
               revenue_estimate = COALESCE(excluded.revenue_estimate, earnings_calendar.revenue_estimate),
               eps_actual = COALESCE(excluded.eps_actual, earnings_calendar.eps_actual),
               revenue_actual = COALESCE(excluded.revenue_actual, earnings_calendar.revenue_actual),
               eps_surprise_pct = COALESCE(excluded.eps_surprise_pct, earnings_calendar.eps_surprise_pct),
               revenue_surprise_pct = COALESCE(excluded.revenue_surprise_pct, earnings_calendar.revenue_surprise_pct),
               status = excluded.status,
               updated_at = datetime('now')""",
        (
            data["ticker"], data["earnings_date"], data.get("earnings_time"),
            data.get("eps_estimate"), data.get("revenue_estimate"),
            data.get("eps_actual"), data.get("revenue_actual"),
            data.get("eps_surprise_pct"), data.get("revenue_surprise_pct"),
            data.get("status", "upcoming"), data.get("source", "yfinance"),
        ),
    )
    await db.commit()


async def get_upcoming_earnings(
    db: aiosqlite.Connection, days: int = 14,
) -> list[dict]:
    """Get earnings in the next N days for all tracked tickers."""
    cursor = await db.execute(
        """SELECT * FROM earnings_calendar
           WHERE earnings_date BETWEEN date('now') AND date('now', ?)
             AND status = 'upcoming'
           ORDER BY earnings_date ASC""",
        (f"+{days} days",),
    )
    return [dict(r) for r in await cursor.fetchall()]


async def get_recent_earnings(
    db: aiosqlite.Connection, days: int = 7,
) -> list[dict]:
    """Get recently reported earnings."""
    cursor = await db.execute(
        """SELECT * FROM earnings_calendar
           WHERE earnings_date >= date('now', ?)
             AND status = 'reported'
           ORDER BY earnings_date DESC""",
        (f"-{days} days",),
    )
    return [dict(r) for r in await cursor.fetchall()]


async def get_earnings_for_ticker(
    db: aiosqlite.Connection, ticker: str,
) -> list[dict]:
    """Get all earnings entries for a specific ticker."""
    cursor = await db.execute(
        """SELECT * FROM earnings_calendar
           WHERE ticker = ?
           ORDER BY earnings_date DESC""",
        (ticker,),
    )
    return [dict(r) for r in await cursor.fetchall()]


# ── Correlation Snapshot ─────────────────────────────────────────────────────

async def store_correlation_snapshot(db: aiosqlite.Connection, data: dict) -> None:
    """Upsert a correlation snapshot."""
    await db.execute(
        """INSERT INTO correlation_snapshot (
               snapshot_date, run_id, correlation_matrix,
               top_correlations, diversification_score, cluster_assignments
           ) VALUES (?, ?, ?, ?, ?, ?)
           ON CONFLICT(snapshot_date) DO UPDATE SET
               run_id = excluded.run_id,
               correlation_matrix = excluded.correlation_matrix,
               top_correlations = excluded.top_correlations,
               diversification_score = excluded.diversification_score,
               cluster_assignments = excluded.cluster_assignments""",
        (
            data["snapshot_date"], data.get("run_id"),
            json.dumps(data["correlation_matrix"]),
            json.dumps(data.get("top_correlations", [])),
            data.get("diversification_score"),
            json.dumps(data.get("cluster_assignments", {})),
        ),
    )
    await db.commit()


async def get_latest_correlation_snapshot(db: aiosqlite.Connection) -> dict | None:
    """Get the most recent correlation snapshot."""
    cursor = await db.execute(
        "SELECT * FROM correlation_snapshot ORDER BY snapshot_date DESC LIMIT 1"
    )
    row = await cursor.fetchone()
    if row is None:
        return None
    d = dict(row)
    for key in ("correlation_matrix", "top_correlations", "cluster_assignments"):
        if isinstance(d.get(key), str):
            try:
                d[key] = json.loads(d[key])
            except json.JSONDecodeError:
                pass
    return d


# ── Onboarding State ─────────────────────────────────────────────────────────

async def get_onboarding_state(db: aiosqlite.Connection) -> dict | None:
    """Get the current onboarding state (singleton row)."""
    cursor = await db.execute("SELECT * FROM onboarding_state WHERE id = 1")
    row = await cursor.fetchone()
    if row is None:
        return None
    d = dict(row)
    if isinstance(d.get("steps_completed"), str):
        try:
            d["steps_completed"] = json.loads(d["steps_completed"])
        except json.JSONDecodeError:
            d["steps_completed"] = []
    return d


async def update_onboarding_step(
    db: aiosqlite.Connection,
    step: str,
    completed_steps: list[str] | None = None,
) -> None:
    """Update the onboarding state. Creates the row if it doesn't exist."""
    existing = await get_onboarding_state(db)
    if existing is None:
        steps = json.dumps(completed_steps or [])
        await db.execute(
            "INSERT INTO onboarding_state (id, current_step, steps_completed) VALUES (1, ?, ?)",
            (step, steps),
        )
    else:
        if completed_steps is not None:
            steps = json.dumps(completed_steps)
        else:
            # Add the current step to the completed list if transitioning
            prev_steps = existing.get("steps_completed", [])
            if existing["current_step"] not in prev_steps and existing["current_step"] != step:
                prev_steps.append(existing["current_step"])
            steps = json.dumps(prev_steps)

        completed_at = datetime.utcnow().isoformat() if step == "done" else None
        await db.execute(
            """UPDATE onboarding_state
               SET current_step = ?, steps_completed = ?, completed_at = ?
               WHERE id = 1""",
            (step, steps, completed_at),
        )
    await db.commit()
