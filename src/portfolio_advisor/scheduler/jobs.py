"""Scheduled job implementations — daily, weekly, pre-compute, midday, evening, forecast eval."""

from __future__ import annotations

import json
import logging
from datetime import date, timedelta
from typing import Any

from agents import Runner

from portfolio_advisor.agents.context import AppContext
from portfolio_advisor.config import get_settings
from portfolio_advisor.db import queries
from portfolio_advisor.db.connection import get_db
from portfolio_advisor.telegram_bot.bot import send_message

logger = logging.getLogger(__name__)


async def _build_context() -> AppContext:
    """Build the shared AppContext for a run."""
    settings = get_settings()
    async with get_db(settings.db_path) as db:
        prefs = await queries.get_user_preferences(db)
    watchlist = prefs.get("watchlist", settings.default_watchlist)

    return AppContext(
        db_path=settings.db_path,
        telegram_chat_id=settings.telegram_chat_id,
        run_date=date.today(),
        watchlist=watchlist,
        token_budget_remaining=settings.daily_token_budget,
        max_web_search_calls_daily=settings.max_web_searches_daily,
    )


# ── Pre-compute pipeline ────────────────────────────────────────────────────


async def precompute_job() -> None:
    """Run the pre-compute pipeline: batch compute all indicators for the watchlist."""
    logger.info("Starting pre-compute pipeline")
    ctx = await _build_context()

    try:
        from portfolio_advisor.tools.precomputed import run_precompute_pipeline

        results = await run_precompute_pipeline(ctx)
        logger.info(
            f"Pre-compute completed: {len(results['processed'])} processed, "
            f"{len(results['failed'])} failed"
        )
    except Exception as e:
        logger.exception(f"Pre-compute pipeline failed: {e}")


# ── Daily context builder (pure Python, 0 LLM tokens) ───────────────────────


async def _build_daily_context(ctx: AppContext) -> str:
    """Build a structured context document from pre-computed data for the synthesis agent.

    Loads all cached indicators, quant metrics, macro snapshot, earnings,
    correlations, and portfolio state from the database. Returns a text
    document (~2-4 KB) ready to pass to the Daily Synthesis Agent.

    This replaces the Technical Agent + Quantitative Agent calls, saving ~60K tokens.
    """
    today = date.today().isoformat()
    sections: list[str] = [f"# Pre-Computed Daily Analysis — {today}\n"]

    async with get_db(ctx.db_path) as db:
        # ── Per-ticker analysis ──────────────────────────────────────────
        tech_indicators = await queries.get_bulk_technical_indicators(
            db, ctx.watchlist, today
        )
        quant_metrics = await queries.get_bulk_quant_metrics(db, ctx.watchlist, today)

        # Index quant by ticker for merging
        quant_by_ticker: dict[str, dict[str, Any]] = {
            qm["ticker"]: qm for qm in quant_metrics
        }

        if tech_indicators:
            sections.append("## Ticker Signals\n")
            for ind in sorted(
                tech_indicators,
                key=lambda x: x.get("overall_confidence", 0),
                reverse=True,
            ):
                ticker = ind["ticker"]
                # Use pre-generated narrative if available
                narrative = ind.get("narrative")
                if narrative:
                    # Append quant summary
                    qm = quant_by_ticker.get(ticker, {})
                    quant_parts = []
                    if qm.get("garch_vol") is not None:
                        quant_parts.append(f"GARCH vol={qm['garch_vol']:.1f}%")
                    if qm.get("hmm_state"):
                        quant_parts.append(f"HMM={qm['hmm_state']}")
                    if qm.get("kalman_beta") is not None:
                        quant_parts.append(f"Kalman β={qm['kalman_beta']:.2f}")
                    if qm.get("return_1w_pct") is not None:
                        quant_parts.append(
                            f"1w forecast={qm['return_1w_pct']:+.1f}% "
                            f"[{qm.get('return_1w_ci_low', '?')}, "
                            f"{qm.get('return_1w_ci_high', '?')}]"
                        )
                    if qm.get("regime"):
                        quant_parts.append(f"Regime: {qm['regime']}")
                    if qm.get("vol_regime"):
                        quant_parts.append(f"Vol regime: {qm['vol_regime']}")
                    quant_str = (" | " + ", ".join(quant_parts)) if quant_parts else ""
                    sections.append(f"- {narrative}{quant_str}")
                else:
                    # Fallback: structured data
                    bias = ind.get("overall_bias", "?")
                    conf = ind.get("overall_confidence", 0)
                    rsi = ind.get("rsi_14")
                    rsi_str = f", RSI={rsi:.0f}" if rsi else ""
                    sections.append(f"- {ticker}: {bias} (conf {conf:.2f}{rsi_str})")
            sections.append("")

        # ── Macro snapshot ───────────────────────────────────────────────
        risk = await queries.get_latest_risk_metrics(db)
        if risk:
            sections.append("## Macro & Risk Snapshot\n")
            macro_parts = []
            if risk.get("macro_regime"):
                macro_parts.append(f"Macro regime: {risk['macro_regime']}")
            if risk.get("vix_level") is not None:
                macro_parts.append(
                    f"VIX={risk['vix_level']:.1f} ({risk.get('vix_regime', '?')})"
                )
            if risk.get("yield_curve_slope") is not None:
                inv = " (INVERTED)" if risk.get("yield_curve_inverted") else ""
                macro_parts.append(f"Yield curve slope={risk['yield_curve_slope']:.2f}{inv}")
            if risk.get("credit_spread") is not None:
                macro_parts.append(f"Credit spread={risk['credit_spread']:.2f}")
            if macro_parts:
                sections.append("- " + " | ".join(macro_parts))

            risk_parts = []
            if risk.get("var_95") is not None:
                risk_parts.append(f"VaR95={risk['var_95']:.2f}%")
            if risk.get("es_95") is not None:
                risk_parts.append(f"ES95={risk['es_95']:.2f}%")
            if risk.get("portfolio_beta") is not None:
                risk_parts.append(f"Beta={risk['portfolio_beta']:.2f}")
            if risk.get("max_drawdown") is not None:
                risk_parts.append(f"Max DD={risk['max_drawdown']:.1f}%")
            if risk_parts:
                sections.append("- Portfolio risk: " + ", ".join(risk_parts))
            sections.append("")

        # ── Earnings ─────────────────────────────────────────────────────
        upcoming = await queries.get_upcoming_earnings(db, days=7)
        recent = await queries.get_recent_earnings(db, days=3)

        if upcoming or recent:
            sections.append("## Earnings\n")
            if upcoming:
                sections.append("Upcoming:")
                for e in upcoming[:8]:
                    eps_str = f" (est EPS ${e['eps_estimate']:.2f})" if e.get("eps_estimate") else ""
                    sections.append(
                        f"- {e['ticker']}: {e['earnings_date']} "
                        f"{e.get('earnings_time', '')}{eps_str}"
                    )
            if recent:
                sections.append("Recently reported:")
                for e in recent[:5]:
                    surprise = e.get("eps_surprise_pct", 0)
                    beat_miss = "beat" if surprise > 0 else "miss" if surprise < 0 else "in-line"
                    sections.append(
                        f"- {e['ticker']}: EPS ${e.get('eps_actual', '?')} "
                        f"vs ${e.get('eps_estimate', '?')} est ({beat_miss} {abs(surprise):.1f}%)"
                    )
            sections.append("")

        # ── Correlations ─────────────────────────────────────────────────
        corr = await queries.get_latest_correlation_snapshot(db)
        if corr:
            sections.append("## Correlation & Diversification\n")
            if corr.get("diversification_score") is not None:
                sections.append(
                    f"- Diversification score: {corr['diversification_score']:.2f}"
                )
            top_corr = corr.get("top_correlations")
            if top_corr:
                try:
                    pairs = json.loads(top_corr) if isinstance(top_corr, str) else top_corr
                    if pairs:
                        sections.append("- Most correlated pairs:")
                        for pair in pairs[:5]:
                            sections.append(
                                f"  {pair.get('pair', '?')}: {pair.get('correlation', 0):.2f}"
                            )
                except (json.JSONDecodeError, TypeError):
                    pass
            sections.append("")

        # ── Portfolio state ──────────────────────────────────────────────
        portfolio = await queries.get_portfolio_state(db)
        if portfolio:
            sections.append("## Current Portfolio\n")
            for pos in portfolio[:15]:
                sections.append(
                    f"- {pos['ticker']}: {pos['weight_pct']:.1f}% ({pos.get('asset_class', '?')})"
                )
            sections.append("")

    return "\n".join(sections)


# ── Earnings alerts (used by midday + evening jobs) ──────────────────────────


async def _check_earnings_alerts(ctx: AppContext) -> None:
    """Check for earnings events and send proactive Telegram alerts.

    - Earnings reporting TODAY: note them (research agent may have already covered).
    - Earnings TOMORROW: send a proactive heads-up alert.
    - Earnings JUST REPORTED with actual results: send surprise notification.
    """
    try:
        async with get_db(ctx.db_path) as db:
            upcoming_1d = await queries.get_upcoming_earnings(db, days=1)
            upcoming_2d = await queries.get_upcoming_earnings(db, days=2)
            recent = await queries.get_recent_earnings(db, days=1)

        alerts: list[str] = []

        # Earnings tomorrow (in upcoming_2d but not in upcoming_1d)
        today_tickers = {e["ticker"] for e in upcoming_1d}
        for e in upcoming_2d:
            if e["ticker"] not in today_tickers:
                eps_str = f" | Est EPS: ${e['eps_estimate']:.2f}" if e.get("eps_estimate") else ""
                rev_str = (
                    f" | Est Rev: ${e['revenue_estimate']/1e9:.1f}B"
                    if e.get("revenue_estimate") and e["revenue_estimate"] > 1e6
                    else ""
                )
                alerts.append(
                    f"**{e['ticker']}** reports earnings tomorrow "
                    f"({e.get('earnings_time', 'time TBD')}){eps_str}{rev_str}"
                )

        # Recently reported with results
        for e in recent:
            if e.get("eps_actual") is not None and e.get("eps_estimate") is not None:
                surprise = e.get("eps_surprise_pct", 0)
                beat_miss = "BEAT" if surprise > 0 else "MISS" if surprise < 0 else "IN-LINE"
                alerts.append(
                    f"**{e['ticker']}** earnings: EPS ${e['eps_actual']:.2f} "
                    f"vs ${e['eps_estimate']:.2f} est "
                    f"({beat_miss} by {abs(surprise):.1f}%)"
                )

        if alerts:
            msg = "**Earnings Alert**\n\n" + "\n".join(alerts)
            await send_message(msg)
            logger.info(f"Sent {len(alerts)} earnings alerts")

    except Exception as e:
        logger.warning(f"Earnings alert check failed: {e}")


# ── Daily monitoring ─────────────────────────────────────────────────────────


async def daily_job() -> None:
    """Run the daily monitoring pipeline.

    v3 optimization: Instead of calling Technical + Quantitative agents (~60K tokens),
    loads pre-computed data from the DB (0 tokens) and passes it as context to the
    Daily Synthesis Agent (gpt-5.2). Only the Research Agent is called live for fresh
    news (~20K tokens).

    Total: ~40-50K tokens (down from ~130K).
    """
    logger.info("Starting daily monitoring pipeline (v3 — pre-computed context)")
    ctx = await _build_context()

    try:
        today = date.today().isoformat()

        # Step 1: Build context from pre-computed data (0 tokens — pure Python)
        analysis_context = await _build_daily_context(ctx)
        logger.info(
            f"Built daily context: {len(analysis_context)} chars from pre-computed data"
        )

        # Step 2: Run research agent for fresh news (~20K tokens, gpt-5-mini)
        from portfolio_advisor.agents.research import get_research_agent

        # Build earnings-aware research prompt
        earnings_hint = ""
        settings = get_settings()
        async with get_db(settings.db_path) as db:
            upcoming = await queries.get_upcoming_earnings(db, days=2)
        if upcoming:
            tickers_reporting = [e["ticker"] for e in upcoming]
            earnings_hint = (
                f"\nIMPORTANT: These tickers report earnings within 48 hours: "
                f"{', '.join(tickers_reporting)}. Include an earnings-focused search "
                f"for each (analyst expectations, guidance, recent revisions)."
            )

        research_result = await Runner.run(
            starting_agent=get_research_agent(),
            input=(
                f"Research macro/market context for today ({today}). "
                f"Watchlist: {', '.join(ctx.watchlist)}. "
                f"Identify top themes, ticker-specific news, and macro developments."
                f"{earnings_hint}"
            ),
            context=ctx,
        )
        research_output = research_result.final_output or "No research findings."
        logger.info("Research agent completed")

        # Step 3: Synthesize with Daily Synthesis Agent (~20K tokens, gpt-5-mini)
        from portfolio_advisor.agents.orchestrator import get_daily_synthesis_agent

        prompt = (
            f"Synthesize the following pre-computed analysis with market research "
            f"into a DailyBrief for {today}.\n\n"
            f"=== PRE-COMPUTED ANALYSIS (fresh from today's pipeline) ===\n\n"
            f"{analysis_context}\n\n"
            f"=== MARKET RESEARCH (fresh web search) ===\n\n"
            f"{research_output}\n\n"
            f"Produce the DailyBrief JSON with a telegram_summary. "
            f"Store the brief and log forecasts for tickers with return predictions."
        )

        result = await Runner.run(
            starting_agent=get_daily_synthesis_agent(),
            input=prompt,
            context=ctx,
        )

        # Send the summary to Telegram
        output = result.final_output or "Daily analysis completed but no summary was generated."

        async with get_db(settings.db_path) as db:
            briefs = await queries.retrieve_daily_briefs(db, today, today)
            if briefs:
                brief_data = briefs[0]
                telegram_text = brief_data.get("telegram_summary") or output
            else:
                telegram_text = output

        await send_message(telegram_text)
        logger.info("Daily pipeline completed successfully (v3)")

    except Exception as e:
        logger.exception(f"Daily pipeline failed: {e}")
        await send_message(f"Daily analysis failed: {str(e)[:500]}")


# ── Midday update ────────────────────────────────────────────────────────────


async def midday_update_job() -> None:
    """Run midday update: re-compute indicators and detect significant signal changes.

    v3: Morning data is now preserved (snapshot_hour=6 vs 13). Compares morning
    vs midday snapshots without data loss. Also checks for earnings alerts.
    """
    logger.info("Starting midday update")
    ctx = await _build_context()
    settings = get_settings()
    today = date.today().isoformat()

    try:
        # Get morning signals (snapshot_hour=6, preserved by v3 schema)
        async with get_db(settings.db_path) as db:
            morning_indicators = await queries.get_bulk_technical_indicators(
                db, ctx.watchlist, today
            )

        morning_signals = {
            ind["ticker"]: {
                "bias": ind.get("overall_bias"),
                "confidence": ind.get("overall_confidence"),
            }
            for ind in morning_indicators
        }

        # Run midday pre-compute (stores as snapshot_hour=13, does NOT overwrite morning)
        from portfolio_advisor.tools.precomputed import run_precompute_pipeline

        await run_precompute_pipeline(ctx)

        # Get midday signals (latest snapshot for today = the one we just stored)
        async with get_db(settings.db_path) as db:
            midday_indicators = await queries.get_bulk_technical_indicators(
                db, ctx.watchlist, today
            )

        # Compare and detect significant changes
        changes = []
        for ind in midday_indicators:
            ticker = ind["ticker"]
            new_bias = ind.get("overall_bias")
            new_conf = ind.get("overall_confidence", 0)

            old = morning_signals.get(ticker, {})
            old_bias = old.get("bias")
            old_conf = old.get("confidence", 0)

            if old_bias is None:
                continue

            # Signal reversal (bullish→bearish or vice versa)
            bullish_set = {"bullish", "slightly_bullish"}
            bearish_set = {"bearish", "slightly_bearish"}
            if (old_bias in bullish_set and new_bias in bearish_set) or (
                old_bias in bearish_set and new_bias in bullish_set
            ):
                changes.append(
                    f"**{ticker}**: {old_bias} → {new_bias} "
                    f"(conf {old_conf:.2f} → {new_conf:.2f})"
                )
            # Large confidence shift (>0.15)
            elif abs(new_conf - old_conf) > 0.15:
                direction = "stronger" if new_conf > old_conf else "weaker"
                changes.append(
                    f"**{ticker}**: {new_bias} signal {direction} "
                    f"(conf {old_conf:.2f} → {new_conf:.2f})"
                )

        if changes:
            msg = (
                "**Midday Signal Update**\n\n"
                + "\n".join(changes)
                + "\n\n_Pre-computed data refreshed (morning data preserved)._"
            )
            await send_message(msg)
            logger.info(f"Midday update: {len(changes)} signal changes detected")
        else:
            logger.info("Midday update: no significant signal changes")

        # Check for earnings alerts
        await _check_earnings_alerts(ctx)

        # Run news alert pipeline to detect high-impact themes
        try:
            from portfolio_advisor.scheduler.alerts import run_news_alert_pipeline

            alert_result = await run_news_alert_pipeline(ctx)
            logger.info(f"Midday news alerts: {alert_result}")
        except Exception as alert_err:
            logger.warning(f"News alert pipeline failed during midday: {alert_err}")

    except Exception as e:
        logger.exception(f"Midday update failed: {e}")


# ── Evening summary ──────────────────────────────────────────────────────────


async def evening_summary_job() -> None:
    """Generate and send an evening day scorecard.

    Summarizes: morning signals vs actual close, forecast accuracy for the day,
    top movers, and a brief risk snapshot.
    """
    logger.info("Starting evening summary")
    ctx = await _build_context()
    settings = get_settings()
    today = date.today().isoformat()

    try:
        async with get_db(settings.db_path) as db:
            tech_indicators = await queries.get_bulk_technical_indicators(
                db, ctx.watchlist, today
            )
            quant_metrics = await queries.get_bulk_quant_metrics(db, ctx.watchlist, today)
            risk_metrics = await queries.get_risk_metrics_history(db, days=1)

        if not tech_indicators:
            logger.info("Evening summary: no data for today, skipping")
            return

        # Build scorecard
        lines = [f"**Evening Scorecard — {today}**\n"]

        # Top signals
        sorted_tech = sorted(
            tech_indicators,
            key=lambda x: x.get("overall_confidence", 0),
            reverse=True,
        )
        lines.append("**Top Signals:**")
        for ind in sorted_tech[:5]:
            ticker = ind["ticker"]
            bias = ind.get("overall_bias", "?")
            conf = ind.get("overall_confidence", 0)
            rsi = ind.get("rsi_14")
            rsi_str = f" RSI={rsi:.0f}" if rsi else ""
            lines.append(f"  {ticker}: {bias} ({conf:.2f}){rsi_str}")

        # Quant highlights
        if quant_metrics:
            lines.append("\n**Quant Highlights:**")
            for qm in sorted(quant_metrics, key=lambda x: abs(x.get("return_1w_pct") or 0),
                              reverse=True)[:3]:
                ticker = qm["ticker"]
                ret_1w = qm.get("return_1w_pct")
                regime = qm.get("regime", "?")
                vol_regime = qm.get("vol_regime", "?")
                ret_str = f" 1w forecast={ret_1w:+.1f}%" if ret_1w is not None else ""
                lines.append(f"  {ticker}: {regime}/{vol_regime}{ret_str}")

        # Risk snapshot
        if risk_metrics:
            rm = risk_metrics[0]
            var_95 = rm.get("var_95")
            beta = rm.get("portfolio_beta")
            dd = rm.get("current_drawdown")
            risk_parts = []
            if var_95 is not None:
                risk_parts.append(f"VaR95={var_95:.2f}%")
            if beta is not None:
                risk_parts.append(f"Beta={beta:.2f}")
            if dd is not None:
                risk_parts.append(f"DD={dd:.1f}%")
            if risk_parts:
                lines.append(f"\n**Portfolio Risk:** {', '.join(risk_parts)}")

        msg = "\n".join(lines)
        await send_message(msg)
        logger.info("Evening summary sent")

        # Check for earnings alerts
        await _check_earnings_alerts(ctx)

        # Run news alert pipeline for evening theme detection
        try:
            from portfolio_advisor.scheduler.alerts import run_news_alert_pipeline

            alert_result = await run_news_alert_pipeline(ctx)
            logger.info(f"Evening news alerts: {alert_result}")
        except Exception as alert_err:
            logger.warning(f"News alert pipeline failed during evening: {alert_err}")

    except Exception as e:
        logger.exception(f"Evening summary failed: {e}")


# ── News check (lightweight, between main runs) ─────────────────────────────


async def news_check_job() -> None:
    """Run a lightweight news-only check between the main scheduled runs.

    Only calls the news alert pipeline (research agent + theme detection).
    No pre-compute, no signal comparison. Also checks earnings alerts.
    Reduces maximum news blind-spot from ~6-10 hours to ~3-4 hours.
    """
    logger.info("Starting news check job")
    ctx = await _build_context()

    try:
        # Check earnings alerts
        await _check_earnings_alerts(ctx)

        # Run news alert pipeline
        from portfolio_advisor.scheduler.alerts import run_news_alert_pipeline

        alert_result = await run_news_alert_pipeline(ctx)
        logger.info(f"News check completed: {alert_result}")

    except Exception as e:
        logger.exception(f"News check job failed: {e}")


# ── Weekly report ────────────────────────────────────────────────────────────


async def weekly_job() -> None:
    """Run the weekly portfolio recommendation pipeline."""
    logger.info("Starting weekly portfolio recommendation pipeline")
    ctx = await _build_context()
    ctx.token_budget_remaining = get_settings().weekly_token_budget

    try:
        from portfolio_advisor.agents.orchestrator import get_weekly_orchestrator

        today = date.today()
        week_start = (today - timedelta(days=7)).isoformat()
        week_end = today.isoformat()

        prompt = (
            f"Run the weekly portfolio recommendation pipeline for week ending {week_end}.\n"
            f"Review daily briefs from {week_start} to {week_end}.\n"
            f"Watchlist: {', '.join(ctx.watchlist)}\n"
            f"Produce a comprehensive investment committee memo with allocation "
            f"recommendations, risk assessment, and outlook. Store the report "
            f"and produce a Telegram-formatted summary."
        )

        result = await Runner.run(
            starting_agent=get_weekly_orchestrator(),
            input=prompt,
            context=ctx,
        )

        output = result.final_output or "Weekly analysis completed but no summary was generated."

        # Try to extract telegram_summary from stored report
        settings = get_settings()
        async with get_db(settings.db_path) as db:
            reports = await queries.retrieve_weekly_reports(db, count=1)
            if reports:
                report_data = reports[0]
                try:
                    content = json.loads(report_data.get("content_json", "{}"))
                    telegram_text = content.get("telegram_summary") or output
                except (json.JSONDecodeError, TypeError):
                    telegram_text = output
            else:
                telegram_text = output

        await send_message(telegram_text)
        logger.info("Weekly pipeline completed successfully")

    except Exception as e:
        logger.exception(f"Weekly pipeline failed: {e}")
        await send_message(f"Weekly analysis failed: {str(e)[:500]}")


# ── Forecast evaluation ──────────────────────────────────────────────────────


async def forecast_evaluation_job() -> None:
    """Evaluate past forecasts by comparing predictions to actual returns.

    Finds unevaluated forecasts from 7+ days ago (giving time for the forecast
    horizon to play out), fetches actual prices, computes actual returns,
    and backfills the forecast_accuracy table.
    """
    logger.info("Starting forecast evaluation")
    settings = get_settings()

    try:
        import yfinance as yf

        async with get_db(settings.db_path) as db:
            # Find unevaluated forecasts from 7+ days ago
            cutoff = (date.today() - timedelta(days=7)).isoformat()
            cursor = await db.execute(
                """SELECT fl.id, fl.ticker, fl.forecast_date, fl.horizon,
                          fl.predicted_value, fl.was_correct
                   FROM forecasts_log fl
                   LEFT JOIN forecast_accuracy fa ON fl.id = fa.forecast_id
                   WHERE fa.id IS NULL
                     AND fl.forecast_date <= ?
                   ORDER BY fl.forecast_date
                   LIMIT 50""",
                (cutoff,),
            )
            unevaluated = [dict(row) for row in await cursor.fetchall()]

        if not unevaluated:
            logger.info("No unevaluated forecasts found")
            return

        # Group by ticker to batch-fetch prices
        tickers = list({f["ticker"] for f in unevaluated})
        logger.info(f"Evaluating {len(unevaluated)} forecasts for {len(tickers)} tickers")

        # Fetch recent prices
        ticker_prices = {}
        if tickers:
            df = yf.download(tickers, period="3mo", progress=False)
            if not df.empty:
                for t in tickers:
                    try:
                        if len(tickers) == 1:
                            closes = df["Close"]
                        else:
                            closes = df["Close"][t]
                        ticker_prices[t] = closes.dropna()
                    except (KeyError, AttributeError):
                        pass

        evaluated_count = 0
        async with get_db(settings.db_path) as db:
            for forecast in unevaluated:
                ticker = forecast["ticker"]
                forecast_date = forecast["forecast_date"]
                horizon = forecast.get("horizon", "1w")

                # Determine evaluation date based on horizon
                horizon_days = {"1w": 7, "1m": 21, "3m": 63}.get(horizon, 7)
                try:
                    fc_date = date.fromisoformat(forecast_date)
                    eval_date = fc_date + timedelta(days=horizon_days)
                except (ValueError, TypeError):
                    continue

                # Check if we have price data for both dates
                prices = ticker_prices.get(ticker)
                if prices is None or prices.empty:
                    continue

                # Find prices closest to forecast date and eval date
                try:
                    fc_price = None
                    eval_price = None
                    for idx in prices.index:
                        idx_date = idx.date() if hasattr(idx, "date") else idx
                        if fc_price is None and idx_date >= fc_date:
                            fc_price = float(prices[idx])
                        if eval_price is None and idx_date >= eval_date:
                            eval_price = float(prices[idx])

                    if fc_price and eval_price and fc_price > 0:
                        actual_return_pct = ((eval_price / fc_price) - 1) * 100
                        await queries.evaluate_forecast(
                            db, forecast["id"], round(actual_return_pct, 2)
                        )
                        evaluated_count += 1
                except Exception:
                    continue

        logger.info(f"Forecast evaluation: evaluated {evaluated_count}/{len(unevaluated)}")

    except Exception as e:
        logger.exception(f"Forecast evaluation failed: {e}")
