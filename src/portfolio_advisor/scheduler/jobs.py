"""Scheduled job implementations — daily, weekly, pre-compute, midday, evening, forecast eval."""

from __future__ import annotations

import json
import logging
from datetime import date, timedelta

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


# ── Daily monitoring ─────────────────────────────────────────────────────────


async def daily_job() -> None:
    """Run the daily monitoring pipeline."""
    logger.info("Starting daily monitoring pipeline")
    ctx = await _build_context()

    try:
        from portfolio_advisor.agents.orchestrator import get_daily_orchestrator

        today = date.today().isoformat()
        prompt = (
            f"Run the daily analysis pipeline for {today}.\n"
            f"Watchlist: {', '.join(ctx.watchlist)}\n"
            f"Analyze all tickers, synthesize a daily brief, store it, "
            f"and produce a Telegram-formatted summary."
        )

        result = await Runner.run(
            starting_agent=get_daily_orchestrator(),
            input=prompt,
            context=ctx,
        )

        # Send the summary to Telegram
        output = result.final_output or "Daily analysis completed but no summary was generated."

        # Try to extract telegram_summary from stored brief
        settings = get_settings()
        async with get_db(settings.db_path) as db:
            briefs = await queries.retrieve_daily_briefs(db, today, today)
            if briefs:
                brief_data = briefs[0]
                telegram_text = brief_data.get("telegram_summary") or output
            else:
                telegram_text = output

        await send_message(telegram_text)
        logger.info("Daily pipeline completed successfully")

    except Exception as e:
        logger.exception(f"Daily pipeline failed: {e}")
        await send_message(f"Daily analysis failed: {str(e)[:500]}")


# ── Midday update ────────────────────────────────────────────────────────────


async def midday_update_job() -> None:
    """Run midday update: re-compute indicators and detect significant signal changes.

    Compares morning vs midday signals. If any ticker has a signal reversal
    or large confidence shift, sends a Telegram alert.
    """
    logger.info("Starting midday update")
    ctx = await _build_context()
    settings = get_settings()
    today = date.today().isoformat()

    try:
        # Get morning signals (already stored by precompute_morning)
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

        # Run midday pre-compute (will overwrite morning data)
        from portfolio_advisor.tools.precomputed import run_precompute_pipeline

        await run_precompute_pipeline(ctx)

        # Get updated signals
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
                + "\n\n_Pre-computed data refreshed._"
            )
            await send_message(msg)
            logger.info(f"Midday update: {len(changes)} signal changes detected")
        else:
            logger.info("Midday update: no significant signal changes")

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

        # Run news alert pipeline for evening theme detection
        try:
            from portfolio_advisor.scheduler.alerts import run_news_alert_pipeline

            alert_result = await run_news_alert_pipeline(ctx)
            logger.info(f"Evening news alerts: {alert_result}")
        except Exception as alert_err:
            logger.warning(f"News alert pipeline failed during evening: {alert_err}")

    except Exception as e:
        logger.exception(f"Evening summary failed: {e}")


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
