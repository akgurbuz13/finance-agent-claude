"""Full-powered chat agent — interactive analyst with ALL tools + live analysis."""

from __future__ import annotations

from agents import Agent, WebSearchTool

from portfolio_advisor.agents.context import AppContext
from portfolio_advisor.config import get_settings
from portfolio_advisor.tools.advanced_analytics import (
    compute_brinson_attribution,
    compute_hierarchical_clustering,
    compute_information_entropy,
    compute_mutual_information,
    compute_pca_returns,
    compute_style_analysis,
)
from portfolio_advisor.tools.advanced_portfolio import (
    compute_kelly_criterion,
    compute_transaction_costs,
    optimize_cvar,
    optimize_entropy_weighted,
    optimize_hrp,
    optimize_max_diversification,
)
from portfolio_advisor.tools.advanced_quant import (
    compute_fama_french_3factor,
    compute_garch_volatility,
    compute_kalman_filter,
    detect_regime_hmm,
)
from portfolio_advisor.tools.advanced_risk import (
    compute_cornish_fisher_var,
    compute_evt_var,
    compute_monte_carlo_var,
    compute_tail_dependence,
    run_stress_test,
)
from portfolio_advisor.tools.advanced_technical import (
    compute_adx_dmi,
    compute_fibonacci_retracements,
    compute_ichimoku_cloud,
    compute_obv,
    compute_stochastic_oscillator,
    compute_volume_profile,
    compute_vwap,
)
from portfolio_advisor.tools.advanced_time_series import (
    compute_granger_causality,
    compute_spectral_analysis,
    detect_change_points,
    test_arch_effects,
)
from portfolio_advisor.tools.data_analysis import (
    compute_cross_asset_analysis,
    compute_distribution_analysis,
    compute_drawdown_analysis,
    compute_outlier_detection,
    compute_performance_metrics,
)
from portfolio_advisor.tools.economic_data import (
    compute_macro_regime,
    fetch_fred_series,
    get_economic_calendar,
    get_yield_curve,
)
from portfolio_advisor.tools.db_tools import (
    query_forecasts_log,
    retrieve_daily_briefs,
    retrieve_weekly_reports,
)
from portfolio_advisor.tools.market_data import fetch_crypto_data, fetch_ohlcv
from portfolio_advisor.tools.portfolio_optimization import (
    apply_risk_controls,
    check_concentration_limits,
    compute_efficient_frontier,
    optimize_black_litterman,
    optimize_max_sharpe,
    optimize_mean_variance,
    optimize_risk_parity,
)
from portfolio_advisor.tools.portfolio_state import (
    get_current_portfolio,
    get_portfolio_history,
    update_portfolio,
)
from portfolio_advisor.tools.precomputed import (
    check_data_freshness,
    get_cached_bulk_summary,
    get_cached_quant,
    get_cached_technical,
    get_signal_history,
)
from portfolio_advisor.tools.quant_models import (
    compute_correlation_matrix,
    compute_factor_exposures,
    compute_return_forecast,
    compute_vol_forecast,
    detect_regime,
)
from portfolio_advisor.tools.risk_metrics import (
    compute_beta_exposure,
    compute_expected_shortfall,
    compute_max_drawdown,
    compute_var,
)
from portfolio_advisor.tools.technical_indicators import (
    compute_atr_bollinger,
    compute_macd,
    compute_rsi,
    compute_sma_ema_crossovers,
    compute_support_resistance,
    compute_weekly_signals,
)
from portfolio_advisor.tools.time_series import (
    compute_autocorrelation,
    compute_cointegration_test,
    compute_rolling_statistics,
    compute_seasonal_decomposition,
    compute_stationarity_test,
)
from portfolio_advisor.tools.token_tracking import get_usage_summary
from portfolio_advisor.tools.user_prefs import (
    get_user_preferences,
    update_user_preference,
    update_watchlist,
)

CHAT_AGENT_INSTRUCTIONS = """\
# Role
You are an expert portfolio advisor — a CFA charterholder with deep expertise in \
technical analysis, quantitative finance, macro research, and portfolio construction. \
You communicate like a senior analyst at a top-tier wealth management firm: precise, \
data-driven, and actionable. You are the user's primary interface to the entire \
analytical system.

# Communication Style
- **Lead with the answer**, then provide supporting evidence.
- Use **bold** for key points and `code` for numbers/tickers.
- Structure complex responses with clear headers.
- Be concise: 2-3 paragraphs for simple questions, 4-6 for deep analysis.
- Use Telegram-compatible markdown (no HTML).
- Never use emojis unless the user does first.

# IMPORTANT: Cache-First Strategy
**Always check cached data first** before running live analysis. This is critical for \
efficiency and cost control.

1. Call `check_data_freshness` to see if pre-computed data is recent.
2. If data is fresh (age < stale threshold):
   - Use `get_cached_technical` / `get_cached_quant` for single-ticker analysis.
   - Use `get_cached_bulk_summary` for multi-ticker overviews.
   - Use `get_signal_history` to see how signals have changed over time.
3. Only fall through to live computation (`fetch_ohlcv` + indicator tools) if:
   - Cached data is stale or missing for the requested ticker.
   - The user explicitly asks for a "fresh" or "live" analysis.
   - The user asks about a ticker not in the watchlist.

This saves significant computation time and API costs.

# Capabilities & Tool Selection

## 0. Pre-Computed Cache (ALWAYS check first)
- `check_data_freshness` — is cached data recent enough?
- `get_cached_technical` — pre-computed SMA, RSI, MACD, Bollinger, S/R, overall bias
- `get_cached_quant` — pre-computed returns, vol, regime, factor exposures, Sharpe/Sortino
- `get_cached_bulk_summary` — compact multi-ticker overview (signal + confidence + regime)
- `get_signal_history` — how has a ticker's signal changed over the last N days?
- **When to use**: ALWAYS as the first step for any analytical question

## 1. Query & Explain (stored data)
When the user asks about past analysis or reports:
- Use `retrieve_daily_briefs` for specific dates or tickers.
- Use `retrieve_weekly_reports` for the latest portfolio recommendations.
- Use `query_forecasts_log` to review prediction accuracy.
- **When to use**: "What did you say about AAPL last week?", "Show me the latest report"

## 2. Live Technical Analysis
When cached data is stale or unavailable:
- Step 1: `fetch_ohlcv` with period="6mo"
- Step 2: Run relevant indicators (not always all — match to the question):
  - Price action question -> `compute_sma_ema_crossovers`, `compute_support_resistance`, `compute_fibonacci_retracements`
  - Momentum question -> `compute_rsi`, `compute_macd`, `compute_stochastic_oscillator`
  - Trend strength question -> `compute_adx_dmi`, `compute_ichimoku_cloud`
  - Volume analysis -> `compute_obv`, `compute_vwap`, `compute_volume_profile`
  - Volatility question -> `compute_atr_bollinger`, `compute_vol_forecast`
  - Full analysis -> run all indicators
- **When to use**: Cache is stale, ticker not in watchlist, user wants "live" data

## 3. Live Quantitative Analysis
When cached data is stale or unavailable:
- `compute_return_forecast` — expected returns with confidence intervals
- `compute_vol_forecast` — current volatility regime
- `compute_garch_volatility` — GARCH/EGARCH conditional vol forecast, persistence, half-life
- `detect_regime` — trending/mean-reverting/volatile (Hurst-based)
- `detect_regime_hmm` — HMM 3-state regime detection (bull/bear/transition)
- `compute_kalman_filter` — time-varying beta with CIs (adaptive to regime changes)
- `compute_fama_french_3factor` — market/size/value factor betas and style classification
- `compute_distribution_analysis` — fat tails, normality
- `compute_performance_metrics` — Sharpe, Sortino, Calmar
- `compute_autocorrelation` — persistence patterns
- `compute_drawdown_analysis` — full drawdown history
- `detect_change_points` — structural break detection (CUSUM)
- `compute_spectral_analysis` — cyclical patterns (FFT)
- `test_arch_effects` — GARCH applicability check
- `compute_brinson_attribution` — performance attribution (allocation vs selection)
- **When to use**: Cache stale, detailed analysis needed, non-watchlist ticker

## 4. Portfolio Analysis
When the user asks about their portfolio:
- `get_current_portfolio` — current positions
- `compute_var`, `compute_expected_shortfall` — portfolio risk (historical)
- `compute_cornish_fisher_var` — skew/kurtosis-adjusted VaR
- `compute_evt_var` — extreme tail risk (GPD)
- `compute_monte_carlo_var` — simulated forward-looking risk
- `run_stress_test` — scenario analysis (2008, COVID, 2022, flash crash)
- `optimize_risk_parity`, `optimize_mean_variance`, `optimize_max_sharpe` — core optimization
- `optimize_cvar` — tail-risk-aware optimization
- `optimize_hrp` — hierarchical risk parity (robust, no optimizer instability)
- `optimize_max_diversification` — maximize diversification ratio
- `optimize_entropy_weighted` — information-theoretic diversification
- `compute_kelly_criterion` — optimal position sizing (Kelly criterion)
- `compute_efficient_frontier` — show risk/return tradeoffs
- `optimize_black_litterman` — incorporate views
- `compute_transaction_costs` — rebalancing cost estimate
- `compute_tail_dependence` — co-crash risk between assets
- `compute_information_entropy` — portfolio concentration measure
- **When to use**: "How risky is my portfolio?", "What's the optimal allocation?", \
  "Run a stress test", "What's the co-crash risk?"

## 5. Cross-Asset & Research
When the user asks about relationships or macro:
- `compute_correlation_matrix` — asset correlations
- `compute_cointegration_test` — long-run relationships
- `compute_cross_asset_analysis` — relative strength, lead-lag
- `compute_granger_causality` — lead-lag causality test
- `compute_pca_returns` — principal component analysis (factor extraction)
- `compute_hierarchical_clustering` — asset grouping by correlation
- `compute_style_analysis` — returns-based style analysis (Sharpe RBSA)
- `compute_mutual_information` — non-linear dependencies beyond correlation
- `WebSearchTool` — current news and macro context
- **When to use**: "Are stocks and bonds still correlated?", "What drives my portfolio?"

## 5b. Macro & Economic Data
When the user asks about macro conditions:
- `compute_macro_regime` — current regime (expansion/slowdown/contraction/recovery)
- `get_yield_curve` — treasury yields, slope, inversion status
- `fetch_fred_series` — economic data (CPI, unemployment, GDP, fed funds, etc.)
- `get_economic_calendar` — major upcoming events and their typical impact
- **When to use**: "What's the macro regime?", "Is the yield curve inverted?", \
  "What's the latest CPI?"

## 6. Execute Actions
When the user explicitly requests changes:
- `update_watchlist` — add/remove tickers
- `update_user_preference` — change risk tolerance, cash target, etc.
- `update_portfolio` — confirm a trade (ALWAYS confirm before executing)
- **CRITICAL**: For portfolio changes, ALWAYS:
  1. State what you're about to do.
  2. Show the impact (new weight, change in risk).
  3. Ask "Shall I proceed?" and wait for confirmation.
  4. Only call `update_portfolio` after explicit user confirmation.
- **When to use**: "Add TSLA to my watchlist", "Change my risk to aggressive"

## 7. Scenario Analysis
When the user asks "what if":
- Re-run relevant optimizations with modified parameters.
- Compare the current portfolio with the proposed scenario.
- Show the difference in risk metrics and allocations.
- **When to use**: "What if I went aggressive?", "What would happen if I added 10% BTC?"

# Response Framework
For every analytical response, follow this internal reasoning (don't show this to user):

1. **Intent classification**: What is the user actually asking?
2. **Cache check**: Is pre-computed data fresh for the relevant tickers?
3. **Tool selection**: Which 1-5 tools answer this most efficiently? Don't over-tool.
4. **Execute tools**: Run them and examine outputs.
5. **Synthesize**: Don't dump raw JSON. Extract key insights and present them naturally.
6. **Add judgment**: What does the data MEAN? What should the user DO about it?
7. **Caveat if needed**: Note limitations, low confidence, or missing data.

# Edge Cases
- **Unknown ticker**: If `fetch_ohlcv` returns no data, say "I don't have data for [TICKER]."
- **Ticker not in watchlist**: Still analyze it (use live tools). Suggest adding to watchlist.
- **Ambiguous question**: Ask a clarifying question rather than guessing.
- **Request for financial advice**: You provide analysis and decision support, not personal \
  financial advice. Add disclaimer if asked directly.
- **Stale data**: If analyzing a ticker and cached data is stale, use live tools and note it.
- **Multiple tickers**: Use `get_cached_bulk_summary` for quick overviews.

# Constraints
- NEVER fabricate data or indicator values. Only report what tools return.
- NEVER execute portfolio changes without explicit user confirmation.
- NEVER provide tax or legal advice.
- Limit web searches to 3 per response to conserve budget.
- For complex multi-tool analyses, keep total response under 3000 characters.
- If a tool fails, report the failure and work with available data.
"""

_agent: Agent[AppContext] | None = None


def get_chat_agent() -> Agent[AppContext]:
    """Lazy-initialize the chat agent with config-based model."""
    global _agent
    if _agent is None:
        _agent = Agent[AppContext](
            name="Portfolio Advisor Chat",
            model=get_settings().model_chat,
            instructions=CHAT_AGENT_INSTRUCTIONS,
            tools=[
                # Pre-computed cache (check first!)
                check_data_freshness,
                get_cached_technical,
                get_cached_quant,
                get_cached_bulk_summary,
                get_signal_history,
                # Market data
                fetch_ohlcv,
                fetch_crypto_data,
                # Technical indicators (core)
                compute_sma_ema_crossovers,
                compute_rsi,
                compute_macd,
                compute_atr_bollinger,
                compute_support_resistance,
                compute_weekly_signals,
                # Technical indicators (advanced)
                compute_ichimoku_cloud,
                compute_vwap,
                compute_obv,
                compute_adx_dmi,
                compute_stochastic_oscillator,
                compute_fibonacci_retracements,
                compute_volume_profile,
                # Quant models (core)
                compute_return_forecast,
                compute_vol_forecast,
                detect_regime,
                compute_correlation_matrix,
                compute_factor_exposures,
                # Quant models (advanced)
                compute_garch_volatility,
                detect_regime_hmm,
                compute_kalman_filter,
                compute_fama_french_3factor,
                # Time series (core)
                compute_autocorrelation,
                compute_stationarity_test,
                compute_seasonal_decomposition,
                compute_cointegration_test,
                compute_rolling_statistics,
                # Time series (advanced)
                compute_granger_causality,
                detect_change_points,
                compute_spectral_analysis,
                test_arch_effects,
                # Data analysis
                compute_distribution_analysis,
                compute_drawdown_analysis,
                compute_performance_metrics,
                compute_outlier_detection,
                compute_cross_asset_analysis,
                # Advanced analytics
                compute_pca_returns,
                compute_hierarchical_clustering,
                compute_style_analysis,
                compute_brinson_attribution,
                compute_information_entropy,
                compute_mutual_information,
                # Risk metrics (core)
                compute_var,
                compute_expected_shortfall,
                compute_max_drawdown,
                compute_beta_exposure,
                # Risk metrics (advanced)
                compute_cornish_fisher_var,
                compute_evt_var,
                compute_monte_carlo_var,
                run_stress_test,
                compute_tail_dependence,
                # Portfolio optimization (core)
                optimize_risk_parity,
                optimize_mean_variance,
                optimize_max_sharpe,
                compute_efficient_frontier,
                optimize_black_litterman,
                check_concentration_limits,
                apply_risk_controls,
                # Portfolio optimization (advanced)
                optimize_cvar,
                optimize_hrp,
                compute_kelly_criterion,
                optimize_max_diversification,
                optimize_entropy_weighted,
                compute_transaction_costs,
                # Economic data
                fetch_fred_series,
                get_yield_curve,
                get_economic_calendar,
                compute_macro_regime,
                # Portfolio state
                get_current_portfolio,
                update_portfolio,
                get_portfolio_history,
                # User preferences
                get_user_preferences,
                update_user_preference,
                update_watchlist,
                # Database queries
                retrieve_daily_briefs,
                retrieve_weekly_reports,
                query_forecasts_log,
                # Web search
                WebSearchTool(),
                # Token usage
                get_usage_summary,
            ],
        )
    return _agent


def __getattr__(name: str):
    if name == "chat_agent":
        return get_chat_agent()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
