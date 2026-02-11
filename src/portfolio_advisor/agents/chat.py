"""Chat agent — interactive analyst with hierarchical tool architecture.

v3: Reduced from 86 flat tools to ~33 tools using agents-as-tools delegation.
Direct tools handle cache queries, market data, earnings, portfolio state, and preferences.
Deep analysis (technical, quantitative, portfolio optimization, research) is delegated to
specialist agents via .as_tool(), hiding their 60+ tools from the chat agent's context.
"""

from __future__ import annotations

from agents import Agent, WebSearchTool

from portfolio_advisor.agents.context import AppContext
from portfolio_advisor.config import get_settings

# ── Direct tools (lightweight, instant access) ────────────────────────────────

# Pre-computed cache (always check first)
from portfolio_advisor.tools.precomputed import (
    check_data_freshness,
    get_cached_bulk_summary,
    get_cached_correlations,
    get_cached_macro,
    get_cached_quant,
    get_cached_technical,
    get_daily_analysis_snapshot,
    get_intraday_changes,
    get_indicator_trend,
    get_signal_history,
)

# Market data
from portfolio_advisor.tools.market_data import fetch_crypto_data, fetch_ohlcv

# Economic / macro data
from portfolio_advisor.tools.economic_data import (
    compute_macro_regime,
    fetch_fred_series,
    get_economic_calendar,
    get_yield_curve,
)

# Earnings
from portfolio_advisor.tools.earnings import (
    get_earnings_results,
    get_upcoming_earnings,
)

# Portfolio state
from portfolio_advisor.tools.portfolio_state import (
    get_current_portfolio,
    get_portfolio_history,
    update_portfolio,
)

# User preferences
from portfolio_advisor.tools.user_prefs import (
    get_user_preferences,
    update_user_preference,
    update_watchlist,
)

# Database queries
from portfolio_advisor.tools.db_tools import (
    query_forecasts_log,
    retrieve_daily_briefs,
    retrieve_weekly_reports,
)

# Token tracking
from portfolio_advisor.tools.token_tracking import get_usage_summary


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
**Always check cached data first** before running live analysis.

1. Call `check_data_freshness` to see if pre-computed data is recent.
2. If data is fresh (age < stale threshold):
   - Use `get_cached_technical` / `get_cached_quant` for single-ticker analysis.
   - Use `get_cached_bulk_summary` for multi-ticker overviews.
   - Use `get_signal_history` to see how signals changed over time.
   - Use `get_daily_analysis_snapshot` for the full pre-computed narrative with portfolio context.
   - Use `get_intraday_changes` to compare morning vs midday signals.
   - Use `get_cached_macro` for macro regime, VIX, yield curve snapshot.
   - Use `get_cached_correlations` for diversification and correlation data.
3. Only delegate to specialist agents if:
   - Cached data is stale or missing for the requested ticker.
   - The user explicitly asks for a "fresh" or "live" analysis.
   - The user asks about a ticker not in the watchlist.
   - The question requires deep specialist analysis (optimization, stress tests, etc.).

# Tool Architecture

## Direct Tools (instant, no LLM cost)
You have direct access to pre-computed cache, market data, earnings, macro data, \
portfolio state, preferences, and database queries. Use these for most questions.

## Specialist Agent Delegates (for deep analysis)
For complex analysis, delegate to specialist agents. Each agent has its own full \
toolkit hidden from you — just describe what you need:

- **`run_technical_analysis`**: Full technical analysis (SMA/EMA, RSI, MACD, Bollinger, \
  ADX, Ichimoku, Stochastic, Fibonacci, VWAP, OBV, S/R, weekly signals). \
  Use when: cache is stale, non-watchlist ticker, user wants "live" technical data. \
  Pass: ticker list as natural language.

- **`run_quantitative_analysis`**: Quantitative models (return/vol forecasts, GARCH, \
  HMM regime detection, Kalman filter, Fama-French, correlations, factor exposures, \
  time series analysis, PCA, clustering, style analysis). \
  Use when: deep quant analysis requested, advanced models needed, non-watchlist ticker. \
  Pass: ticker list and specific analysis request.

- **`run_portfolio_analysis`**: Portfolio optimization and risk (mean-variance, risk parity, \
  HRP, CVaR, Black-Litterman, Kelly criterion, max diversification, efficient frontier, \
  VaR/ES, Monte Carlo, stress tests, Cornish-Fisher, EVT, tail dependence, transaction costs). \
  Use when: "optimize my portfolio", "run a stress test", "what's my VaR", risk assessment. \
  Pass: detailed context about positions, preferences, and what analysis is needed.

- **`run_market_research`**: Structured web research with source tiering and impact \
  classification (Tier 1/2/3 sources, high/medium/low impact). \
  Use when: "what's the latest news", specific event research, macro developments. \
  Pass: watchlist tickers and specific research focus.

# Earnings Awareness
- Use `get_upcoming_earnings` to check for earnings in the next 14 days.
- Use `get_earnings_results` for a ticker's earnings history and surprise data.
- When discussing a ticker, note if it has earnings coming up.
- For post-earnings analysis, delegate to `run_market_research` for analyst reactions.

# Query & Explain (stored data)
When the user asks about past analysis or reports:
- Use `retrieve_daily_briefs` for specific dates or tickers.
- Use `retrieve_weekly_reports` for the latest portfolio recommendations.
- Use `query_forecasts_log` to review prediction accuracy.

# Execute Actions
When the user explicitly requests changes:
- `update_watchlist` — add/remove tickers
- `update_user_preference` — change risk tolerance, cash target, etc.
- `update_portfolio` — confirm a trade
- **CRITICAL**: For portfolio changes, ALWAYS:
  1. State what you're about to do.
  2. Show the impact (new weight, change in risk).
  3. Ask "Shall I proceed?" and wait for confirmation.
  4. Only call `update_portfolio` after explicit user confirmation.

# Response Framework
1. **Intent classification**: What is the user actually asking?
2. **Cache check**: Is pre-computed data fresh for the relevant tickers?
3. **Tool selection**: Which tools or delegates answer this most efficiently?
4. **Execute**: Run tools and examine outputs.
5. **Synthesize**: Don't dump raw JSON. Extract key insights and present naturally.
6. **Add judgment**: What does the data MEAN? What should the user DO about it?
7. **Caveat if needed**: Note limitations, low confidence, or missing data.

# Edge Cases
- **Unknown ticker**: If `fetch_ohlcv` returns no data, say "I don't have data for [TICKER]."
- **Ticker not in watchlist**: Analyze it (delegate to specialist). Suggest adding to watchlist.
- **Ambiguous question**: Ask a clarifying question rather than guessing.
- **Request for financial advice**: You provide analysis and decision support, not personal \
  financial advice. Add disclaimer if asked directly.
- **Stale data**: If cached data is stale, delegate to specialist agents and note it.
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
    """Lazy-initialize the chat agent with config-based model.

    v3 architecture: ~33 direct tools + 4 agent-as-tool delegates.
    Down from 86 flat tools — reduces tool schema tokens by ~75%.
    """
    global _agent
    if _agent is None:
        # Lazy-import specialist agents for .as_tool() delegation
        from portfolio_advisor.agents.portfolio import get_portfolio_agent
        from portfolio_advisor.agents.quantitative import get_quantitative_agent
        from portfolio_advisor.agents.research import get_research_agent
        from portfolio_advisor.agents.technical import get_technical_agent

        _agent = Agent[AppContext](
            name="Portfolio Advisor Chat",
            model=get_settings().model_chat,
            instructions=CHAT_AGENT_INSTRUCTIONS,
            tools=[
                # ── Pre-computed cache (check first!) ─────────────────────
                check_data_freshness,
                get_cached_technical,
                get_cached_quant,
                get_cached_bulk_summary,
                get_signal_history,
                get_intraday_changes,
                get_indicator_trend,
                get_daily_analysis_snapshot,
                get_cached_macro,
                get_cached_correlations,
                # ── Market data ───────────────────────────────────────────
                fetch_ohlcv,
                fetch_crypto_data,
                # ── Economic / macro data ─────────────────────────────────
                fetch_fred_series,
                get_yield_curve,
                get_economic_calendar,
                compute_macro_regime,
                # ── Earnings ──────────────────────────────────────────────
                get_upcoming_earnings,
                get_earnings_results,
                # ── Portfolio state ───────────────────────────────────────
                get_current_portfolio,
                update_portfolio,
                get_portfolio_history,
                # ── User preferences ──────────────────────────────────────
                get_user_preferences,
                update_user_preference,
                update_watchlist,
                # ── Database queries ──────────────────────────────────────
                retrieve_daily_briefs,
                retrieve_weekly_reports,
                query_forecasts_log,
                # ── Web search ────────────────────────────────────────────
                WebSearchTool(),
                # ── Token tracking ────────────────────────────────────────
                get_usage_summary,
                # ── Specialist agent delegates (deep analysis) ────────────
                get_technical_agent().as_tool(
                    tool_name="run_technical_analysis",
                    tool_description=(
                        "Run full technical analysis (SMA/EMA, RSI, MACD, Bollinger, "
                        "ADX, Ichimoku, Stochastic, Fibonacci, VWAP, OBV, S/R, weekly "
                        "signals) on specified tickers. Pass tickers as a comma-separated "
                        "list in a natural language request."
                    ),
                ),
                get_quantitative_agent().as_tool(
                    tool_name="run_quantitative_analysis",
                    tool_description=(
                        "Run quantitative analysis (return/vol forecasts, GARCH, HMM "
                        "regime detection, Kalman filter, Fama-French 3-factor, "
                        "correlations, factor exposures, time series analysis, PCA, "
                        "clustering, style analysis) on specified tickers. Pass tickers "
                        "and describe the specific analysis needed."
                    ),
                ),
                get_portfolio_agent().as_tool(
                    tool_name="run_portfolio_analysis",
                    tool_description=(
                        "Run portfolio optimization and risk analysis (mean-variance, "
                        "risk parity, HRP, CVaR, Black-Litterman, Kelly criterion, "
                        "max diversification, efficient frontier, VaR/ES, Monte Carlo, "
                        "stress tests, Cornish-Fisher, EVT, tail dependence, transaction "
                        "costs). Provide detailed context about positions and preferences."
                    ),
                ),
                get_research_agent().as_tool(
                    tool_name="run_market_research",
                    tool_description=(
                        "Run structured market research with source tiering and impact "
                        "classification. Searches for macro, sector, and ticker-specific "
                        "news. Pass watchlist tickers and any specific research focus."
                    ),
                ),
            ],
        )
    return _agent


def __getattr__(name: str):
    if name == "chat_agent":
        return get_chat_agent()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
