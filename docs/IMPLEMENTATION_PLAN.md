# Portfolio Advisor Agent - Implementation Plan

## Context

Build a 24/7 autonomous portfolio advisory system that monitors financial instruments and produces daily monitoring notes + weekly portfolio recommendation reports via Telegram. The system does NOT place trades -- it generates allocations and changes as decision support. It must combine deep quantitative/technical analysis with macro/news context, apply explicit risk controls, and maintain a full audit trail of every forecast and rationale. Runs on a $5-10/month VPS. Uses OpenAI Agents SDK for multi-agent orchestration, with `gpt-5.2` for weekly synthesis and `gpt-5-mini` for daily monitoring.

---

## Technology Stack

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Language | Python 3.11+ | Agents SDK is Python, pandas/numpy ecosystem |
| Agent Framework | `openai-agents-python` >=0.7.0 | Multi-agent orchestration with tools, handoffs, sessions, tracing |
| Weekly Model | `gpt-5.2` | Powerful reasoning for investment committee synthesis |
| Daily Model | `gpt-5-mini` | Cost-effective for routine monitoring/summarization |
| Market Data | yfinance (equities/ETFs/bonds/commodities) + CoinGecko free API (crypto) | Free, reliable, sufficient coverage |
| News/Research | OpenAI WebSearchTool (built into Agents SDK) | Simplest integration, most capable |
| Communication | python-telegram-bot >=22.0 | Mature async Telegram framework |
| Storage | SQLite via aiosqlite | Zero-config, single-file, fits VPS budget |
| Scheduling | APScheduler 4.x | Async-native, in-process, coalescing + misfire recovery |
| Deployment | systemd service on VPS | Simple, robust, auto-restart |

---

## Project Structure

```
finance-agent_claude/
├── pyproject.toml
├── .env.example
├── .gitignore
├── README.md
├── docs/
│   └── IMPLEMENTATION_PLAN.md
├── src/
│   └── portfolio_advisor/
│       ├── __init__.py
│       ├── main.py                     # Entry point: scheduler + telegram bot
│       ├── config.py                   # pydantic-settings, .env loading
│       ├── agents/
│       │   ├── __init__.py
│       │   ├── context.py              # AppContext dataclass for RunContextWrapper
│       │   ├── orchestrator.py         # Daily + Weekly orchestrator agents
│       │   ├── technical.py            # Technical analysis agent
│       │   ├── quantitative.py         # Quant/risk metrics agent
│       │   ├── research.py             # Macro/news research agent (WebSearchTool)
│       │   ├── portfolio.py            # Portfolio construction agent
│       │   ├── reporting.py            # Report synthesis agent
│       │   └── chat.py                 # Full-powered chat agent (ALL tools + live analysis + command execution)
│       ├── tools/
│       │   ├── __init__.py
│       │   ├── market_data.py          # yfinance + CoinGecko wrappers
│       │   ├── technical_indicators.py # SMA, EMA, RSI, MACD, ATR, Bollinger, S/R
│       │   ├── quant_models.py         # Return/vol forecasts, regime, correlations
│       │   ├── portfolio_optimization.py # Risk-parity, mean-variance, constraints
│       │   ├── risk_metrics.py         # VaR, ES, drawdown, beta/duration
│       │   ├── time_series.py          # Autocorrelation, decomposition, stationarity, cointegration
│       │   ├── data_analysis.py        # Distribution analysis, statistical tests, outlier detection
│       │   ├── db_tools.py             # Store/retrieve briefs, reports, forecasts
│       │   ├── portfolio_state.py      # Read/update current allocations
│       │   ├── user_prefs.py           # Read/update user preferences
│       │   └── token_tracking.py       # Token usage logging + budget enforcement
│       ├── models/
│       │   ├── __init__.py
│       │   ├── market.py               # TechnicalSignals, QuantMetrics, InstrumentAnalysis
│       │   ├── briefs.py               # DailyBrief, InstrumentBrief, ThemeBrief, PortfolioRiskSnapshot
│       │   ├── reports.py              # WeeklyReport, AllocationRecommendation, RiskAssessment
│       │   ├── portfolio.py            # PortfolioState, Position, RebalanceDelta
│       │   └── preferences.py          # UserPreferences, RiskTolerance enum
│       ├── db/
│       │   ├── __init__.py
│       │   ├── schema.py               # DDL for all tables
│       │   ├── connection.py           # aiosqlite connection manager
│       │   └── queries.py              # Typed CRUD functions
│       ├── telegram_bot/
│       │   ├── __init__.py
│       │   ├── bot.py                  # Application builder, handler registration
│       │   ├── commands.py             # /start /status /portfolio /prefs /set /watchlist etc.
│       │   ├── chat_handler.py         # Free-text -> Chat Agent routing
│       │   └── formatters.py           # Markdown formatting for Telegram messages
│       ├── scheduler/
│       │   ├── __init__.py
│       │   ├── jobs.py                 # daily_job(), weekly_job() implementations
│       │   └── runner.py               # APScheduler setup + lifecycle
│       └── utils/
│           ├── __init__.py
│           ├── logging.py              # Structured logging setup
│           └── rate_limiter.py         # Token budget + API rate limiting
├── tests/
│   ├── test_tools/
│   ├── test_agents/
│   ├── test_db/
│   └── test_telegram/
├── data/                               # Runtime (gitignored)
│   └── portfolio_advisor.db
└── deploy/
    ├── portfolio-advisor.service       # systemd unit file
    └── setup_vps.sh                    # VPS provisioning script
```

---

## Agent Architecture

Uses **agents-as-tools** pattern (not handoffs). The orchestrator calls each specialist agent as a tool, collects their structured JSON output, and coordinates the pipeline. This is a fan-out/fan-in pattern where the orchestrator retains control.

### Agent Definitions

| Agent | Model | Role | Key Tools |
|-------|-------|------|-----------|
| **Daily Orchestrator** | gpt-5-mini | Coordinates daily pipeline: calls specialists, stores results, sends Telegram | technical_agent.as_tool(), quant_agent.as_tool(), research_agent.as_tool(), portfolio_risk.as_tool(), store_daily_brief, send_telegram_summary |
| **Weekly Orchestrator** | gpt-5.2 | Coordinates weekly pipeline: retrieves week's data, calls portfolio+reporting agents | retrieve_weekly_briefs, portfolio_agent.as_tool(), reporting_agent.as_tool(), store_weekly_report, send_telegram_report |
| **Technical Analysis** | gpt-5-mini | Computes + interprets technical indicators | fetch_ohlcv, compute_sma_ema_crossovers, compute_rsi, compute_macd, compute_atr_bollinger, compute_support_resistance, compute_weekly_signals |
| **Quantitative** | gpt-5-mini | Computes + interprets quant metrics | compute_return_forecast, compute_vol_forecast, detect_regime, compute_correlation_matrix, compute_factor_exposures |
| **Research** | gpt-5-mini | Web search for macro/news context | WebSearchTool (built-in, rate-limited), get_watchlist_context |
| **Portfolio Construction** | gpt-5-mini (daily risk) / gpt-5.2 (weekly) | Produces allocation recommendations with risk controls | optimize_risk_parity, optimize_mean_variance, check_concentration_limits, apply_risk_controls, get_current_portfolio, get_user_preferences |
| **Reporting** | gpt-5.2 | Synthesizes "investment committee memo" | retrieve_daily_briefs_for_week, get_current_portfolio, get_user_preferences |
| **Chat** | gpt-5-mini | **Full-powered interactive agent** — can answer questions, execute commands (add/remove tickers, update prefs, confirm trades), run live technical/macro analysis on-demand, and reason beyond pre-computed scripts | **ALL tools from every specialist** — fetch_ohlcv, all technical indicators, all quant models, all risk metrics, portfolio optimization, WebSearchTool, update_watchlist, update_user_preference, update_portfolio, query_daily_briefs, query_weekly_reports, query_forecasts_log; uses SQLiteSession for conversation memory |

### Data Flow

```
DAILY (gpt-5-mini):
  APScheduler → Daily Orchestrator
    ├─ Technical Agent → TechnicalAnalysisResult (per ticker)
    ├─ Quantitative Agent → QuantAnalysisResult (per ticker)
    ├─ Research Agent → ResearchResult (news items with sources)
    └─ Portfolio Risk → PortfolioRiskMetrics
  → store_daily_brief (SQLite) → send_telegram_summary

WEEKLY (gpt-5.2):
  APScheduler → Weekly Orchestrator
    ├─ retrieve_weekly_briefs (from SQLite)
    ├─ Portfolio Construction Agent → PortfolioRecommendation
    └─ Reporting Agent → WeeklyReport (investment committee memo)
  → store_weekly_report (SQLite) → send_telegram_report

CHAT (gpt-5-mini, on-demand — FULL-POWERED):
  Telegram message → Chat Agent (with SQLiteSession)
    ├─ Can query stored briefs/reports/forecasts
    ├─ Can run LIVE technical analysis (fetch data + compute indicators)
    ├─ Can run LIVE macro research (WebSearchTool)
    ├─ Can run LIVE quant models (regime, vol, correlations)
    ├─ Can EXECUTE commands: add/remove tickers, update prefs, confirm trades
    ├─ Can run portfolio optimization and risk checks on-demand
    ├─ LLM applies its OWN reasoning on top of tool outputs (not just script relay)
    └─ Returns rich answer with analysis, citations, and any actions taken
```

### Shared Context

All agents typed as `Agent[AppContext]`. Tools access via `RunContextWrapper[AppContext]`.

```python
@dataclass
class AppContext:
    db_path: str
    telegram_chat_id: int
    run_date: date
    watchlist: list[str]
    token_budget_remaining: int
    web_search_calls_today: int = 0
    max_web_search_calls_daily: int = 20
```

---

## Tool Specifications

### Market Data (`tools/market_data.py`)
- `fetch_ohlcv(tickers, period, interval)` — yfinance for equities/ETFs, CoinGecko for crypto (BTC/ETH/SOL/AVAX). Caches in `price_cache` table.
- `fetch_crypto_data(coins, days)` — CoinGecko `/coins/{id}/market_chart`. Maps: BTC→bitcoin, ETH→ethereum, SOL→solana, AVAX→avalanche-2.

### Technical Indicators (`tools/technical_indicators.py`)
- `compute_sma_ema_crossovers(ticker, prices_json)` — SMA(50/200), EMA(12/26), golden/death cross, trend direction
- `compute_rsi(ticker, prices_json, period=14)` — RSI value, overbought/oversold, divergence
- `compute_macd(ticker, prices_json)` — MACD line, signal, histogram, crossover
- `compute_atr_bollinger(ticker, prices_json)` — ATR(14), Bollinger bands, bandwidth, %B
- `compute_support_resistance(ticker, prices_json)` — Pivot-based S1/S2/S3, R1/R2/R3
- `compute_weekly_signals(ticker, period="2y")` — Weekly-timeframe SMA/RSI/MACD for multi-TF confirmation

All use numpy/pandas. Return JSON with raw values + interpretation (bullish/bearish/neutral + confidence).

### Time Series Analysis (`tools/time_series.py`)
- `compute_autocorrelation(ticker, prices_json, max_lag)` — ACF/PACF with significance bounds, identifies mean-reversion/momentum timescales
- `compute_stationarity_test(ticker, prices_json)` — Augmented Dickey-Fuller test on returns and log-prices, unit root interpretation
- `compute_seasonal_decomposition(ticker, prices_json, period)` — Trend/seasonal/residual decomposition, identifies cyclical patterns
- `compute_cointegration_test(tickers, period)` — Engle-Granger cointegration for pairs, identifies long-run equilibria for pair trading
- `compute_rolling_statistics(ticker, prices_json, windows)` — Rolling mean, std, skewness, kurtosis across multiple windows

### Data Analysis (`tools/data_analysis.py`)
- `compute_distribution_analysis(ticker, prices_json)` — Return distribution: skewness, kurtosis, Jarque-Bera test, tail analysis, QQ deviation
- `compute_drawdown_analysis(ticker, prices_json)` — Full drawdown table: depth, duration, recovery, current position in drawdown cycle
- `compute_performance_metrics(ticker, prices_json, benchmark)` — Sharpe, Sortino, Calmar, Information Ratio, max drawdown, annualized return/vol
- `compute_outlier_detection(ticker, prices_json)` — Z-score and IQR outlier detection on returns, identifies anomalous days
- `compute_cross_asset_analysis(tickers, period)` — Rolling correlations, lead-lag relationships, relative strength

### Quantitative Models (`tools/quant_models.py`)
- `compute_return_forecast(ticker, prices_json)` — Momentum + mean-reversion blend, 1w/1m/3m horizons with confidence intervals
- `compute_vol_forecast(ticker, prices_json)` — EWMA vol, vol regime (low/normal/high), percentile vs 1Y
- `detect_regime(ticker, prices_json)` — Rolling Hurst exponent approx + volatility clustering → trending/mean-reverting/volatile
- `compute_correlation_matrix(tickers, period)` — NxN correlations, notable changes, diversification score
- `compute_factor_exposures(ticker, prices_json)` — Market beta, factor loadings via simple regression

### Portfolio Optimization (`tools/portfolio_optimization.py`)
- `optimize_risk_parity(tickers, vol_forecasts_json, correlation_json)` — Inverse-vol weighting adjusted for correlations
- `optimize_mean_variance(tickers, return_forecasts_json, vol_forecasts_json, correlation_json, risk_tolerance)` — scipy.optimize with constraints, risk aversion mapped from tolerance
- `optimize_max_sharpe(tickers, return_forecasts_json, vol_forecasts_json, correlation_json)` — Tangency portfolio maximizing Sharpe ratio
- `compute_efficient_frontier(tickers, return_forecasts_json, vol_forecasts_json, correlation_json, n_points)` — Generate n points along the efficient frontier for visualization
- `optimize_black_litterman(tickers, market_caps_json, views_json, vol_forecasts_json, correlation_json)` — Black-Litterman model incorporating analyst views with market equilibrium
- `check_concentration_limits(proposed_weights_json, max_position_pct)` — Validate against limits, return violations + adjusted weights
- `apply_risk_controls(proposed_weights_json, current_portfolio_json, risk_metrics_json, user_prefs_json)` — Full constraint pass: concentration, drawdown awareness, vol-aware sizing, excluded assets, cash target

### Risk Metrics (`tools/risk_metrics.py`)
- `compute_var(portfolio_weights_json, prices_json, confidence=0.95)` — Historical simulation VaR
- `compute_expected_shortfall(portfolio_weights_json, prices_json, confidence=0.95)` — Average loss beyond VaR
- `compute_max_drawdown(portfolio_weights_json, prices_json)` — Current + max drawdown, duration
- `compute_beta_exposure(portfolio_weights_json, prices_json)` — Portfolio beta vs SPY, sector breakdown, bond duration proxy

### Database Tools (`tools/db_tools.py`)
- `store_daily_brief(brief_json)` — Write DailyBrief to `daily_briefs` + denormalize to `instrument_briefs`
- `retrieve_daily_briefs(start_date, end_date, ticker?)` — Query by date range, optional ticker filter
- `store_weekly_report(report_json)` — Write WeeklyReport
- `retrieve_weekly_reports(count=4)` — Get N most recent reports
- `store_forecast(ticker, forecast_type, forecast_json)` — Log prediction for later evaluation
- `query_forecasts_log(ticker?, forecast_type?, start_date?, end_date?)` — Query historical forecasts

### Portfolio State (`tools/portfolio_state.py`)
- `get_current_portfolio()` — Current weights, cash%, last update
- `update_portfolio(ticker, new_weight_pct, reason)` — Update position (user-confirmed trades)
- `get_portfolio_history(days=30)` — Snapshots over time

### User Preferences (`tools/user_prefs.py`)
- `get_user_preferences()` — risk_tolerance, time_horizon, excluded_assets, allowed_regions, cash_target_pct, max_position_pct, watchlist
- `update_user_preference(key, value)` — Update single preference
- `update_watchlist(action, tickers)` — Add/remove tickers

---

## Database Schema (SQLite)

**9 tables:**

1. **`user_preferences`** — Single-row. risk_tolerance, time_horizon, excluded_assets (JSON), allowed_regions (JSON), cash_target_pct, max_position_pct, watchlist (JSON)
2. **`portfolio_state`** — One row per ticker (UNIQUE). weight_pct, asset_class, updated_at
3. **`portfolio_history`** — Snapshots. snapshot_date, state_json, trigger (user_confirmed/initial/weekly_rebalance)
4. **`daily_briefs`** — Per day. brief_date, content_json (full DailyBrief), market_summary, telegram_summary. Indexed on date.
5. **`instrument_briefs`** — Denormalized per-ticker per-day. ticker, signal, confidence, what_happened, why_it_matters, technical_json, quant_json, sources. Indexed on (date, ticker).
6. **`weekly_reports`** — Per week. week_ending, content_json (full WeeklyReport), executive_summary, allocations_json. Indexed on date.
7. **`price_cache`** — OHLCV cache. ticker, date, OHLCV, source. UNIQUE(ticker, date).
8. **`forecasts_log`** — Every prediction. forecast_date, ticker, forecast_type, horizon, predicted_value (JSON), actual_value (backfilled), was_correct (backfilled). Indexed on (ticker, date).
9. **`token_usage`** — Per-call. usage_date, model, run_type, input_tokens, output_tokens, estimated_cost_usd.

---

## Telegram Bot

### Commands
| Command | Description |
|---------|-------------|
| `/start` | Welcome + initialize defaults |
| `/status` | System status: last runs, next runs, token usage |
| `/portfolio` | Current allocations as formatted table |
| `/prefs` | Show current preferences |
| `/set <key> <value>` | Update preference (e.g., `/set risk_tolerance aggressive`) |
| `/watchlist` | Show current watchlist |
| `/addticker TSLA GOOG` | Add tickers |
| `/removeticker IWM` | Remove tickers |
| `/confirm <ticker> <weight>` | Confirm trade execution |
| `/brief` | Latest daily brief |
| `/report` | Latest weekly report |
| `/usage` | Token usage + cost |
| `/rundaily` | Force daily run now |
| `/runweekly` | Force weekly run now |
| `/help` | List all commands |

### Free-text Chat (Full-Powered Agent)
Any non-command message routes to the **Chat Agent** via `handle_chat_message`. The Chat Agent is the primary user interface and has access to **every tool in the system**:

- **Query & Explain**: Retrieve stored briefs, reports, forecasts; explain rationales with citations
- **Live Analysis**: Fetch real-time data, compute technical indicators, run quant models, perform web searches for macro context — the LLM reasons over the raw data itself, not just pre-computed summaries
- **Execute Actions**: Add/remove tickers from watchlist, update user preferences, confirm portfolio changes — the agent calls the same tools the bot commands use
- **Scenario Analysis**: "What if I increase risk tolerance?" → agent re-runs portfolio optimization with modified parameters and explains the difference
- **On-Demand Research**: "What's happening with NVDA?" → agent fetches live prices, computes indicators, searches news, and synthesizes a mini-brief on the spot

Uses `SQLiteSession` for conversation memory. The LLM combines tool outputs with its own financial reasoning — tools provide data and pre-computed signals, but the LLM synthesizes, contextualizes, and can identify patterns/implications the scripts don't explicitly model.

---

## Configuration (`.env`)

```bash
PA_OPENAI_API_KEY=sk-...
PA_TELEGRAM_BOT_TOKEN=123456:ABC-DEF...
PA_TELEGRAM_CHAT_ID=123456789
PA_DB_PATH=data/portfolio_advisor.db
PA_DAILY_RUN_HOUR=7           # UTC
PA_WEEKLY_RUN_DAY=sun
PA_WEEKLY_RUN_HOUR=18         # UTC
PA_DAILY_TOKEN_BUDGET=100000
PA_WEEKLY_TOKEN_BUDGET=200000
PA_MAX_WEB_SEARCHES_DAILY=20
```

---

## Default Watchlist

**Equities/Sectors:** SPY, QQQ, IWM, EFA, EEM, VNQ, XLE, XLK, XLF
**Bonds:** TLT, IEF, HYG
**Commodities:** GLD, SLV
**Large-cap:** AAPL, MSFT, NVDA, AMZN
**Crypto:** BTC, ETH, SOL, AVAX
**Starting portfolio:** 100% cash

Watchlist is dynamic — updatable by agent (research-driven) and user (Telegram commands).

---

## Key Design Decisions

- **Agents-as-tools (not handoffs)**: Orchestrator retains control, collects structured output from each specialist. Fan-out/fan-in pattern.
- **Denormalized `instrument_briefs`**: Fast per-ticker queries for chat agent without JSON extraction.
- **APScheduler in-process**: No external cron dependency. Async-native. Coalescing handles VPS restarts.
- **gpt-5-mini for daily, gpt-5.2 for weekly only**: Cost control. Daily is routine; weekly needs nuanced judgment.
- **SQLite**: Zero-config, single file, easily backed up, sufficient for years of data at this volume.
- **Forecasts log with backfill columns**: `actual_value` and `was_correct` filled retroactively. Enables future self-improvement without complex pipelines.
- **LLM-first analysis, tools as foundation**: Tools (technical indicators, quant models, risk metrics) provide the quantitative substrate — reliable, reproducible calculations. But the LLM layer adds interpretation, cross-asset reasoning, macro synthesis, and nuanced judgment. The Chat Agent and the specialist agents don't just relay script outputs; they reason over the data, identify implications, and form views. This means every agent prompt instructs the model to think independently, challenge signals when context warrants it, and flag when quantitative signals conflict with macro reality.
- **Chat Agent = full analyst, not a dashboard**: The Telegram chat agent has access to every tool in the system. It can run live analysis on demand, execute administrative commands (watchlist, prefs, portfolio updates), and perform scenario analysis. The user interacts with an intelligent analyst, not a command router.
