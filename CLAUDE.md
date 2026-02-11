# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install (editable, with dev deps)
pip install -e ".[dev]"

# Run the application (starts Telegram bot + APScheduler)
portfolio-advisor

# Lint (use venv binary — ruff is not on system PATH)
.venv/bin/ruff check .
.venv/bin/ruff check --fix .
.venv/bin/ruff format .

# Test
.venv/bin/pytest                          # all tests
.venv/bin/pytest tests/test_tools/        # one directory
.venv/bin/pytest tests/test_tools/test_market_data.py  # one file
.venv/bin/pytest -k "test_fetch_ohlcv"    # one test by name

# Verify imports (quick smoke test — requires PA_ env vars)
PA_OPENAI_API_KEY=test PA_TELEGRAM_BOT_TOKEN=test PA_TELEGRAM_CHAT_ID=123 \
  .venv/bin/python -c "from portfolio_advisor.agents.chat import get_chat_agent; a = get_chat_agent(); print(f'{len(a.tools)} tools')"
```

Ruff config: `target-version = "py311"`, `line-length = 100`. Pytest uses `asyncio_mode = "auto"`.

## Architecture

Multi-agent portfolio advisory system using the **OpenAI Agents SDK** (`openai-agents-python`). Runs automated daily/weekly/midday analysis pipelines via APScheduler and an interactive Telegram chat interface. Does NOT place trades.

### Agents-as-Tools Pattern

Orchestrators call specialist agents via `.as_tool()` (fan-out/fan-in). The orchestrator retains control and synthesizes results. All agents use lazy initialization via `get_*_agent()` factory functions with config-based model selection.

```
Pre-compute Pipeline (scheduler, no LLM)
└── run_precompute_pipeline() — batch technical + quant for watchlist

Daily Orchestrator (gpt-5.2)
├── Technical Agent.as_tool() — 14 tools (core + advanced indicators)
├── Quantitative Agent.as_tool() — 23 tools (core + GARCH/HMM/Kalman/FF3 + time series + analytics)
└── Research Agent.as_tool() — web search

Weekly Orchestrator (gpt-5.2)
├── Portfolio Agent.as_tool() — 24 tools (core + CVaR/HRP/Kelly + advanced risk)
└── Reporting Agent.as_tool()

Chat Agent (gpt-5.2) — 86 tools, cache-first strategy
Onboarding Agent (gpt-5-mini) — 6 tools, guided 5-step new-user setup

News Alert Pipeline (midday + evening jobs)
└── Research Agent → theme comparison → Telegram alert for HIGH impact
```

### Scheduler (7 Jobs)

| Job | Time (UTC) | Function |
|-----|-----------|----------|
| precompute_morning | 06:00 | `precompute_job()` |
| daily_monitoring | 06:30 | `daily_job()` |
| precompute_midday | 13:00 | `precompute_job()` |
| midday_update | 13:30 | `midday_update_job()` |
| evening_summary | 20:00 | `evening_summary_job()` |
| weekly_report | Sun 18:00 | `weekly_job()` |
| forecast_eval | 22:00 | `forecast_evaluation_job()` |

### Pre-Compute Pipeline

`tools/precomputed.py` runs 2-3x daily via scheduler. Batch-fetches OHLCV for the entire watchlist, computes all technical indicators (12 indicators including Ichimoku, VWAP, OBV, ADX, Stochastic, Fibonacci) and quant metrics (GARCH vol, HMM state, Kalman beta), stores in `technical_indicators` and `quant_metrics` tables.

The chat agent checks cache freshness first (`check_data_freshness`) and uses cached data (`get_cached_technical`, `get_cached_quant`, `get_cached_bulk_summary`) before falling through to live computation.

### Tool Pattern

Tools follow a two-layer pattern: pure `_raw()` computation functions (for pipeline reuse) and thin `@function_tool` wrappers (for agent access).

```python
# Pure function (called by precompute pipeline directly)
def compute_rsi_raw(df: pd.DataFrame, period: int = 14) -> dict:
    ...

# Tool wrapper (called by agents via @function_tool)
@function_tool
async def compute_rsi(ctx: RunContextWrapper[AppContext], ticker: str, prices_json: str) -> str:
    df = _prices_to_series(prices_json)
    raw = compute_rsi_raw(df)
    return json.dumps({"ticker": ticker, **raw})
```

### Shared Context

All agents typed as `Agent[AppContext]`. Every `@function_tool` receives `ctx: RunContextWrapper[AppContext]` as first parameter and accesses the database via `ctx.context.db_path`. The `AppContext` dataclass (`agents/context.py`) carries db_path, telegram_chat_id, run_date, watchlist, and token budget state.

### Database Layer

SQLite via aiosqlite. Three files: `db/schema.py` (DDL), `db/connection.py` (`init_db` + `get_db` context manager), `db/queries.py` (typed CRUD). 18 tables:

- **v1**: user_preferences, portfolio_state, portfolio_history, daily_briefs, instrument_briefs, weekly_reports, price_cache, forecasts_log, token_usage
- **v2**: technical_indicators, quant_metrics, daily_risk_metrics, research_themes, forecast_accuracy, onboarding_state, chat_history, analysis_runs

### Runtime Flow

`main.py` → `init_db()` → `setup_scheduler()` + `start_scheduler()` → `build_application().run_polling()`. APScheduler 4.x (`AsyncScheduler`) fires 7 scheduled jobs. Telegram free-text routes to the onboarding agent (if setup incomplete) or chat agent via `chat_handler.py` with persistent DB-backed conversation history. Midday and evening jobs include a news alert pipeline that detects and alerts on new high-impact research themes.

### Key Files

- `agents/chat.py` — Chat agent with 86 tools (the primary user interface)
- `agents/orchestrator.py` — Daily + Weekly orchestrator definitions
- `agents/portfolio.py` — Portfolio agent with 24 tools (core + advanced optimization + risk)
- `agents/quantitative.py` — Quant agent with 23 tools (core + time series + analytics)
- `tools/precomputed.py` — Pre-compute pipeline + 5 cache query tools
- `tools/advanced_technical.py` — Ichimoku, VWAP, OBV, ADX, Stochastic, Fibonacci, Volume Profile
- `tools/advanced_quant.py` — GARCH, HMM, Kalman filter, Fama-French 3-factor
- `tools/advanced_portfolio.py` — CVaR, HRP, Kelly Criterion, Max Diversification, Entropy, Transaction Costs
- `tools/advanced_risk.py` — Cornish-Fisher VaR, EVT, Monte Carlo VaR, Stress Testing, Tail Dependence
- `tools/advanced_time_series.py` — Granger Causality, Change Points, Spectral Analysis, ARCH Test
- `tools/advanced_analytics.py` — PCA, Clustering, Style Analysis, Brinson Attribution, Entropy, Mutual Info
- `tools/economic_data.py` — FRED integration, Yield Curve, Economic Calendar, Macro Regime
- `agents/onboarding.py` — 5-step guided new-user setup (risk, watchlist, portfolio, preferences)
- `scheduler/runner.py` — 7 scheduled jobs configuration
- `scheduler/jobs.py` — Job implementations (precompute, daily, midday, evening, weekly, forecast eval)
- `scheduler/alerts.py` — News alert pipeline (research agent → theme detection → Telegram alerts)
- `telegram_bot/bot.py` — 13 command handlers + free-text chat handler
- `telegram_bot/chat_handler.py` — Routes to onboarding or chat agent based on onboarding state
- `config.py` — pydantic-settings with `PA_` prefix, per-agent model assignments

### Agent Prompt Convention

Each agent file has a `*_INSTRUCTIONS` string constant with structured sections: Role, Procedure (numbered steps), Output Format (JSON schema), Edge Cases, and Constraints. Agent prompts instruct the LLM to reason over tool outputs, not just relay them.

## Configuration

Copy `.env.example` to `.env`. Required: `PA_OPENAI_API_KEY`, `PA_TELEGRAM_BOT_TOKEN`, `PA_TELEGRAM_CHAT_ID`. All settings use the `PA_` prefix.

Key optional settings: `PA_MODEL_CHAT`, `PA_MODEL_TECHNICAL`, etc. for per-agent model selection (default: `gpt-5.2`). `PA_PRECOMPUTE_STALE_HOURS` controls cache freshness threshold. `PA_FRED_API_KEY` enables FRED economic data (optional — yield curve and macro regime work without it via yfinance). `PA_ONBOARDING_ENABLED` controls whether new users are guided through setup (default: `true`).
