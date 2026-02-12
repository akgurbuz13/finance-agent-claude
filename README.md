# Portfolio Advisor

24/7 autonomous portfolio advisory system that monitors financial instruments and delivers daily market briefs, weekly portfolio recommendations, and on-demand interactive analysis via Telegram.

Built with the [OpenAI Agents SDK](https://github.com/openai/openai-agents-python) using a multi-agent orchestration architecture. Runs on a VPS via APScheduler + Telegram bot.

**Does NOT place trades** -- generates allocations, risk analysis, and decision support only.

---

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Data Provider Setup](#data-provider-setup)
- [VPS Deployment](#vps-deployment)
- [Telegram Commands](#telegram-commands)
- [Testing](#testing)
- [Architecture](#architecture)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Overview

Portfolio Advisor is a multi-agent system that runs continuously on a VPS, performing automated market analysis and delivering insights through Telegram. It combines technical analysis, quantitative modeling, macro research, and portfolio optimization into a unified advisory pipeline.

### Key Features

- **Multi-agent orchestration** -- Specialist agents (technical, quantitative, portfolio, research) coordinated via agents-as-tools pattern with fan-out/fan-in control flow
- **Pre-computed analysis pipeline** -- Batch computation runs 2-3x daily (morning, midday, evening), caching results for instant retrieval at zero LLM cost
- **12 technical indicators** -- SMA/EMA, RSI, MACD, Bollinger Bands, ATR, Ichimoku, VWAP, OBV, ADX, Stochastic, Fibonacci, Volume Profile
- **Quantitative models** -- GARCH volatility, HMM regime detection, Kalman filter beta, Fama-French 3-factor
- **Portfolio optimization** -- CVaR, Hierarchical Risk Parity (HRP), Kelly Criterion, Max Diversification, Mean-Variance, Black-Litterman, Risk Parity
- **Advanced risk analytics** -- Cornish-Fisher VaR, Extreme Value Theory (EVT), Monte Carlo VaR, Stress Testing, Tail Dependence
- **Multi-provider data with fallback chains** -- Massive (Polygon.io rebrand), Alpha Vantage, FRED, yfinance, CoinGecko with automatic failover and round-robin key rotation
- **Earnings monitoring** -- Calendar tracking with upcoming earnings alerts
- **News alert pipeline** -- 5x daily news coverage with high-impact theme detection and Telegram alerts
- **Interactive chat** -- Free-text Telegram conversations with a 39-tool chat agent backed by 4 specialist agent delegates
- **Guided onboarding** -- 5-step new-user setup (risk profile, watchlist, portfolio, preferences, confirmation)
- **Health monitoring** -- HTTP health/status server on port 8080 with liveness probe and detailed system status

### What It Does NOT Do

- Execute trades or interact with brokerage APIs
- Provide personalized investment advice (it is a decision-support tool)
- Guarantee returns or make forward-looking promises

---

## Quick Start

### Prerequisites

- Python 3.11+
- pip
- An OpenAI API key
- A Telegram bot token and chat ID

### Installation

```bash
git clone https://github.com/alikaangurbuz/finance-agent-claude.git
cd finance-agent-claude

python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### Environment Setup

```bash
cp .env.example .env
```

Edit `.env` with your credentials:

```bash
PA_OPENAI_API_KEY=sk-...
PA_TELEGRAM_BOT_TOKEN=123456:ABC-DEF...
PA_TELEGRAM_CHAT_ID=123456789
```

### Run

```bash
portfolio-advisor
```

This starts the Telegram bot (interactive chat) and APScheduler (automated daily/weekly/midday pipelines) together in a single process. The database is automatically initialized on first run.

---

## Configuration

All settings use the `PA_` prefix and are loaded from environment variables or a `.env` file via pydantic-settings.

### Required

| Variable | Description |
|----------|-------------|
| `PA_OPENAI_API_KEY` | OpenAI API key for all agent LLM calls |
| `PA_TELEGRAM_BOT_TOKEN` | Telegram bot token from BotFather |
| `PA_TELEGRAM_CHAT_ID` | Telegram chat ID to send automated reports to |

### Data Providers (Optional)

All data provider keys are optional. The system degrades gracefully with yfinance/CoinGecko fallbacks when premium providers are unavailable.

| Variable | Description |
|----------|-------------|
| `PA_MASSIVE_API_KEYS` | Comma-separated Massive (Polygon.io) API keys for round-robin rotation. 5 calls/min per key. Used for earnings, news, fundamentals, yields, short interest, dividends |
| `PA_MASSIVE_API_KEY` | Legacy single-key field (still supported as fallback) |
| `PA_ALPHA_VANTAGE_API_KEYS` | Comma-separated Alpha Vantage API keys. 25 calls/day per key, 5/min. Used as fundamentals fallback |
| `PA_ALPHA_VANTAGE_API_KEY` | Legacy single-key field (still supported as fallback) |
| `PA_FRED_API_KEY` | FRED API key for treasury yields, VIX, credit spread. 120 calls/min |

### Model Selection (Optional)

Per-agent model assignments. All default to `gpt-5.2` unless noted.

| Variable | Default | Agent |
|----------|---------|-------|
| `PA_MODEL_ORCHESTRATOR` | `gpt-5.2` | Daily and Weekly orchestrators |
| `PA_MODEL_TECHNICAL` | `gpt-5.2` | Technical analysis agent |
| `PA_MODEL_QUANTITATIVE` | `gpt-5.2` | Quantitative modeling agent |
| `PA_MODEL_PORTFOLIO` | `gpt-5.2` | Portfolio construction agent |
| `PA_MODEL_REPORTING` | `gpt-5.2` | Weekly reporting agent |
| `PA_MODEL_CHAT` | `gpt-5.2` | Interactive chat agent |
| `PA_MODEL_RESEARCH` | `gpt-5-mini` | Research agent (web search) |
| `PA_MODEL_ONBOARDING` | `gpt-5-mini` | Onboarding agent |
| `PA_MODEL_DAILY_SYNTHESIS` | `gpt-5.2` | Daily synthesis agent |
| `PA_WEEKLY_MODEL` | `gpt-5.2` | Legacy weekly model (backward compat) |
| `PA_DAILY_MODEL` | `gpt-5-mini` | Legacy daily model (backward compat) |

### Scheduler (Optional)

| Variable | Default | Description |
|----------|---------|-------------|
| `PA_DAILY_RUN_HOUR` | `7` | Daily brief generation hour (UTC) |
| `PA_MORNING_RUN_HOUR` | `6` | Morning pre-compute hour (UTC) |
| `PA_MIDDAY_RUN_HOUR` | `13` | Midday pre-compute + update hour (UTC) |
| `PA_EVENING_RUN_HOUR` | `20` | Evening summary hour (UTC) |
| `PA_WEEKLY_RUN_DAY` | `sun` | Weekly report day |
| `PA_WEEKLY_RUN_HOUR` | `18` | Weekly report hour (UTC) |
| `PA_NEWS_CHECK_HOURS` | `[9, 15]` | Additional news check hours (UTC), JSON array |

### Other (Optional)

| Variable | Default | Description |
|----------|---------|-------------|
| `PA_DB_PATH` | `data/portfolio_advisor.db` | SQLite database file path |
| `PA_DAILY_TOKEN_BUDGET` | `100000` | Max tokens per daily run |
| `PA_WEEKLY_TOKEN_BUDGET` | `200000` | Max tokens per weekly run |
| `PA_MAX_WEB_SEARCHES_DAILY` | `20` | Max web searches per day |
| `PA_PRECOMPUTE_ENABLED` | `true` | Enable/disable pre-compute pipeline |
| `PA_PRECOMPUTE_STALE_HOURS` | `8.0` | Cache freshness threshold (hours) |
| `PA_ONBOARDING_ENABLED` | `true` | Enable guided new-user onboarding flow |
| `PA_HEALTH_PORT` | `8080` | HTTP health check server port |

### Default Watchlist

The default watchlist covers major asset classes and is customizable per user via Telegram commands:

| Category | Tickers |
|----------|---------|
| US Equities/Sectors | SPY, QQQ, IWM, EFA, EEM, VNQ, XLE, XLK, XLF |
| Bonds | TLT, IEF, HYG |
| Commodities | GLD, SLV |
| Large-cap Stocks | AAPL, MSFT, NVDA, AMZN |
| Crypto | BTC, ETH, SOL, AVAX |

---

## Data Provider Setup

### OpenAI (Required)

Powers all agent LLM calls. Sign up at [platform.openai.com](https://platform.openai.com) and create an API key.

### Telegram (Required)

1. **Create a bot**: Message [@BotFather](https://t.me/BotFather) on Telegram, send `/newbot`, follow the prompts. Copy the bot token.
2. **Get your chat ID**: Message [@userinfobot](https://t.me/userinfobot) on Telegram. It will reply with your numeric chat ID.
3. **Start your bot**: Send `/start` to your new bot so it can message you.

### Massive / Polygon.io (Optional)

Provides earnings calendars, news, fundamentals, treasury yields, short interest, dividends, and analyst ratings.

- Sign up at [massivecorp.com](https://massivecorp.com)
- Free tier: 5 API calls per minute per key
- Supports multiple keys for round-robin rotation (recommended): set `PA_MASSIVE_API_KEYS=key1,key2,key3,key4`
- With 4 keys, effective rate is 20 calls/min

### Alpha Vantage (Optional)

Fundamentals fallback provider when Massive is unavailable.

- Sign up at [alphavantage.co](https://www.alphavantage.co)
- Free tier: 25 API calls per day per key, 5 calls per minute
- Supports multiple keys: set `PA_ALPHA_VANTAGE_API_KEYS=key1,key2`

### FRED (Optional)

Federal Reserve Economic Data for treasury yields, VIX, and credit spreads.

- Get a free API key at [fred.stlouisfed.org/docs/api/api_key.html](https://fred.stlouisfed.org/docs/api/api_key.html)
- Rate limit: 120 calls per minute
- Set `PA_FRED_API_KEY=your_key`

### yfinance + CoinGecko (Built-in, No Key Required)

- **yfinance**: Equity and ETF OHLCV data, basic fundamentals. Always available as fallback.
- **CoinGecko**: Cryptocurrency price data. No API key required for the free tier.

### Fallback Chains

The provider registry implements automatic failover per data type:

| Data Type | Fallback Order |
|-----------|---------------|
| Treasury Yields | Massive -> FRED -> yfinance |
| VIX | FRED -> yfinance |
| Credit Spread | FRED -> ETF proxy |
| Earnings | Massive -> yfinance |
| News | Massive -> Web Search |
| Fundamentals | Massive -> Alpha Vantage |
| OHLCV (Equities) | yfinance (primary) |
| OHLCV (Crypto) | CoinGecko (primary) |

---

## VPS Deployment

### Recommended Specs

- **CPU**: 1 vCPU
- **RAM**: 2 GB
- **Disk**: 20 GB SSD
- **OS**: Ubuntu 22.04 LTS

### Step 1: Server Setup

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python 3.11+ and dependencies
sudo apt install -y python3.11 python3.11-venv python3-pip git

# Create a dedicated user (optional but recommended)
sudo useradd -m -s /bin/bash portfolio
sudo su - portfolio
```

### Step 2: Clone and Install

```bash
git clone https://github.com/alikaangurbuz/finance-agent-claude.git
cd finance-agent-claude

python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Step 3: Configure Environment

```bash
cp .env.example .env
nano .env
```

Set at minimum the three required variables:

```bash
PA_OPENAI_API_KEY=sk-...
PA_TELEGRAM_BOT_TOKEN=123456:ABC-DEF...
PA_TELEGRAM_CHAT_ID=123456789
```

### Step 4: Initialize Database

The database is created automatically on first run. To verify:

```bash
portfolio-advisor &
# Wait a few seconds for startup, then Ctrl+C
# The database file will be at data/portfolio_advisor.db
```

### Step 5: Create systemd Service

Create the service file:

```bash
sudo nano /etc/systemd/system/portfolio-advisor.service
```

Paste the following content (adjust paths and user as needed):

```ini
[Unit]
Description=Portfolio Advisor - Multi-Agent Financial Advisory System
After=network.target
Wants=network-online.target

[Service]
Type=simple
User=portfolio
Group=portfolio
WorkingDirectory=/home/portfolio/finance-agent-claude
ExecStart=/home/portfolio/finance-agent-claude/.venv/bin/portfolio-advisor
Restart=always
RestartSec=30
StandardOutput=journal
StandardError=journal

# Environment
EnvironmentFile=/home/portfolio/finance-agent-claude/.env

# Resource limits
MemoryMax=1G
CPUQuota=80%

# Security hardening
NoNewPrivileges=yes
ProtectSystem=strict
ProtectHome=read-only
ReadWritePaths=/home/portfolio/finance-agent-claude/data
PrivateTmp=yes

[Install]
WantedBy=multi-user.target
```

### Step 6: Start and Enable

```bash
sudo systemctl daemon-reload
sudo systemctl enable portfolio-advisor
sudo systemctl start portfolio-advisor
```

### Step 7: Verify

```bash
# Check service status
sudo systemctl status portfolio-advisor

# View logs
sudo journalctl -u portfolio-advisor -f

# Health check (once the service is running)
curl http://127.0.0.1:8080/health
curl http://127.0.0.1:8080/status
```

The `/health` endpoint returns a simple liveness probe. The `/status` endpoint returns detailed system status including database connectivity, last pre-compute time, provider status, and scheduler state.

### Maintenance

**Updating:**

```bash
cd /home/portfolio/finance-agent-claude
git pull
source .venv/bin/activate
pip install -e .
sudo systemctl restart portfolio-advisor
```

**Database backup:**

```bash
# SQLite safe backup (while running, WAL mode handles this)
cp data/portfolio_advisor.db data/portfolio_advisor.db.bak
```

**Log rotation:**

systemd journal handles log rotation automatically. To view recent logs:

```bash
sudo journalctl -u portfolio-advisor --since "1 hour ago"
sudo journalctl -u portfolio-advisor --since today
```

**API key rotation:**

```bash
# Edit .env with new keys
nano /home/portfolio/finance-agent-claude/.env
# Restart to pick up changes
sudo systemctl restart portfolio-advisor
```

---

## Telegram Commands

| Command | Description |
|---------|-------------|
| `/start` | Welcome message and initialize defaults |
| `/help` | List available commands |
| `/status` | System status, last runs, data freshness |
| `/portfolio` | Current portfolio allocations |
| `/prefs` | Show user preferences |
| `/set <key> <value>` | Update a preference |
| `/watchlist` | Current watchlist |
| `/addticker TSLA` | Add ticker(s) to watchlist |
| `/removeticker IWM` | Remove ticker(s) from watchlist |
| `/confirm` | Confirm onboarding step |
| `/brief` | Latest daily brief |
| `/report` | Latest weekly report |
| `/earnings` | Upcoming earnings for watchlist |
| `/news` | Latest news themes and alerts |
| `/usage` | Token usage and estimated cost |
| `/rundaily` | Force a daily analysis run |
| `/runweekly` | Force a weekly report run |

Any free-text message is routed to the Chat Agent for interactive analysis. The chat agent has access to 39 tools and 4 specialist agent delegates, enabling on-demand technical analysis, quantitative modeling, portfolio optimization, and research.

---

## Testing

The test suite includes both unit tests (mocked dependencies) and live integration tests (real API calls).

### Unit Tests

```bash
source .venv/bin/activate

# Run all unit tests
.venv/bin/pytest

# Run a specific test directory
.venv/bin/pytest tests/test_tools/
.venv/bin/pytest tests/test_db/
.venv/bin/pytest tests/test_providers/
.venv/bin/pytest tests/test_utils/

# Run a specific test file
.venv/bin/pytest tests/test_tools/test_technical_indicators.py

# Run a specific test by name
.venv/bin/pytest -k "test_fetch_ohlcv"
```

### Live Integration Tests

Live tests require real API keys and network access. They are marked with `@pytest.mark.live`:

```bash
# Run live tests (requires PA_* env vars to be set)
.venv/bin/pytest -m live

# Specific live test suites
.venv/bin/pytest tests/test_tools/test_technical_live.py -m live
.venv/bin/pytest tests/test_tools/test_quant_live.py -m live
.venv/bin/pytest tests/test_tools/test_portfolio_live.py -m live
.venv/bin/pytest tests/test_tools/test_risk_live.py -m live
.venv/bin/pytest tests/test_providers/test_live_providers.py -m live
```

### End-to-End Pipeline Test

```bash
.venv/bin/pytest tests/test_tools/test_pipeline_e2e.py -m live
```

### Quick Import Smoke Test

Verify all modules load correctly without running the full application:

```bash
PA_OPENAI_API_KEY=test PA_TELEGRAM_BOT_TOKEN=test PA_TELEGRAM_CHAT_ID=123 \
  .venv/bin/python -c "from portfolio_advisor.agents.chat import get_chat_agent; a = get_chat_agent(); print(f'{len(a.tools)} tools')"
```

### Linting

```bash
.venv/bin/ruff check .
.venv/bin/ruff format --check .
```

---

## Architecture

### Agent Hierarchy

```
Pre-compute Pipeline (scheduler, no LLM)
└── run_precompute_pipeline()
    ├── Batch OHLCV fetch for entire watchlist
    ├── 12 technical indicators per ticker
    ├── Quant metrics (GARCH, HMM, Kalman, FF3)
    ├── Macro snapshot (VIX, yield curve, credit spread, regime)
    ├── Earnings calendar
    ├── Correlation matrix
    └── Per-ticker analysis narratives → SQLite

Daily Pipeline v3 (token-optimized)
├── _build_daily_context() — loads pre-computed data from DB (0 tokens)
├── Research Agent — fresh news via web search (~20K tokens)
└── Daily Synthesis Agent — combines context + research → DailyBrief

Weekly Orchestrator
├── Portfolio Agent.as_tool() — 24 tools (CVaR/HRP/Kelly + advanced risk)
└── Reporting Agent.as_tool() — investment committee memo

Chat Agent — 39 tools (35 direct + 4 agent delegates)
├── 10 pre-computed cache tools (instant, 0 LLM cost)
├── Market data, macro, earnings, portfolio, preferences, DB, web, usage tools
├── Technical Agent.as_tool() — 14 hidden tools
├── Quantitative Agent.as_tool() — 23 hidden tools
├── Portfolio Agent.as_tool() — 24 hidden tools
└── Research Agent.as_tool() — web search

Onboarding Agent — 6 tools, guided 5-step new-user setup

News Alert Pipeline (midday + evening)
└── Research Agent → theme comparison → Telegram alert
```

### Agents-as-Tools Pattern

Orchestrators call specialist agents via `.as_tool()` (fan-out/fan-in). The orchestrator retains control of the conversation, dispatches to specialists for focused analysis, and synthesizes results. All agents use lazy initialization via `get_*_agent()` factory functions with config-based model selection.

### Scheduler Jobs

9 automated jobs run via APScheduler 4.x (`AsyncScheduler`):

| Job | Time (UTC) | Description |
|-----|------------|-------------|
| `precompute_morning` | 06:00 | Batch technical + quant + macro + earnings + correlations |
| `daily_monitoring` | 06:30 | Daily brief generation (uses pre-computed data + fresh research) |
| `news_check_0` | 09:00 | News alert check |
| `precompute_midday` | 13:00 | Midday pre-compute refresh |
| `midday_update` | 13:30 | Midday market update |
| `news_check_1` | 15:00 | News alert check |
| `evening_summary` | 20:00 | Evening summary with news alerts |
| `weekly_report` | Sun 18:00 | Full portfolio review and investment memo |
| `forecast_eval` | 22:00 | Evaluate prior forecast accuracy |

Total news coverage: 5x daily (06:30, 09:00, 13:30, 15:00, 20:00) with max ~3.5h gaps.

### Pre-Compute Pipeline

The pre-compute pipeline (`tools/precomputed.py`) runs 2-3x daily and performs all compute-heavy analysis without LLM calls:

1. Batch-fetches OHLCV data for the entire watchlist
2. Computes 12 technical indicators per ticker (RSI, MACD, Bollinger, Ichimoku, VWAP, OBV, ADX, Stochastic, Fibonacci, etc.)
3. Calculates quant metrics (GARCH volatility, HMM regime state, Kalman beta, Fama-French 3-factor betas)
4. Captures macro snapshot (VIX, yield curve, credit spread, macro regime)
5. Updates earnings calendar
6. Builds correlation matrix
7. Generates per-ticker analysis narratives
8. Stores all results in SQLite with `snapshot_hour` for intraday data preservation

The chat agent checks cache freshness first and uses cached data (10 cache tools at zero LLM cost) before delegating to specialist agents for live computation.

### Tool Pattern

Tools follow a two-layer architecture: pure `_raw()` computation functions for pipeline reuse, and thin `@function_tool` wrappers for agent access:

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

### Database

SQLite via aiosqlite with WAL mode and 30-second busy timeout. 22 tables across 4 schema versions:

- **v1**: user_preferences, portfolio_state, portfolio_history, daily_briefs, instrument_briefs, weekly_reports, price_cache, forecasts_log, token_usage
- **v2**: technical_indicators, quant_metrics, daily_risk_metrics, research_themes, forecast_accuracy, onboarding_state, chat_history, analysis_runs
- **v3**: earnings_calendar, correlation_snapshot
- **v4**: fundamentals, sentiment_metrics

### Project Structure

```
src/portfolio_advisor/
├── main.py                          # Entry point (scheduler + Telegram bot)
├── config.py                        # PA_* settings via pydantic-settings
├── health.py                        # HTTP health/status server (port 8080)
├── agents/
│   ├── chat.py                      # Chat agent (39 tools + 4 delegates)
│   ├── orchestrator.py              # Daily + Weekly orchestrators
│   ├── portfolio.py                 # Portfolio agent (24 tools)
│   ├── quantitative.py              # Quant agent (23 tools)
│   ├── technical.py                 # Technical agent (14 tools)
│   ├── research.py                  # Research agent (web search)
│   ├── reporting.py                 # Reporting agent
│   ├── onboarding.py                # 5-step onboarding agent
│   └── context.py                   # AppContext dataclass
├── tools/
│   ├── precomputed.py               # Pre-compute pipeline + 10 cache tools
│   ├── market_data.py               # OHLCV data fetching
│   ├── technical_indicators.py      # Core technical indicators
│   ├── advanced_technical.py        # Ichimoku, VWAP, OBV, ADX, etc.
│   ├── quant_models.py              # Core quant models
│   ├── advanced_quant.py            # GARCH, HMM, Kalman, FF3
│   ├── portfolio_optimization.py    # Core portfolio optimization
│   ├── advanced_portfolio.py        # CVaR, HRP, Kelly, Max Div
│   ├── advanced_risk.py             # VaR variants, stress testing
│   ├── advanced_time_series.py      # Granger, change points, spectral
│   ├── advanced_analytics.py        # PCA, clustering, style analysis
│   ├── economic_data.py             # FRED, yield curve, macro regime
│   ├── earnings.py                  # Earnings calendar
│   ├── fundamentals.py              # PE, PB, ROE, margins
│   ├── news_data.py                 # News with sentiment
│   ├── sentiment.py                 # Short interest, squeeze risk
│   └── corporate_actions.py         # Dividends, stock splits
├── providers/
│   ├── registry.py                  # Fallback chains + round-robin rotation
│   ├── massive_provider.py          # Massive (Polygon.io) API
│   ├── fred_provider.py             # FRED API
│   ├── alpha_vantage_provider.py    # Alpha Vantage API
│   ├── yfinance_provider.py         # yfinance (no key required)
│   └── coingecko_provider.py        # CoinGecko (no key required)
├── db/
│   ├── schema.py                    # DDL (22 tables, 4 versions)
│   ├── connection.py                # init_db, get_db, migrations
│   └── queries.py                   # Typed CRUD operations
├── telegram_bot/
│   ├── bot.py                       # 17 command handlers + chat routing
│   ├── commands.py                  # Command implementations
│   └── chat_handler.py              # Onboarding/chat agent routing
├── scheduler/
│   ├── runner.py                    # 9 job definitions
│   ├── jobs.py                      # Job implementations
│   └── alerts.py                    # News alert pipeline
└── utils/
    ├── logging.py                   # Structured logging setup
    ├── circuit_breaker.py           # CLOSED -> OPEN -> HALF_OPEN
    └── retry.py                     # Exponential backoff with jitter
```

---

## Troubleshooting

### API Rate Limits

**Symptom**: `429 Too Many Requests` errors in logs.

**Solutions**:
- **Massive**: Add more API keys for rotation. Set `PA_MASSIVE_API_KEYS=key1,key2,key3,key4` to increase from 5 to 20 calls/min.
- **Alpha Vantage**: Add more keys. Each key provides 25 calls/day. Set `PA_ALPHA_VANTAGE_API_KEYS=key1,key2`.
- **OpenAI**: The system tracks token budgets (`PA_DAILY_TOKEN_BUDGET`, `PA_WEEKLY_TOKEN_BUDGET`) to avoid overspending. Increase the budget or upgrade your OpenAI tier if hitting limits.
- The provider registry includes circuit breakers and exponential backoff with jitter for automatic recovery.

### SQLite Lock Errors

**Symptom**: `database is locked` errors during concurrent access.

**Solutions**:
- The database is configured with WAL mode and 30-second busy timeout, which should handle normal concurrency.
- If errors persist, check for runaway processes: `lsof data/portfolio_advisor.db`
- Ensure only one instance of `portfolio-advisor` is running: `systemctl status portfolio-advisor`

### yfinance 404 Errors

**Symptom**: `404 Client Error` for certain tickers.

**Solutions**:
- Some tickers may be delisted or renamed. Remove them with `/removeticker <TICKER>`.
- Crypto tickers use CoinGecko, not yfinance. Ensure crypto symbols are in the correct format (BTC, ETH, not BTC-USD).
- yfinance may have temporary outages. The system will retry automatically with exponential backoff.

### Settings Validation Errors

**Symptom**: `ValidationError` on startup with missing or invalid settings.

**Solutions**:
- Ensure all three required variables are set: `PA_OPENAI_API_KEY`, `PA_TELEGRAM_BOT_TOKEN`, `PA_TELEGRAM_CHAT_ID`.
- `PA_TELEGRAM_CHAT_ID` must be a valid integer, not a string.
- Check for trailing whitespace or quotes in `.env` values.
- Run `env | grep PA_` to verify environment variables are loaded correctly.

### Health Check Not Responding

**Symptom**: `curl http://127.0.0.1:8080/health` times out.

**Solutions**:
- Verify the service is running: `systemctl status portfolio-advisor`
- Check if the port is in use: `ss -tlnp | grep 8080`
- Change the port with `PA_HEALTH_PORT=8081` if there is a conflict.
- The health server binds to `127.0.0.1` only (not accessible externally by default).

### Pre-Compute Pipeline Not Running

**Symptom**: Cache data is stale, chat responses are slow (delegating to live agents).

**Solutions**:
- Check the last pre-compute run: `curl http://127.0.0.1:8080/status | python3 -m json.tool`
- Verify `PA_PRECOMPUTE_ENABLED=true` (default).
- Check scheduler logs: `journalctl -u portfolio-advisor --since "6 hours ago" | grep precompute`
- Adjust freshness threshold with `PA_PRECOMPUTE_STALE_HOURS` (default: 8 hours).

---

## License

Private -- all rights reserved.
