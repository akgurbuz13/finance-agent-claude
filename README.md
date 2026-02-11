# Portfolio Advisor Agent

24/7 autonomous portfolio advisory system that monitors financial instruments and produces daily market briefs + weekly portfolio recommendation reports via Telegram.

**Does NOT place trades** — generates allocations and decision support only.

## Architecture

Multi-agent system using the [OpenAI Agents SDK](https://github.com/openai/openai-agents-python) with agents-as-tools orchestration pattern:

| Agent | Model | Role |
|-------|-------|------|
| Daily Orchestrator | gpt-5-mini | Coordinates daily pipeline |
| Weekly Orchestrator | gpt-5.2 | Coordinates weekly investment committee |
| Technical Analysis | gpt-5-mini | SMA/EMA, RSI, MACD, Bollinger, S/R |
| Quantitative | gpt-5-mini | Return/vol forecasts, regime, correlations |
| Research | gpt-5-mini | Macro/news via web search |
| Portfolio Construction | gpt-5-mini | Risk-parity, mean-variance, Black-Litterman |
| Reporting | gpt-5.2 | Weekly investment committee memo |
| Chat | gpt-5-mini | Full-powered interactive analyst (45 tools) |

### Data Flow

```
Daily (automated, 7am UTC):
  Orchestrator → Technical + Quant + Research agents → Synthesized brief → SQLite + Telegram

Weekly (automated, Sunday 6pm UTC):
  Orchestrator → Portfolio Construction + Reporting agents → Investment memo → SQLite + Telegram

Chat (on-demand via Telegram):
  User message → Chat Agent (all 45 tools) → Live analysis + actions
```

## Tools (45 total)

- **Market Data**: yfinance (equities/ETFs) + CoinGecko (crypto)
- **Technical Indicators**: SMA/EMA crossovers, RSI, MACD, ATR/Bollinger, support/resistance, weekly signals
- **Quantitative Models**: Return/vol forecasts, regime detection, correlations, factor exposures
- **Time Series**: Autocorrelation, stationarity (ADF), seasonal decomposition, cointegration, rolling statistics
- **Data Analysis**: Distribution analysis, drawdown analysis, performance metrics (Sharpe/Sortino/Calmar), outlier detection, cross-asset analysis
- **Risk Metrics**: Historical VaR, Expected Shortfall, max drawdown, beta exposure
- **Portfolio Optimization**: Risk-parity, mean-variance, max-Sharpe, efficient frontier, Black-Litterman, concentration limits, risk controls
- **Storage**: Daily briefs, weekly reports, forecast logging, portfolio state, user preferences

## Tech Stack

| Component | Choice |
|-----------|--------|
| Language | Python 3.11+ |
| Agent Framework | openai-agents-python |
| Market Data | yfinance + CoinGecko |
| Communication | python-telegram-bot |
| Storage | SQLite (aiosqlite) |
| Scheduling | APScheduler 4.x |
| Deployment | systemd on VPS |

## Setup

### Prerequisites

- Python 3.11+
- OpenAI API key
- Telegram bot token (from [@BotFather](https://t.me/BotFather))

### Installation

```bash
git clone https://github.com/alikaangurbuz/finance-agent-claude.git
cd finance-agent-claude

python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### Configuration

```bash
cp .env.example .env
# Edit .env with your credentials:
#   PA_OPENAI_API_KEY=sk-...
#   PA_TELEGRAM_BOT_TOKEN=123456:ABC-DEF...
#   PA_TELEGRAM_CHAT_ID=123456789
```

### Run

```bash
portfolio-advisor
```

This starts both the Telegram bot (for interactive chat) and the APScheduler (for automated daily/weekly runs).

## Telegram Commands

| Command | Description |
|---------|-------------|
| `/start` | Welcome + initialize defaults |
| `/status` | System status, last runs, token usage |
| `/portfolio` | Current allocations |
| `/prefs` | Show preferences |
| `/set <key> <value>` | Update a preference |
| `/watchlist` | Current watchlist |
| `/addticker TSLA` | Add tickers |
| `/removeticker IWM` | Remove tickers |
| `/brief` | Latest daily brief |
| `/report` | Latest weekly report |
| `/usage` | Token usage + cost |
| `/rundaily` | Force daily run |
| `/runweekly` | Force weekly run |

Any free-text message routes to the Chat Agent for interactive analysis.

## Default Watchlist

**Equities/Sectors:** SPY, QQQ, IWM, EFA, EEM, VNQ, XLE, XLK, XLF
**Bonds:** TLT, IEF, HYG
**Commodities:** GLD, SLV
**Large-cap:** AAPL, MSFT, NVDA, AMZN
**Crypto:** BTC, ETH, SOL, AVAX

## VPS Deployment

```bash
# On Ubuntu VPS:
sudo bash deploy/setup_vps.sh
# Then configure .env and enable the systemd service
```

See `deploy/portfolio-advisor.service` for the systemd unit file.

## Project Structure

```
src/portfolio_advisor/
├── main.py                     # Entry point
├── config.py                   # Settings (.env loading)
├── agents/                     # 8 agent definitions with detailed prompts
├── tools/                      # 11 tool modules (45 functions)
├── models/                     # Pydantic data models
├── db/                         # SQLite schema + queries
├── telegram_bot/               # Bot, commands, chat handler
├── scheduler/                  # APScheduler jobs
└── utils/                      # Logging, rate limiting
```

See [docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md) for the full architecture document.

## License

Private — all rights reserved.
