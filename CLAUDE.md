# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install (editable, with dev deps)
pip install -e ".[dev]"

# Run the application (starts Telegram bot + APScheduler)
portfolio-advisor

# Lint
ruff check .
ruff check --fix .
ruff format .

# Test
pytest                          # all tests
pytest tests/test_tools/        # one directory
pytest tests/test_tools/test_market_data.py  # one file
pytest -k "test_fetch_ohlcv"    # one test by name

# Verify imports (quick smoke test)
python -c "from portfolio_advisor.agents.chat import chat_agent; print(f'{len(chat_agent.tools)} tools')"
```

Ruff config: `target-version = "py311"`, `line-length = 100`. Pytest uses `asyncio_mode = "auto"`.

## Architecture

Multi-agent portfolio advisory system using the **OpenAI Agents SDK** (`openai-agents-python`). Runs automated daily/weekly analysis pipelines and an interactive Telegram chat interface. Does NOT place trades.

### Agents-as-Tools Pattern

Orchestrators call specialist agents via `.as_tool()` (fan-out/fan-in). The orchestrator retains control and synthesizes results.

```
Daily Orchestrator (gpt-5-mini)
├── Technical Agent.as_tool("run_technical_analysis")
├── Quantitative Agent.as_tool("run_quantitative_analysis")
└── Research Agent.as_tool("run_market_research")

Weekly Orchestrator (gpt-5.2)
├── Portfolio Agent.as_tool("run_portfolio_construction")
└── Reporting Agent.as_tool("run_reporting")

Chat Agent (gpt-5-mini) — 45 tools directly, no sub-agent delegation
```

### Shared Context

All agents typed as `Agent[AppContext]`. Every `@function_tool` receives `ctx: RunContextWrapper[AppContext]` as first parameter and accesses the database via `ctx.context.db_path`. The `AppContext` dataclass (`agents/context.py`) carries db_path, telegram_chat_id, run_date, watchlist, and token budget state.

### Tool Pattern

Tools in `tools/` use `@function_tool` from `openai-agents`. They are async, take `ctx: RunContextWrapper[AppContext]` as first arg, and return JSON strings. Docstrings become tool descriptions for the LLM.

```python
@function_tool
async def example_tool(ctx: RunContextWrapper[AppContext], ticker: str) -> str:
    async with get_db(ctx.context.db_path) as db:
        data = await queries.some_query(db, ticker)
    return json.dumps(data)
```

### Database Layer

SQLite via aiosqlite. Three files: `db/schema.py` (DDL), `db/connection.py` (`init_db` + `get_db` context manager), `db/queries.py` (typed CRUD). 9 tables: user_preferences, portfolio_state, portfolio_history, daily_briefs, instrument_briefs, weekly_reports, price_cache, forecasts_log, token_usage.

### Runtime Flow

`main.py` → `init_db()` → `setup_scheduler()` + `start_scheduler()` → `build_application().run_polling()`. APScheduler 4.x (`AsyncScheduler`) fires daily/weekly jobs that call `Runner.run()` on orchestrator agents. Telegram free-text routes to the chat agent via `chat_handler.py`.

### Key Files

- `agents/orchestrator.py` — Daily + Weekly orchestrator definitions (the pipeline coordinators)
- `agents/chat.py` — Chat agent with all 45 tools (the primary user interface)
- `scheduler/jobs.py` — `daily_job()` and `weekly_job()` build AppContext and call `Runner.run()`
- `telegram_bot/bot.py` — Registers 13 command handlers + free-text chat handler
- `config.py` — pydantic-settings with `PA_` env prefix

### Agent Prompt Convention

Each agent file has a `*_INSTRUCTIONS` string constant with structured sections: Role, Procedure (numbered steps), Output Format (JSON schema), Edge Cases, and Constraints. Agent prompts instruct the LLM to reason over tool outputs, not just relay them.

## Configuration

Copy `.env.example` to `.env`. Required: `PA_OPENAI_API_KEY`, `PA_TELEGRAM_BOT_TOKEN`, `PA_TELEGRAM_CHAT_ID`. All settings use the `PA_` prefix.
