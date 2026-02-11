"""Application configuration via pydantic-settings."""

from __future__ import annotations

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_prefix": "PA_", "env_file": ".env", "env_file_encoding": "utf-8"}

    # API keys
    openai_api_key: str
    telegram_bot_token: str
    telegram_chat_id: int

    # Database
    db_path: str = "data/portfolio_advisor.db"

    # Schedule (UTC)
    daily_run_hour: int = 7
    weekly_run_day: str = "sun"
    weekly_run_hour: int = 18

    # Multi-schedule (UTC hours)
    morning_run_hour: int = 6
    midday_run_hour: int = 13
    evening_run_hour: int = 20

    # Token budgets
    daily_token_budget: int = 100_000
    weekly_token_budget: int = 200_000
    max_web_searches_daily: int = 20

    # Models — legacy (kept for backward compat)
    weekly_model: str = "gpt-5.2"
    daily_model: str = "gpt-5-mini"

    # Per-agent model assignments
    model_orchestrator: str = "gpt-5.2"
    model_technical: str = "gpt-5.2"
    model_quantitative: str = "gpt-5.2"
    model_portfolio: str = "gpt-5.2"
    model_reporting: str = "gpt-5.2"
    model_chat: str = "gpt-5.2"
    model_research: str = "gpt-5-mini"
    model_onboarding: str = "gpt-5-mini"

    # Pre-computation
    precompute_enabled: bool = True
    precompute_stale_hours: float = 8.0

    # FRED API
    fred_api_key: str = ""

    # Onboarding
    onboarding_enabled: bool = True

    # Default watchlist
    default_watchlist: list[str] = [
        "SPY", "QQQ", "IWM", "EFA", "EEM", "VNQ", "XLE", "XLK", "XLF",
        "TLT", "IEF", "HYG",
        "GLD", "SLV",
        "AAPL", "MSFT", "NVDA", "AMZN",
        "BTC", "ETH", "SOL", "AVAX",
    ]


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
