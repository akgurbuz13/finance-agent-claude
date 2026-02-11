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

    # Token budgets
    daily_token_budget: int = 100_000
    weekly_token_budget: int = 200_000
    max_web_searches_daily: int = 20

    # Models
    weekly_model: str = "gpt-5.2"
    daily_model: str = "gpt-5-mini"

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
