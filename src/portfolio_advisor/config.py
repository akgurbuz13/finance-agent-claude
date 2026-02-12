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
    daily_token_budget: int = 200_000
    weekly_token_budget: int = 1200000
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
    model_daily_synthesis: str = "gpt-5.2"

    # Additional news check times (UTC hours, between main runs)
    news_check_hours: list[int] = [9, 15]

    # Pre-computation
    precompute_enabled: bool = True
    precompute_stale_hours: float = 8.0

    # FRED API
    fred_api_key: str = ""

    # Massive API — supports multiple comma-separated keys for rotation
    # New field (preferred): PA_MASSIVE_API_KEYS=key1,key2,key3,key4
    # Legacy single-key field still works: PA_MASSIVE_API_KEY=key1
    massive_api_keys: str = ""
    massive_api_key: str = ""  # legacy single-key (fallback)

    # Alpha Vantage API — supports multiple comma-separated keys for rotation
    # New field (preferred): PA_ALPHA_VANTAGE_API_KEYS=key1,key2
    # Legacy single-key field still works: PA_ALPHA_VANTAGE_API_KEY=key1
    alpha_vantage_api_keys: str = ""
    alpha_vantage_api_key: str = ""  # legacy single-key (fallback)

    # Health check
    health_port: int = 8080

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

    def get_massive_keys(self) -> list[str]:
        """Get list of Massive API keys (from comma-separated string or single key)."""
        if self.massive_api_keys:
            return [k.strip() for k in self.massive_api_keys.split(",") if k.strip()]
        if self.massive_api_key:
            return [self.massive_api_key]
        return []

    def get_alpha_vantage_keys(self) -> list[str]:
        """Get list of Alpha Vantage API keys (from comma-separated string or single key)."""
        if self.alpha_vantage_api_keys:
            return [k.strip() for k in self.alpha_vantage_api_keys.split(",") if k.strip()]
        if self.alpha_vantage_api_key:
            return [self.alpha_vantage_api_key]
        return []


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
