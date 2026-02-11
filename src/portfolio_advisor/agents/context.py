"""Shared context for all agents via RunContextWrapper[AppContext]."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date


@dataclass
class AppContext:
    db_path: str
    telegram_chat_id: int
    run_date: date = field(default_factory=date.today)
    watchlist: list[str] = field(default_factory=list)
    token_budget_remaining: int = 100_000
    web_search_calls_today: int = 0
    max_web_search_calls_daily: int = 20
