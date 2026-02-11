"""Daily brief models."""

from __future__ import annotations

from pydantic import BaseModel


class InstrumentBrief(BaseModel):
    ticker: str
    signal: str  # bullish / bearish / neutral
    confidence: float
    what_happened: str
    why_it_matters: str
    technical_json: dict
    quant_json: dict
    sources: list[str] = []


class ThemeBrief(BaseModel):
    theme: str
    summary: str
    affected_tickers: list[str]
    sources: list[str] = []


class PortfolioRiskSnapshot(BaseModel):
    portfolio_var_95: float | None = None
    portfolio_es_95: float | None = None
    max_drawdown: float | None = None
    portfolio_beta: float | None = None
    concentration_warnings: list[str] = []


class DailyBrief(BaseModel):
    brief_date: str  # YYYY-MM-DD
    market_summary: str
    instruments: list[InstrumentBrief]
    themes: list[ThemeBrief] = []
    risk_snapshot: PortfolioRiskSnapshot | None = None
    telegram_summary: str  # Pre-formatted for Telegram
