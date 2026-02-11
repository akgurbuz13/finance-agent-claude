"""Weekly report models."""

from __future__ import annotations

from pydantic import BaseModel


class AllocationRecommendation(BaseModel):
    ticker: str
    asset_class: str
    current_weight_pct: float
    recommended_weight_pct: float
    delta_pct: float
    rationale: str


class RiskAssessment(BaseModel):
    overall_risk_level: str  # low / moderate / elevated / high
    key_risks: list[str]
    hedging_suggestions: list[str] = []
    portfolio_var_95: float | None = None
    portfolio_es_95: float | None = None
    max_drawdown_current: float | None = None


class WeeklyReport(BaseModel):
    week_ending: str  # YYYY-MM-DD
    executive_summary: str
    market_review: str
    allocations: list[AllocationRecommendation]
    risk_assessment: RiskAssessment
    outlook: str
    action_items: list[str]
    telegram_summary: str  # Pre-formatted for Telegram
