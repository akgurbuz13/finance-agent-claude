"""Portfolio state models."""

from __future__ import annotations

from pydantic import BaseModel


class Position(BaseModel):
    ticker: str
    weight_pct: float
    asset_class: str


class PortfolioState(BaseModel):
    positions: list[Position]
    cash_pct: float
    updated_at: str  # ISO datetime


class RebalanceDelta(BaseModel):
    ticker: str
    current_weight_pct: float
    target_weight_pct: float
    delta_pct: float
    action: str  # buy / sell / hold
