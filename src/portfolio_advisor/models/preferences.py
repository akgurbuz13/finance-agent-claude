"""User preference models."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel


class RiskTolerance(str, Enum):
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


class UserPreferences(BaseModel):
    risk_tolerance: RiskTolerance = RiskTolerance.MODERATE
    time_horizon: str = "medium"  # short / medium / long
    excluded_assets: list[str] = []
    allowed_regions: list[str] = ["US", "EU", "APAC", "EM"]
    cash_target_pct: float = 10.0
    max_position_pct: float = 15.0
    watchlist: list[str] = []
