"""User preference models."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class RiskTolerance(str, Enum):
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


class InvestmentStyle(str, Enum):
    PASSIVE = "passive"
    ACTIVE = "active"
    TACTICAL = "tactical"


class RebalanceFrequency(str, Enum):
    WEEKLY = "weekly"
    BIWEEKLY = "biweekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"


class DividendPreference(str, Enum):
    GROWTH = "growth"
    INCOME = "income"
    NEUTRAL = "neutral"


class NotificationLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class AnalysisDepth(str, Enum):
    BRIEF = "brief"
    DETAILED = "detailed"
    EXHAUSTIVE = "exhaustive"


class UserPreferences(BaseModel):
    # Original fields
    risk_tolerance: RiskTolerance = RiskTolerance.MODERATE
    time_horizon: str = "medium"  # short / medium / long
    excluded_assets: list[str] = []
    allowed_regions: list[str] = ["US", "EU", "APAC", "EM"]
    cash_target_pct: float = 10.0
    max_position_pct: float = 15.0
    watchlist: list[str] = []

    # v2 expanded fields
    investment_style: InvestmentStyle = InvestmentStyle.PASSIVE
    rebalance_frequency: RebalanceFrequency = RebalanceFrequency.MONTHLY
    max_crypto_pct: float = Field(default=15.0, ge=0.0, le=100.0)
    min_bond_pct: float = Field(default=5.0, ge=0.0, le=100.0)
    max_single_sector_pct: float = Field(default=40.0, ge=0.0, le=100.0)
    preferred_sectors: list[str] = []
    esg_filter: bool = False
    dividend_preference: DividendPreference = DividendPreference.NEUTRAL
    tax_aware: bool = False
    notification_level: NotificationLevel = NotificationLevel.MEDIUM
    analysis_depth: AnalysisDepth = AnalysisDepth.DETAILED
    benchmark: str = "SPY"
    notes: str = ""
