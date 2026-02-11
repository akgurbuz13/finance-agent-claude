"""Market data and analysis models."""

from __future__ import annotations

from pydantic import BaseModel


class OHLCVBar(BaseModel):
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: float


class TechnicalSignal(BaseModel):
    indicator: str
    value: float | dict
    interpretation: str  # bullish / bearish / neutral
    confidence: float  # 0-1


class TechnicalSignals(BaseModel):
    ticker: str
    signals: list[TechnicalSignal]
    overall_bias: str  # bullish / bearish / neutral
    overall_confidence: float


class QuantMetrics(BaseModel):
    ticker: str
    return_forecast: dict  # horizons → {expected, ci_low, ci_high}
    vol_forecast: dict  # {ewma_vol, regime, percentile}
    regime: str  # trending / mean-reverting / volatile
    beta: float | None = None
    hurst_exponent: float | None = None


class InstrumentAnalysis(BaseModel):
    ticker: str
    technical: TechnicalSignals
    quantitative: QuantMetrics
    news_summary: str | None = None
    sources: list[str] = []
