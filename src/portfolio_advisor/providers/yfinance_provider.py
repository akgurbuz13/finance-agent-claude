"""yfinance wrapper provider — no rate limiting needed (free, single HTTP call)."""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


class YFinanceProvider:
    """Wraps existing yfinance calls with provider interface.

    yfinance is the primary OHLCV provider — bulk download via yf.download()
    is efficient (single HTTP call, no API key needed).
    This wrapper exists mainly for fallback treasury yield data.
    """

    async def fetch_treasury_yields(self) -> dict | None:
        """Fetch treasury yields via yfinance ticker symbols."""
        import yfinance as yf

        yield_tickers = {"3m": "^IRX", "5y": "^FVX", "10y": "^TNX", "30y": "^TYX"}
        yields = {}
        try:
            tickers_str = " ".join(yield_tickers.values())
            data = yf.download(tickers_str, period="5d", progress=False)
            if data.empty:
                return None
            for label, ticker in yield_tickers.items():
                try:
                    if len(yield_tickers) == 1:
                        val = float(data["Close"].dropna().iloc[-1])
                    else:
                        val = float(data["Close"][ticker].dropna().iloc[-1])
                    yields[label] = round(val, 3)
                except (KeyError, IndexError):
                    pass
        except Exception as e:
            logger.warning(f"yfinance treasury yield fetch failed: {e}")
            return None
        return yields if yields else None

    async def fetch_vix(self) -> float | None:
        """Fetch VIX level via yfinance ^VIX ticker."""
        import yfinance as yf

        try:
            data = yf.download("^VIX", period="5d", progress=False)
            if not data.empty:
                close = data["Close"]
                # yfinance may return MultiIndex columns even for a single ticker
                if isinstance(close, pd.DataFrame):
                    close = close.iloc[:, 0]
                return round(float(close.dropna().iloc[-1]), 2)
        except Exception as e:
            logger.warning(f"yfinance VIX fetch failed: {e}")
        return None

    def status(self) -> dict:
        return {"provider": "yfinance", "has_key": True, "status": "always_available"}
