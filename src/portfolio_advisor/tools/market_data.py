"""Market data fetching — yfinance for equities/ETFs, CoinGecko for crypto."""

from __future__ import annotations

import json
from datetime import date, timedelta

import httpx
import pandas as pd
import yfinance as yf
from agents import RunContextWrapper, function_tool

from portfolio_advisor.agents.context import AppContext
from portfolio_advisor.db.connection import get_db
from portfolio_advisor.db.queries import cache_prices, get_cached_prices

CRYPTO_MAP = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "SOL": "solana",
    "AVAX": "avalanche-2",
}

COINGECKO_BASE = "https://api.coingecko.com/api/v3"


def _is_crypto(ticker: str) -> bool:
    return ticker.upper() in CRYPTO_MAP


def _df_to_bars(df: pd.DataFrame) -> list[dict]:
    bars = []
    for idx, row in df.iterrows():
        dt_str = idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else str(idx)[:10]
        bars.append({
            "date": dt_str,
            "open": round(float(row["Open"]), 4),
            "high": round(float(row["High"]), 4),
            "low": round(float(row["Low"]), 4),
            "close": round(float(row["Close"]), 4),
            "volume": float(row.get("Volume", 0)),
        })
    return bars


async def _fetch_crypto_ohlcv(coin_id: str, days: int) -> list[dict]:
    url = f"{COINGECKO_BASE}/coins/{coin_id}/ohlc"
    params = {"vs_currency": "usd", "days": days}
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()

    bars = []
    for candle in data:
        ts, o, h, l, c = candle
        dt_str = pd.Timestamp(ts, unit="ms").strftime("%Y-%m-%d")
        bars.append({
            "date": dt_str,
            "open": round(o, 4),
            "high": round(h, 4),
            "low": round(l, 4),
            "close": round(c, 4),
            "volume": 0.0,
        })
    return bars


@function_tool
async def fetch_ohlcv(
    ctx: RunContextWrapper[AppContext],
    tickers: str,
    period: str = "6mo",
    interval: str = "1d",
) -> str:
    """Fetch OHLCV price data for given tickers (comma-separated). Uses yfinance for equities/ETFs and CoinGecko for crypto (BTC, ETH, SOL, AVAX)."""
    ticker_list = [t.strip().upper() for t in tickers.split(",")]
    results = {}

    equity_tickers = [t for t in ticker_list if not _is_crypto(t)]
    crypto_tickers = [t for t in ticker_list if _is_crypto(t)]

    # Fetch equities via yfinance
    if equity_tickers:
        df = yf.download(
            equity_tickers,
            period=period,
            interval=interval,
            group_by="ticker" if len(equity_tickers) > 1 else None,
            progress=False,
        )
        if len(equity_tickers) == 1:
            bars = _df_to_bars(df)
            results[equity_tickers[0]] = bars
            # Cache
            async with get_db(ctx.context.db_path) as db:
                await cache_prices(db, equity_tickers[0], bars)
        else:
            for t in equity_tickers:
                try:
                    sub = df[t].dropna()
                    bars = _df_to_bars(sub)
                    results[t] = bars
                    async with get_db(ctx.context.db_path) as db:
                        await cache_prices(db, t, bars)
                except (KeyError, AttributeError):
                    results[t] = []

    # Fetch crypto via CoinGecko
    days_map = {"1mo": 30, "3mo": 90, "6mo": 180, "1y": 365, "2y": 730}
    days = days_map.get(period, 180)
    for t in crypto_tickers:
        coin_id = CRYPTO_MAP[t]
        try:
            bars = await _fetch_crypto_ohlcv(coin_id, days)
            results[t] = bars
            async with get_db(ctx.context.db_path) as db:
                await cache_prices(db, t, bars, source="coingecko")
        except Exception as e:
            results[t] = {"error": str(e)}

    return json.dumps(results)


@function_tool
async def fetch_crypto_data(
    ctx: RunContextWrapper[AppContext],
    coins: str,
    days: int = 180,
) -> str:
    """Fetch crypto OHLCV from CoinGecko. Coins: comma-separated (BTC, ETH, SOL, AVAX)."""
    coin_list = [c.strip().upper() for c in coins.split(",")]
    results = {}
    for c in coin_list:
        coin_id = CRYPTO_MAP.get(c)
        if not coin_id:
            results[c] = {"error": f"Unknown crypto: {c}"}
            continue
        try:
            bars = await _fetch_crypto_ohlcv(coin_id, days)
            results[c] = bars
        except Exception as e:
            results[c] = {"error": str(e)}
    return json.dumps(results)
