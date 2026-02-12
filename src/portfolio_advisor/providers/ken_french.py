"""Kenneth French Data Library — actual Fama-French 3-factor data."""

from __future__ import annotations

import io
import logging
import time
import zipfile
from datetime import datetime, timedelta

import httpx
import pandas as pd

logger = logging.getLogger(__name__)

# URL for daily Fama-French 3 factors
FF3_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip"

# Module-level cache
_ff3_cache: dict | None = None
_ff3_cache_time: float = 0.0
_CACHE_TTL = 86400  # 24 hours


async def fetch_french_factors(lookback_days: int = 365) -> dict | None:
    """Download daily FF3 factors from Kenneth French data library.

    Returns dict with keys: mkt_rf, smb, hml, rf as pandas Series (date-indexed).
    Caches locally for 24 hours.
    """
    global _ff3_cache, _ff3_cache_time

    # Check cache
    if _ff3_cache is not None and (time.monotonic() - _ff3_cache_time) < _CACHE_TTL:
        return _ff3_cache

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(FF3_URL)
            resp.raise_for_status()

        # Parse the ZIP file
        z = zipfile.ZipFile(io.BytesIO(resp.content))
        csv_name = z.namelist()[0]
        raw = z.read(csv_name).decode("utf-8")

        # Find the data section (skip header lines)
        lines = raw.strip().split("\n")
        start_idx = None
        end_idx = None
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped and stripped[0].isdigit() and len(stripped.split(",")[0].strip()) == 8:
                if start_idx is None:
                    start_idx = i
                end_idx = i
            elif start_idx is not None and not stripped[0].isdigit():
                break

        if start_idx is None:
            logger.warning("Could not parse French factor data")
            return None

        # Parse CSV data
        data_lines = lines[start_idx:end_idx + 1]
        records = []
        for line in data_lines:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 4:
                try:
                    dt = datetime.strptime(parts[0], "%Y%m%d")
                    records.append({
                        "date": dt,
                        "mkt_rf": float(parts[1]) / 100,  # Convert from pct
                        "smb": float(parts[2]) / 100,
                        "hml": float(parts[3]) / 100,
                        "rf": float(parts[4]) / 100 if len(parts) > 4 else 0.0,
                    })
                except (ValueError, IndexError):
                    continue

        if not records:
            return None

        df = pd.DataFrame(records)
        df.set_index("date", inplace=True)
        df.sort_index(inplace=True)

        # Filter to lookback period
        cutoff = datetime.now() - timedelta(days=lookback_days)
        df = df[df.index >= cutoff]

        if len(df) < 30:
            logger.warning(f"Insufficient French factor data ({len(df)} rows)")
            return None

        result = {
            "mkt_rf": df["mkt_rf"],
            "smb": df["smb"],
            "hml": df["hml"],
            "rf": df["rf"],
        }

        # Cache it
        _ff3_cache = result
        _ff3_cache_time = time.monotonic()
        logger.info(f"Fetched French factors: {len(df)} daily observations")
        return result

    except Exception as e:
        logger.warning(f"Failed to fetch French factors: {e}")
        return None
