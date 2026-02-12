"""Tests for multi-key rotation mechanics in Massive and Alpha Vantage providers.

These tests verify round-robin key selection, exhausted key skipping,
and concurrent access safety. They use fake keys (no network calls needed).

Run with: pytest tests/test_providers/test_multi_key_rotation.py -v
"""

from __future__ import annotations

import asyncio

import pytest

from portfolio_advisor.providers.massive_provider import MassiveProvider
from portfolio_advisor.providers.alpha_vantage_provider import AlphaVantageProvider

# ══════════════════════════════════════════════════════════════════════════════
# Massive: Round-Robin Key Rotation
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.live
async def test_massive_round_robin_4_keys():
    """Create with 4 fake keys, _next_key_index() cycles through 0,1,2,3."""
    provider = MassiveProvider(["key-0", "key-1", "key-2", "key-3"])

    indices = []
    for _ in range(4):
        idx = await provider._next_key_index()
        indices.append(idx)

    assert indices == [0, 1, 2, 3], f"Expected [0,1,2,3], got {indices}"

    # Verify it wraps around
    next_idx = await provider._next_key_index()
    assert next_idx == 0, f"Expected wrap-around to 0, got {next_idx}"


# ══════════════════════════════════════════════════════════════════════════════
# Alpha Vantage: Exhausted Key Skipping
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.live
async def test_alpha_vantage_exhausted_key_skip():
    """With 2 keys, set _daily_calls[0] = 25 (limit), _next_key_index() returns 1."""
    provider = AlphaVantageProvider(["key-0", "key-1"])

    # Exhaust key 0
    provider._daily_calls[0] = 25  # _PER_KEY_DAILY_LIMIT = 25

    idx = await provider._next_key_index()
    # The round-robin starts from wherever the cycle is. If it hits key 0 first,
    # it should skip it and return key 1. If it hits key 1 first, it returns 1.
    assert idx == 1 or idx == 0, f"Unexpected index: {idx}"
    if idx is not None:
        # Whichever key is returned, its daily calls should be below limit
        assert provider._daily_calls[idx] < 25, (
            f"Returned key {idx} but it's exhausted ({provider._daily_calls[idx]} calls)"
        )


@pytest.mark.live
async def test_alpha_vantage_all_keys_exhausted():
    """Set both keys at daily limit, _next_key_index() returns None."""
    provider = AlphaVantageProvider(["key-0", "key-1"])

    # Exhaust both keys
    provider._daily_calls[0] = 25
    provider._daily_calls[1] = 25

    idx = await provider._next_key_index()
    assert idx is None, f"Expected None when all keys exhausted, got {idx}"


# ══════════════════════════════════════════════════════════════════════════════
# Concurrent Rotation Safety
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.live
async def test_concurrent_rotation_no_race():
    """20 concurrent _next_key_index() calls produce valid indices with no crash."""
    provider = MassiveProvider(["key-0", "key-1", "key-2", "key-3"])
    num_keys = 4

    async def get_index():
        return await provider._next_key_index()

    # Launch 20 concurrent calls
    tasks = [asyncio.create_task(get_index()) for _ in range(20)]
    results = await asyncio.gather(*tasks)

    # All results should be valid key indices
    assert len(results) == 20
    for idx in results:
        assert isinstance(idx, int), f"Expected int, got {type(idx)}"
        assert 0 <= idx < num_keys, f"Index {idx} out of range [0, {num_keys})"

    # Each index should appear roughly equally (5 each for 20 calls / 4 keys)
    from collections import Counter
    counts = Counter(results)
    assert len(counts) == num_keys, (
        f"Expected all {num_keys} keys to be used, got {dict(counts)}"
    )
    for key_idx, count in counts.items():
        assert count == 5, f"Key {key_idx} used {count} times, expected 5"
