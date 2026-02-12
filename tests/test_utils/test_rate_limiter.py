"""Tests for AsyncTokenBucketRateLimiter."""

import asyncio
import time


from portfolio_advisor.utils.rate_limiter import AsyncTokenBucketRateLimiter


class TestAsyncTokenBucketRateLimiter:
    """Tests for the async token-bucket rate limiter."""

    async def test_initial_tokens_available(self):
        rl = AsyncTokenBucketRateLimiter("test", max_tokens=5, refill_rate=1.0)
        assert rl.available >= 4.9  # approximately 5, allowing for float precision

    async def test_acquire_decrements_tokens(self):
        rl = AsyncTokenBucketRateLimiter("test", max_tokens=5, refill_rate=1.0)
        await rl.acquire()
        assert rl.available < 5.0
        assert rl.available >= 3.9  # approximately 4

    async def test_acquire_multiple_tokens(self):
        rl = AsyncTokenBucketRateLimiter("test", max_tokens=3, refill_rate=0.1)
        await rl.acquire()
        await rl.acquire()
        await rl.acquire()
        # All 3 tokens consumed; available should be near 0
        assert rl.available < 1.0

    async def test_acquire_blocks_when_empty(self):
        """When bucket is empty, acquire() should block until refill."""
        rl = AsyncTokenBucketRateLimiter("test", max_tokens=1, refill_rate=10.0)
        await rl.acquire()  # consume the single token

        start = time.monotonic()
        await rl.acquire()  # should block briefly (~0.1s at 10 tokens/sec)
        elapsed = time.monotonic() - start

        # Should have waited some time for refill (at least a few ms)
        assert elapsed >= 0.05

    async def test_refill_does_not_exceed_max(self):
        rl = AsyncTokenBucketRateLimiter("test", max_tokens=3, refill_rate=100.0)
        await asyncio.sleep(0.1)  # let refill accumulate
        assert rl.available <= 3.0  # capped at max_tokens

    async def test_concurrent_acquires(self):
        """Multiple concurrent tasks should all get tokens without crashing."""
        rl = AsyncTokenBucketRateLimiter("test", max_tokens=10, refill_rate=100.0)

        results = []

        async def worker(idx):
            await rl.acquire()
            results.append(idx)

        tasks = [asyncio.create_task(worker(i)) for i in range(10)]
        await asyncio.gather(*tasks)

        assert len(results) == 10

    async def test_status_returns_correct_structure(self):
        rl = AsyncTokenBucketRateLimiter("fred", max_tokens=120, refill_rate=2.0)
        status = rl.status()
        assert status["name"] == "fred"
        assert status["max_tokens"] == 120
        assert status["refill_rate_per_sec"] == 2.0
        assert "available_tokens" in status

    async def test_zero_refill_rate_still_works(self):
        """Even with 0 refill rate, initial tokens should be usable."""
        rl = AsyncTokenBucketRateLimiter("test", max_tokens=2, refill_rate=0.0)
        await rl.acquire()
        await rl.acquire()
        # Third acquire would block forever with 0 refill, but we don't test that
        assert rl.available < 0.5
