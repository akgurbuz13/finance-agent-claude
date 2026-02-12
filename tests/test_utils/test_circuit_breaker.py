"""Tests for CircuitBreaker."""


import pytest

from portfolio_advisor.utils.circuit_breaker import (
    CircuitBreaker,
    CircuitOpenError,
    CircuitState,
)


class TestCircuitBreaker:
    """Tests for circuit breaker state transitions and error handling."""

    async def test_initial_state_is_closed(self):
        cb = CircuitBreaker("test")
        assert cb.state == CircuitState.CLOSED
        assert cb.is_closed

    async def test_single_failure_stays_closed(self):
        cb = CircuitBreaker("test", failure_threshold=5)
        await cb.record_failure()
        assert cb.state == CircuitState.CLOSED

    async def test_opens_after_threshold_failures(self):
        cb = CircuitBreaker("test", failure_threshold=3)
        for _ in range(3):
            await cb.record_failure()
        assert cb.state == CircuitState.OPEN

    async def test_open_circuit_raises_error(self):
        cb = CircuitBreaker("test", failure_threshold=1)
        await cb.record_failure()
        assert cb.state == CircuitState.OPEN

        with pytest.raises(CircuitOpenError):
            await cb.before_call()

    async def test_success_resets_failure_count(self):
        cb = CircuitBreaker("test", failure_threshold=5)
        await cb.record_failure()
        await cb.record_failure()
        await cb.record_success()
        # After success, failure count resets — need 5 more failures to trip
        for _ in range(4):
            await cb.record_failure()
        assert cb.state == CircuitState.CLOSED  # only 4 failures, not 5

    async def test_half_open_after_recovery_timeout(self):
        cb = CircuitBreaker("test", failure_threshold=1, recovery_timeout=0.1)
        await cb.record_failure()
        assert cb._state == CircuitState.OPEN

        # Wait for recovery
        import asyncio
        await asyncio.sleep(0.15)

        assert cb.state == CircuitState.HALF_OPEN

    async def test_half_open_success_closes_circuit(self):
        cb = CircuitBreaker("test", failure_threshold=1, recovery_timeout=0.1)
        await cb.record_failure()

        import asyncio
        await asyncio.sleep(0.15)
        assert cb.state == CircuitState.HALF_OPEN

        await cb.record_success()
        assert cb.state == CircuitState.CLOSED
        assert cb._failure_count == 0

    async def test_half_open_failure_reopens_circuit(self):
        cb = CircuitBreaker("test", failure_threshold=1, recovery_timeout=0.1)
        await cb.record_failure()

        import asyncio
        await asyncio.sleep(0.15)
        assert cb.state == CircuitState.HALF_OPEN

        await cb.record_failure()
        assert cb._state == CircuitState.OPEN

    async def test_context_manager_success(self):
        cb = CircuitBreaker("test", failure_threshold=5)
        async with cb:
            pass  # no exception = success
        assert cb._failure_count == 0

    async def test_context_manager_failure(self):
        cb = CircuitBreaker("test", failure_threshold=5)
        with pytest.raises(ValueError):
            async with cb:
                raise ValueError("boom")
        assert cb._failure_count == 1

    async def test_status_dict_structure(self):
        cb = CircuitBreaker("fred", failure_threshold=5, recovery_timeout=300)
        status = cb.status()
        assert status["name"] == "fred"
        assert status["state"] == "closed"
        assert status["failure_count"] == 0
        assert status["failure_threshold"] == 5
        assert status["recovery_timeout"] == 300

    async def test_circuit_open_error_message_includes_name(self):
        cb = CircuitBreaker("massive", failure_threshold=1)
        await cb.record_failure()
        with pytest.raises(CircuitOpenError, match="massive"):
            await cb.before_call()
