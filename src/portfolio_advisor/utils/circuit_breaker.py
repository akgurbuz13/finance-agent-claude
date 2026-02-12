"""Circuit breaker for external API providers."""

from __future__ import annotations

import asyncio
import logging
import time
from enum import Enum

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Circuit breaker: CLOSED -> OPEN (after N failures) -> HALF_OPEN (after recovery) -> CLOSED.

    Each external provider gets its own instance.
    In HALF_OPEN state, only one trial call is allowed at a time.
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 300.0,
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: float = 0.0
        self._half_open_in_flight = False  # limit HALF_OPEN to single trial call
        self._lock = asyncio.Lock()

    @property
    def state(self) -> CircuitState:
        if self._state == CircuitState.OPEN:
            if time.monotonic() - self._last_failure_time >= self.recovery_timeout:
                return CircuitState.HALF_OPEN
        return self._state

    @property
    def is_closed(self) -> bool:
        return self.state == CircuitState.CLOSED

    async def __aenter__(self) -> CircuitBreaker:
        await self.before_call()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool:
        if exc_type is None:
            await self.record_success()
        else:
            await self.record_failure()
        return False  # Don't suppress exceptions

    async def before_call(self) -> None:
        """Check if call is allowed. Raises CircuitOpenError if not.

        Uses lock to atomically check state and transition OPEN->HALF_OPEN.
        Only one trial call is permitted in HALF_OPEN state.
        """
        async with self._lock:
            current = self.state
            if current == CircuitState.OPEN:
                raise CircuitOpenError(
                    f"Circuit breaker '{self.name}' is OPEN "
                    f"(failures={self._failure_count}, "
                    f"recovery in {self._time_until_recovery():.0f}s)"
                )
            if current == CircuitState.HALF_OPEN:
                if self._half_open_in_flight:
                    raise CircuitOpenError(
                        f"Circuit breaker '{self.name}' is HALF_OPEN "
                        f"(trial call already in flight)"
                    )
                # Atomically claim the single trial slot
                self._state = CircuitState.HALF_OPEN
                self._half_open_in_flight = True

    async def record_success(self) -> None:
        """Record a successful call — reset to CLOSED."""
        async with self._lock:
            current = self.state
            if current in (CircuitState.HALF_OPEN, CircuitState.CLOSED):
                self._failure_count = 0
                self._half_open_in_flight = False
                if current != CircuitState.CLOSED:
                    logger.info(f"Circuit breaker '{self.name}': HALF_OPEN -> CLOSED")
                self._state = CircuitState.CLOSED

    async def record_failure(self) -> None:
        """Record a failed call — potentially trip to OPEN."""
        async with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.monotonic()

            current = self.state
            if current == CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
                self._half_open_in_flight = False
                logger.warning(f"Circuit breaker '{self.name}': HALF_OPEN -> OPEN")
            elif self._failure_count >= self.failure_threshold:
                self._state = CircuitState.OPEN
                logger.warning(
                    f"Circuit breaker '{self.name}': CLOSED -> OPEN "
                    f"(failures={self._failure_count})"
                )

    def _time_until_recovery(self) -> float:
        elapsed = time.monotonic() - self._last_failure_time
        return max(0.0, self.recovery_timeout - elapsed)

    def status(self) -> dict:
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self._failure_count,
            "failure_threshold": self.failure_threshold,
            "recovery_timeout": self.recovery_timeout,
            "time_until_recovery": round(self._time_until_recovery(), 1)
            if self._state == CircuitState.OPEN
            else 0,
        }


class CircuitOpenError(Exception):
    """Raised when a circuit breaker is open and calls are blocked."""
