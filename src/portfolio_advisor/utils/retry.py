"""Retry with exponential backoff and jitter for async operations."""

from __future__ import annotations

import asyncio
import logging
import random
from typing import TypeVar

import httpx

from portfolio_advisor.utils.circuit_breaker import CircuitOpenError

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Default retryable exceptions
RETRYABLE_EXCEPTIONS: tuple[type[Exception], ...] = (
    httpx.HTTPStatusError,
    httpx.ConnectTimeout,
    httpx.ReadTimeout,
    httpx.ConnectError,
    ConnectionError,
    TimeoutError,
    OSError,
)


async def retry_with_backoff(
    coro_factory,
    *,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    retryable: tuple[type[Exception], ...] = RETRYABLE_EXCEPTIONS,
    operation_name: str = "",
):
    """Execute an async callable with exponential backoff and jitter.

    Args:
        coro_factory: A callable that returns a coroutine (called on each attempt).
        max_retries: Maximum number of retry attempts (not counting the initial try).
        base_delay: Base delay in seconds for exponential backoff.
        max_delay: Maximum delay cap in seconds.
        retryable: Tuple of exception types that trigger a retry.
        operation_name: Human-readable label for logging.

    Returns:
        The result of the successful call.

    Raises:
        The last exception if all retries are exhausted.
        CircuitOpenError immediately (never retried).
    """
    last_exc: Exception | None = None

    for attempt in range(1 + max_retries):
        try:
            return await coro_factory()
        except CircuitOpenError:
            raise  # Never retry circuit-open
        except retryable as exc:
            last_exc = exc

            # Check for non-retryable HTTP status codes
            if isinstance(exc, httpx.HTTPStatusError):
                status = exc.response.status_code
                if status in (400, 401, 403, 404, 422):
                    raise  # Client errors are not retryable

            if attempt < max_retries:
                delay = min(base_delay * (2**attempt), max_delay)
                jitter = random.uniform(0, delay * 0.3)
                wait = delay + jitter
                name = operation_name or "operation"
                logger.warning(
                    f"Retry {attempt + 1}/{max_retries} for {name} "
                    f"after {type(exc).__name__}: {exc}. "
                    f"Waiting {wait:.1f}s"
                )
                await asyncio.sleep(wait)
            else:
                logger.error(
                    f"All {max_retries} retries exhausted for "
                    f"{operation_name or 'operation'}: {exc}"
                )

    if last_exc is not None:
        raise last_exc
    raise RuntimeError("retry_with_backoff: unreachable")
