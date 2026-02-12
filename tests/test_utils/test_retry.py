"""Tests for retry_with_backoff."""


import httpx
import pytest

from portfolio_advisor.utils.circuit_breaker import CircuitOpenError
from portfolio_advisor.utils.retry import retry_with_backoff


class TestRetryWithBackoff:
    """Tests for exponential backoff retry logic."""

    async def test_succeeds_on_first_try(self):
        call_count = 0

        async def success():
            nonlocal call_count
            call_count += 1
            return "ok"

        result = await retry_with_backoff(success, max_retries=3, base_delay=0.01)
        assert result == "ok"
        assert call_count == 1

    async def test_retries_on_connection_error(self):
        call_count = 0

        async def flaky():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("connection refused")
            return "recovered"

        result = await retry_with_backoff(flaky, max_retries=3, base_delay=0.01)
        assert result == "recovered"
        assert call_count == 3

    async def test_exhausts_retries_and_raises(self):
        call_count = 0

        async def always_fails():
            nonlocal call_count
            call_count += 1
            raise TimeoutError("timed out")

        with pytest.raises(TimeoutError):
            await retry_with_backoff(always_fails, max_retries=2, base_delay=0.01)
        assert call_count == 3  # 1 initial + 2 retries

    async def test_circuit_open_never_retried(self):
        call_count = 0

        async def circuit_open():
            nonlocal call_count
            call_count += 1
            raise CircuitOpenError("circuit is open")

        with pytest.raises(CircuitOpenError):
            await retry_with_backoff(circuit_open, max_retries=5, base_delay=0.01)
        assert call_count == 1  # no retries

    async def test_4xx_http_errors_not_retried(self):
        """Client errors (400, 401, 403, 404, 422) should not be retried."""
        for status_code in [400, 401, 403, 404, 422]:
            call_count = 0
            request = httpx.Request("GET", "https://example.com")
            response = httpx.Response(status_code, request=request)

            async def client_error():
                nonlocal call_count
                call_count += 1
                raise httpx.HTTPStatusError(
                    f"HTTP {status_code}", request=request, response=response
                )

            with pytest.raises(httpx.HTTPStatusError):
                await retry_with_backoff(client_error, max_retries=3, base_delay=0.01)
            assert call_count == 1, f"HTTP {status_code} should not be retried"

    async def test_5xx_http_errors_are_retried(self):
        """Server errors (500, 502, 503) should be retried."""
        call_count = 0
        request = httpx.Request("GET", "https://example.com")

        async def server_error():
            nonlocal call_count
            call_count += 1
            response = httpx.Response(500, request=request)
            raise httpx.HTTPStatusError("HTTP 500", request=request, response=response)

        with pytest.raises(httpx.HTTPStatusError):
            await retry_with_backoff(server_error, max_retries=2, base_delay=0.01)
        assert call_count == 3  # 1 initial + 2 retries

    async def test_non_retryable_exceptions_not_retried(self):
        call_count = 0

        async def value_error():
            nonlocal call_count
            call_count += 1
            raise ValueError("bad input")

        with pytest.raises(ValueError):
            await retry_with_backoff(value_error, max_retries=3, base_delay=0.01)
        assert call_count == 1
