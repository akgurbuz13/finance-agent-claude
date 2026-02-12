"""Tests for FRED provider."""

from unittest.mock import AsyncMock, MagicMock


from portfolio_advisor.providers.fred_provider import FREDProvider


class TestFREDProvider:
    """Tests for the FRED API provider."""

    def _make_provider(self, api_key="test-key"):
        return FREDProvider(api_key)

    async def test_fetch_series_returns_observations(self):
        provider = self._make_provider()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "observations": [
                {"date": "2024-01-15", "value": "4.25"},
                {"date": "2024-01-14", "value": "4.20"},
                {"date": "2024-01-13", "value": "."},  # missing data marker
            ]
        }

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.is_closed = False
        provider._client = mock_client

        result = await provider.fetch_series("DGS10", limit=5)
        assert result is not None
        assert len(result) == 2  # "." value should be filtered out
        assert result[0]["value"] == 4.25
        assert result[0]["date"] == "2024-01-15"

    async def test_fetch_series_returns_none_without_api_key(self):
        provider = FREDProvider("")
        result = await provider.fetch_series("DGS10")
        assert result is None

    async def test_fetch_treasury_yields_returns_dict(self):
        provider = self._make_provider()

        async def mock_fetch_series(series_id, **kwargs):
            data = {
                "DGS2": [{"date": "2024-01-15", "value": 4.5}],
                "DGS5": [{"date": "2024-01-15", "value": 4.3}],
                "DGS10": [{"date": "2024-01-15", "value": 4.2}],
                "DGS30": [{"date": "2024-01-15", "value": 4.4}],
            }
            return data.get(series_id)

        provider.fetch_series = mock_fetch_series

        result = await provider.fetch_treasury_yields()
        assert result is not None
        assert result["2y"] == 4.5
        assert result["10y"] == 4.2
        assert result["30y"] == 4.4

    async def test_fetch_vix_returns_float(self):
        provider = self._make_provider()
        provider.fetch_series = AsyncMock(
            return_value=[{"date": "2024-01-15", "value": 18.5}]
        )
        result = await provider.fetch_vix()
        assert result == 18.5

    async def test_fetch_vix_returns_none_on_failure(self):
        provider = self._make_provider()
        provider.fetch_series = AsyncMock(return_value=None)
        result = await provider.fetch_vix()
        assert result is None

    async def test_fetch_credit_spread_returns_float(self):
        provider = self._make_provider()
        provider.fetch_series = AsyncMock(
            return_value=[{"date": "2024-01-15", "value": 3.75}]
        )
        result = await provider.fetch_credit_spread()
        assert result == 3.75

    async def test_status_reports_key_and_circuit(self):
        provider = self._make_provider()
        status = provider.status()
        assert status["provider"] == "fred"
        assert status["has_key"] is True
        assert "rate_limiter" in status
        assert "circuit_breaker" in status

    async def test_close_closes_client(self):
        provider = self._make_provider()
        mock_client = AsyncMock()
        mock_client.is_closed = False
        provider._client = mock_client

        await provider.close()
        mock_client.aclose.assert_awaited_once()

    async def test_close_noop_when_no_client(self):
        provider = self._make_provider()
        await provider.close()  # should not raise
