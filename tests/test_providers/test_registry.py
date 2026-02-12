"""Tests for ProviderRegistry fallback chains."""

from unittest.mock import AsyncMock, MagicMock


from portfolio_advisor.providers.registry import ProviderRegistry, create_registry


class TestProviderRegistry:
    """Tests for provider registry fallback chains."""

    def _make_registry(self, fred=None, massive=None, alpha_vantage=None):
        """Create a registry with mock providers."""
        return ProviderRegistry(
            fred=fred,
            massive=massive,
            alpha_vantage=alpha_vantage,
            yfinance=MagicMock(),
            coingecko=MagicMock(),
        )

    # ── Treasury yields fallback chain ──────────────────────────────────

    async def test_yields_massive_first(self):
        massive = MagicMock()
        massive.fetch_treasury_yields = AsyncMock(
            return_value={"2y": 4.5, "10y": 4.2}
        )
        fred = MagicMock()
        fred.fetch_treasury_yields = AsyncMock(return_value={"10y": 4.1})

        reg = self._make_registry(fred=fred, massive=massive)
        data, source = await reg.fetch_treasury_yields()

        assert source == "massive"
        assert data["2y"] == 4.5
        fred.fetch_treasury_yields.assert_not_awaited()

    async def test_yields_falls_back_to_fred(self):
        massive = MagicMock()
        massive.fetch_treasury_yields = AsyncMock(return_value=None)
        fred = MagicMock()
        fred.fetch_treasury_yields = AsyncMock(
            return_value={"2y": 4.5, "10y": 4.2}
        )

        reg = self._make_registry(fred=fred, massive=massive)
        data, source = await reg.fetch_treasury_yields()

        assert source == "fred"
        assert data["10y"] == 4.2

    async def test_yields_falls_back_to_yfinance(self):
        reg = self._make_registry()  # no fred or massive
        reg.yfinance.fetch_treasury_yields = AsyncMock(
            return_value={"10y": 4.0}
        )
        data, source = await reg.fetch_treasury_yields()

        assert source == "yfinance"
        assert data["10y"] == 4.0

    async def test_yields_returns_unavailable_when_all_fail(self):
        reg = self._make_registry()
        reg.yfinance.fetch_treasury_yields = AsyncMock(return_value=None)
        data, source = await reg.fetch_treasury_yields()

        assert source == "unavailable"
        assert data is None

    # ── VIX fallback chain ──────────────────────────────────────────────

    async def test_vix_fred_first(self):
        fred = MagicMock()
        fred.fetch_vix = AsyncMock(return_value=18.5)
        reg = self._make_registry(fred=fred)

        val, source = await reg.fetch_vix()
        assert source == "fred"
        assert val == 18.5

    async def test_vix_falls_back_to_yfinance(self):
        fred = MagicMock()
        fred.fetch_vix = AsyncMock(return_value=None)
        reg = self._make_registry(fred=fred)
        reg.yfinance.fetch_vix = AsyncMock(return_value=19.2)

        val, source = await reg.fetch_vix()
        assert source == "yfinance"
        assert val == 19.2

    # ── Credit spread ───────────────────────────────────────────────────

    async def test_credit_fred_only(self):
        fred = MagicMock()
        fred.fetch_credit_spread = AsyncMock(return_value=3.45)
        reg = self._make_registry(fred=fred)

        val, source = await reg.fetch_credit_spread()
        assert source == "fred"
        assert val == 3.45

    async def test_credit_unavailable_without_fred(self):
        reg = self._make_registry()
        val, source = await reg.fetch_credit_spread()
        assert source == "unavailable"
        assert val is None

    # ── Fundamentals fallback chain ─────────────────────────────────────

    async def test_fundamentals_massive_first(self):
        massive = MagicMock()
        massive.fetch_fundamentals = AsyncMock(return_value={"pe_ratio": 28.5})
        av = MagicMock()
        av.fetch_fundamentals = AsyncMock(return_value={"pe_ratio": 27.0})

        reg = self._make_registry(massive=massive, alpha_vantage=av)
        data, source = await reg.fetch_fundamentals("AAPL")

        assert source == "massive"
        assert data["pe_ratio"] == 28.5
        av.fetch_fundamentals.assert_not_awaited()

    async def test_fundamentals_falls_back_to_alpha_vantage(self):
        massive = MagicMock()
        massive.fetch_fundamentals = AsyncMock(return_value=None)
        av = MagicMock()
        av.fetch_fundamentals = AsyncMock(return_value={"pe_ratio": 27.0})

        reg = self._make_registry(massive=massive, alpha_vantage=av)
        data, source = await reg.fetch_fundamentals("AAPL")

        assert source == "alpha_vantage"

    async def test_fundamentals_unavailable_when_all_fail(self):
        reg = self._make_registry()
        data, source = await reg.fetch_fundamentals("AAPL")
        assert source == "unavailable"
        assert data is None

    # ── Earnings ────────────────────────────────────────────────────────

    async def test_earnings_massive(self):
        massive = MagicMock()
        massive.fetch_earnings = AsyncMock(
            return_value=[{"ticker": "AAPL", "eps_estimate": 1.50}]
        )
        reg = self._make_registry(massive=massive)
        entries, source = await reg.fetch_earnings(["AAPL"])
        assert source == "massive_benzinga"
        assert len(entries) == 1

    async def test_earnings_unavailable_without_massive(self):
        reg = self._make_registry()
        entries, source = await reg.fetch_earnings(["AAPL"])
        assert source == "unavailable"
        assert entries == []

    # ── News ────────────────────────────────────────────────────────────

    async def test_news_returns_empty_without_massive(self):
        reg = self._make_registry()
        articles = await reg.fetch_news(["AAPL"])
        assert articles == []

    # ── Market movers ───────────────────────────────────────────────────

    async def test_market_movers_returns_none_without_massive(self):
        reg = self._make_registry()
        result = await reg.fetch_market_movers()
        assert result is None

    # ── Close ───────────────────────────────────────────────────────────

    async def test_close_calls_all_providers(self):
        fred = MagicMock()
        fred.close = AsyncMock()
        massive = MagicMock()
        massive.close = AsyncMock()
        av = MagicMock()
        av.close = AsyncMock()

        reg = self._make_registry(fred=fred, massive=massive, alpha_vantage=av)
        await reg.close()

        fred.close.assert_awaited_once()
        massive.close.assert_awaited_once()
        av.close.assert_awaited_once()

    # ── Provider status ─────────────────────────────────────────────────

    async def test_provider_status_includes_configured_providers(self):
        fred = MagicMock()
        fred.status.return_value = {"provider": "fred"}
        reg = self._make_registry(fred=fred)
        reg.yfinance.status.return_value = {"provider": "yfinance"}
        reg.coingecko.status.return_value = {"provider": "coingecko"}

        status = reg.provider_status()
        assert "fred" in status
        assert "yfinance" in status
        assert "coingecko" in status


class TestCreateRegistry:
    """Tests for the create_registry factory."""

    def test_no_keys_creates_minimal_registry(self):
        settings = MagicMock()
        settings.fred_api_key = ""
        settings.get_massive_keys.return_value = []
        settings.get_alpha_vantage_keys.return_value = []

        reg = create_registry(settings)
        assert reg.fred is None
        assert reg.massive is None
        assert reg.alpha_vantage is None
        assert reg.yfinance is not None
        assert reg.coingecko is not None

    def test_all_keys_creates_full_registry(self):
        settings = MagicMock()
        settings.fred_api_key = "test-fred-key"
        settings.get_massive_keys.return_value = ["key1", "key2", "key3", "key4"]
        settings.get_alpha_vantage_keys.return_value = ["av-key1", "av-key2"]

        reg = create_registry(settings)
        assert reg.fred is not None
        assert reg.massive is not None
        assert reg.alpha_vantage is not None
