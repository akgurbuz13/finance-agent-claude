"""Tests for advanced portfolio tools — transaction costs."""


from portfolio_advisor.tools.advanced_portfolio import (
    _TICKER_COST_BPS,
    _DEFAULT_COST_BPS,
    _get_ticker_cost_bps,
)


class TestTickerSpecificCosts:
    """Tests for the ticker-specific transaction cost model."""

    def test_etf_costs_are_low(self):
        for etf in ["SPY", "QQQ", "IWM"]:
            cost = _get_ticker_cost_bps(etf)
            assert cost <= 5, f"{etf} cost {cost} bps > 5"

    def test_crypto_costs_are_high(self):
        for crypto in ["BTC", "ETH", "SOL"]:
            cost = _get_ticker_cost_bps(crypto)
            assert cost >= 25, f"{crypto} cost {cost} bps < 25"

    def test_em_etf_costs_are_moderate(self):
        cost = _get_ticker_cost_bps("EEM")
        assert cost >= 10

    def test_large_cap_stocks_reasonable(self):
        for ticker in ["AAPL", "MSFT", "NVDA", "AMZN"]:
            cost = _get_ticker_cost_bps(ticker)
            assert 3 <= cost <= 10, f"{ticker} cost {cost} out of range"

    def test_unknown_ticker_gets_default(self):
        cost = _get_ticker_cost_bps("XYZABC123")
        assert cost == _DEFAULT_COST_BPS

    def test_case_insensitive_lookup(self):
        assert _get_ticker_cost_bps("spy") == _get_ticker_cost_bps("SPY")
        assert _get_ticker_cost_bps("btc") == _get_ticker_cost_bps("BTC")

    def test_bond_etfs_have_low_costs(self):
        for ticker in ["TLT", "IEF", "AGG", "BND"]:
            cost = _get_ticker_cost_bps(ticker)
            assert cost <= 5, f"{ticker} cost {cost} bps > 5"

    def test_cost_table_has_reasonable_range(self):
        """All costs should be between 1 and 100 bps."""
        for ticker, cost in _TICKER_COST_BPS.items():
            assert 1 <= cost <= 100, f"{ticker} has unreasonable cost: {cost} bps"
