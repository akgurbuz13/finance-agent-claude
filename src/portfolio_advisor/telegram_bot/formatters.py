"""Markdown formatting helpers for Telegram messages."""

from __future__ import annotations


def format_portfolio_table(positions: list[dict], cash_pct: float) -> str:
    """Format portfolio positions as a readable table."""
    if not positions:
        return "*Portfolio*: 100% Cash\nNo positions yet."

    lines = ["*Current Portfolio*", ""]
    lines.append("`Ticker   Weight  Class`")
    lines.append("`" + "-" * 28 + "`")
    for p in positions:
        ticker = p["ticker"].ljust(8)
        weight = f"{p['weight_pct']:.1f}%".rjust(7)
        asset_class = p.get("asset_class", "equity")[:6]
        lines.append(f"`{ticker} {weight}  {asset_class}`")

    lines.append(f"`{'CASH'.ljust(8)} {f'{cash_pct:.1f}%'.rjust(7)}  cash`")
    lines.append("")
    total = sum(p["weight_pct"] for p in positions) + cash_pct
    lines.append(f"_Total: {total:.1f}%_")
    return "\n".join(lines)


def format_preferences(prefs: dict) -> str:
    """Format user preferences for display."""
    lines = [
        "*User Preferences*",
        "",
        f"Risk Tolerance: `{prefs.get('risk_tolerance', 'moderate')}`",
        f"Time Horizon: `{prefs.get('time_horizon', 'medium')}`",
        f"Cash Target: `{prefs.get('cash_target_pct', 10.0)}%`",
        f"Max Position: `{prefs.get('max_position_pct', 15.0)}%`",
    ]
    excluded = prefs.get("excluded_assets", [])
    if excluded:
        lines.append(f"Excluded: `{', '.join(excluded)}`")
    regions = prefs.get("allowed_regions", [])
    if regions:
        lines.append(f"Regions: `{', '.join(regions)}`")
    return "\n".join(lines)


def format_watchlist(watchlist: list[str]) -> str:
    """Format watchlist for display."""
    if not watchlist:
        return "*Watchlist*: Empty"

    # Group by type
    crypto_set = {"BTC", "ETH", "SOL", "AVAX"}
    bond_set = {"TLT", "IEF", "HYG", "AGG", "BND"}
    commodity_set = {"GLD", "SLV", "USO"}

    equities = [t for t in watchlist if t not in crypto_set | bond_set | commodity_set]
    bonds = [t for t in watchlist if t in bond_set]
    commodities = [t for t in watchlist if t in commodity_set]
    crypto = [t for t in watchlist if t in crypto_set]

    lines = [f"*Watchlist* ({len(watchlist)} tickers)", ""]
    if equities:
        lines.append(f"Equities: `{' '.join(sorted(equities))}`")
    if bonds:
        lines.append(f"Bonds: `{' '.join(sorted(bonds))}`")
    if commodities:
        lines.append(f"Commodities: `{' '.join(sorted(commodities))}`")
    if crypto:
        lines.append(f"Crypto: `{' '.join(sorted(crypto))}`")

    return "\n".join(lines)


def format_usage(summary: dict) -> str:
    """Format token usage summary."""
    lines = [
        "*Token Usage Summary*",
        "",
        f"Period: Last {summary.get('period_days', 30)} days",
        f"Total Calls: `{summary.get('total_calls', 0)}`",
        f"Input Tokens: `{summary.get('total_input_tokens', 0):,}`",
        f"Output Tokens: `{summary.get('total_output_tokens', 0):,}`",
        f"Est. Cost: `${summary.get('total_cost_usd', 0):.4f}`",
        "",
        f"Today Used: `{summary.get('today_tokens_used', 0):,}` tokens",
        f"Today Remaining: `{summary.get('today_budget_remaining', 0):,}` tokens",
    ]
    return "\n".join(lines)


def format_status(
    last_daily: str | None,
    last_weekly: str | None,
    next_daily: str | None,
    next_weekly: str | None,
) -> str:
    """Format system status."""
    lines = [
        "*System Status*",
        "",
        f"Last Daily Run: `{last_daily or 'Never'}`",
        f"Last Weekly Run: `{last_weekly or 'Never'}`",
        f"Next Daily Run: `{next_daily or 'Not scheduled'}`",
        f"Next Weekly Run: `{next_weekly or 'Not scheduled'}`",
    ]
    return "\n".join(lines)


def truncate_for_telegram(text: str, max_len: int = 4096) -> str:
    """Truncate message to Telegram's limit, preserving markdown structure."""
    if len(text) <= max_len:
        return text
    return text[: max_len - 20] + "\n\n... truncated"
