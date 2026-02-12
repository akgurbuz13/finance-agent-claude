"""Agent reset utilities for testing."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def reset_all_agents() -> None:
    """Reset all agent singletons. For use in test fixtures."""
    from portfolio_advisor.agents import chat, onboarding, orchestrator, portfolio
    from portfolio_advisor.agents import quantitative, reporting, research, technical

    modules = [chat, onboarding, orchestrator, portfolio, quantitative, reporting, research, technical]

    for mod in modules:
        if hasattr(mod, "_agent"):
            mod._agent = None
        # Some modules have multiple agents
        if hasattr(mod, "_weekly_orchestrator"):
            mod._weekly_orchestrator = None
        if hasattr(mod, "_daily_synthesis_agent"):
            mod._daily_synthesis_agent = None

    logger.info("All agent singletons reset")
