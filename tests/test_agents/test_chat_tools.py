"""Tests for chat agent tool configuration."""


import pytest


@pytest.fixture(autouse=True)
def _set_env(monkeypatch):
    """Set required env vars for Settings to load."""
    monkeypatch.setenv("PA_OPENAI_API_KEY", "test")
    monkeypatch.setenv("PA_TELEGRAM_BOT_TOKEN", "test")
    monkeypatch.setenv("PA_TELEGRAM_CHAT_ID", "123")


class TestChatAgentTools:
    """Verify chat agent has the correct number and set of tools."""

    def test_chat_agent_has_39_tools(self):
        """The v4 plan requires exactly 39 tools (33 original + 6 new)."""
        from portfolio_advisor.agents.testing import reset_all_agents
        reset_all_agents()
        from portfolio_advisor.agents.chat import get_chat_agent
        agent = get_chat_agent()
        assert len(agent.tools) == 39, (
            f"Expected 39 tools, got {len(agent.tools)}. "
            f"Tool names: {[getattr(t, 'name', str(t)) for t in agent.tools]}"
        )

    def test_new_v4_tools_present(self):
        """Verify all 6 new v4 tools are in the chat agent."""
        from portfolio_advisor.agents.testing import reset_all_agents
        reset_all_agents()
        from portfolio_advisor.agents.chat import get_chat_agent
        agent = get_chat_agent()

        tool_names = set()
        for t in agent.tools:
            name = getattr(t, "name", None)
            if name is None:
                # Agent-as-tool delegates
                name = getattr(t, "tool_name", None) or str(t)
            tool_names.add(name)

        expected_new_tools = {
            "get_ticker_news",
            "get_fundamentals",
            "get_valuation_comparison",
            "get_analyst_consensus",
            "get_short_interest",
            "get_dividend_info",
        }

        missing = expected_new_tools - tool_names
        assert not missing, f"Missing v4 tools from chat agent: {missing}"


class TestAgentReset:
    """Tests for the agent reset testing utilities."""

    def test_reset_all_agents_does_not_crash(self):
        from portfolio_advisor.agents.testing import reset_all_agents
        reset_all_agents()  # should not raise

    def test_agent_recreated_after_reset(self):
        from portfolio_advisor.agents.testing import reset_all_agents
        reset_all_agents()
        from portfolio_advisor.agents.chat import get_chat_agent

        agent1 = get_chat_agent()
        reset_all_agents()
        agent2 = get_chat_agent()
        # After reset, a new instance should be created
        assert agent1 is not agent2
