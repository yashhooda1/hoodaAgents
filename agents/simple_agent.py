"""Compatibility wrapper for the native hoodaAgents runtime."""

from __future__ import annotations

from hooda_agents import HoodaAgent, build_agent

_agent: HoodaAgent | None = None


def create_agent() -> HoodaAgent:
    return build_agent()


def run_agent(user_text: str, *, session_id: str = "default") -> str:
    global _agent
    if _agent is None:
        _agent = create_agent()
    return _agent.run(user_text, session_id=session_id).text
