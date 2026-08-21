"""Compatibility chat entry point backed by the native Ollama agent."""

from __future__ import annotations

from hooda_agents import HoodaAgent, build_agent

_agent: HoodaAgent | None = None


def run_chat(
    query: str,
    history=None,
    *,
    session_id: str = "default",
) -> str:
    """Run one turn; persisted local memory replaces caller-managed history."""

    global _agent
    if _agent is None:
        _agent = build_agent()
    return _agent.run(query, session_id=session_id).text
