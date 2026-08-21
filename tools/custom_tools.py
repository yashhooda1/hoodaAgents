"""Compatibility exports for hoodaAgents tools."""

from hooda_agents.tools import ToolSpec, calculate

simple_calculator = calculate
calculator_tool = ToolSpec(
    name="calculator",
    description="Safely evaluate a bounded arithmetic expression.",
    parameters={
        "type": "object",
        "required": ["expression"],
        "properties": {"expression": {"type": "string"}},
        "additionalProperties": False,
    },
    handler=calculate,
)

__all__ = ["calculator_tool", "simple_calculator"]
