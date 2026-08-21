import unittest

from hooda_agents.agent import (
    AgentLoopLimitError,
    HoodaAgent,
)
from hooda_agents.config import Settings
from hooda_agents.memory import ConversationStore, NullConversationStore
from hooda_agents.tools import ToolRegistry, ToolSpec, calculate


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def chat(self, messages, *, tools=None, format_schema=None):
        self.calls.append(
            {
                "messages": [dict(message) for message in messages],
                "tools": tools,
                "format_schema": format_schema,
            }
        )
        if not self.responses:
            raise AssertionError("unexpected client call")
        return self.responses.pop(0)


def calculator_registry():
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="calculator",
            description="calculate",
            parameters={
                "type": "object",
                "required": ["expression"],
                "properties": {"expression": {"type": "string"}},
            },
            handler=calculate,
        )
    )
    return registry


class AgentLoopTests(unittest.TestCase):
    def test_executes_tool_then_returns_final_answer(self):
        client = FakeClient(
            [
                {
                    "content": "",
                    "thinking": "I should calculate this.",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "calculator",
                                "arguments": {"expression": "sqrt(144) + 7"},
                            }
                        }
                    ],
                },
                {"content": "The answer is 19.", "tool_calls": []},
            ]
        )
        agent = HoodaAgent(
            Settings(memory_enabled=False),
            client,
            calculator_registry(),
            NullConversationStore(),
        )

        result = agent.run("Calculate sqrt(144) + 7")

        self.assertEqual("The answer is 19.", result.text)
        self.assertEqual(2, result.steps)
        self.assertEqual("calculator", result.tool_events[0].name)
        self.assertEqual("success", result.tool_events[0].status)
        self.assertEqual(("I should calculate this.",), result.thinking)
        tool_message = client.calls[1]["messages"][-1]
        self.assertEqual("tool", tool_message["role"])
        self.assertEqual("calculator", tool_message["tool_name"])
        self.assertIn("19.0", tool_message["content"])

    def test_executes_parallel_tool_calls(self):
        registry = ToolRegistry()
        registry.register(
            ToolSpec(
                name="echo",
                description="echo",
                parameters={
                    "type": "object",
                    "required": ["text"],
                    "properties": {"text": {"type": "string"}},
                },
                handler=lambda text: text.upper(),
            )
        )
        client = FakeClient(
            [
                {
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "echo",
                                "arguments": {"text": "first"},
                            }
                        },
                        {
                            "function": {
                                "name": "echo",
                                "arguments": {"text": "second"},
                            }
                        },
                    ],
                },
                {"content": "Done"},
            ]
        )
        agent = HoodaAgent(
            Settings(memory_enabled=False),
            client,
            registry,
            NullConversationStore(),
        )

        result = agent.run("Echo both values")

        self.assertEqual(["FIRST", "SECOND"], [
            event.output for event in result.tool_events
        ])
        tool_messages = [
            message
            for message in client.calls[1]["messages"]
            if message["role"] == "tool"
        ]
        self.assertEqual(2, len(tool_messages))

    def test_enforces_agent_loop_limit(self):
        tool_call = {
            "content": "",
            "tool_calls": [
                {
                    "function": {
                        "name": "calculator",
                        "arguments": {"expression": "1 + 1"},
                    }
                }
            ],
        }
        client = FakeClient([tool_call, tool_call])
        agent = HoodaAgent(
            Settings(max_steps=2, memory_enabled=False),
            client,
            calculator_registry(),
            NullConversationStore(),
        )

        with self.assertRaisesRegex(AgentLoopLimitError, "2-step"):
            agent.run("Never stop")

    def test_persists_only_final_conversation_exchanges(self):
        memory = ConversationStore(":memory:")
        client = FakeClient(
            [
                {"content": "First answer"},
                {"content": "Second answer"},
            ]
        )
        agent = HoodaAgent(
            Settings(memory_enabled=True, history_messages=10),
            client,
            ToolRegistry(),
            memory,
        )

        agent.run("First question", session_id="test")
        agent.run("Second question", session_id="test")

        second_messages = client.calls[1]["messages"]
        self.assertEqual(
            ["system", "user", "assistant", "user"],
            [message["role"] for message in second_messages],
        )
        self.assertEqual("First answer", second_messages[2]["content"])

    def test_requests_schema_constrained_json(self):
        schema = {
            "type": "object",
            "required": ["answer"],
            "properties": {"answer": {"type": "integer"}},
        }
        client = FakeClient([{"content": '{"answer": 42}'}])
        agent = HoodaAgent(
            Settings(memory_enabled=False),
            client,
            ToolRegistry(),
            NullConversationStore(),
        )

        result = agent.complete_json("Return the answer", schema)

        self.assertEqual({"answer": 42}, result)
        self.assertEqual(schema, client.calls[0]["format_schema"])


if __name__ == "__main__":
    unittest.main()
