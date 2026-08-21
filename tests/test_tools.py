import os
import unittest
from unittest import mock

from hooda_agents.tools import (
    ToolRegistry,
    ToolSpec,
    build_default_registry,
    calculate,
)


class CalculatorTests(unittest.TestCase):
    def test_calculates_bounded_math(self):
        self.assertEqual("19.0", calculate("sqrt(144) + 7"))
        self.assertEqual("15", calculate("max(3, 15, 8)"))
        self.assertEqual("3.14", calculate("round(pi, 2)"))

    def test_rejects_code_execution_and_attributes(self):
        expressions = [
            "__import__('os').system('id')",
            "(1).__class__",
            "[x for x in range(10)]",
        ]
        for expression in expressions:
            with self.subTest(expression=expression):
                with self.assertRaises(ValueError):
                    calculate(expression)

    def test_rejects_unbounded_work(self):
        with self.assertRaisesRegex(ValueError, "exponent"):
            calculate("2 ** 1000")
        with self.assertRaisesRegex(ValueError, "complex"):
            calculate("+".join(["1"] * 40))


class ToolRegistryTests(unittest.TestCase):
    def test_unknown_tools_return_bounded_error_event(self):
        registry = ToolRegistry()

        event = registry.execute_call(
            {"function": {"name": "shell", "arguments": {"command": "id"}}}
        )

        self.assertEqual("error", event.status)
        self.assertIn("Unknown tool", event.output)

    def test_validates_required_and_unexpected_arguments(self):
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
                handler=lambda text: text,
            )
        )

        missing = registry.execute_call(
            {"function": {"name": "echo", "arguments": {}}}
        )
        unexpected = registry.execute_call(
            {
                "function": {
                    "name": "echo",
                    "arguments": {"text": "ok", "extra": True},
                }
            }
        )

        self.assertEqual("error", missing.status)
        self.assertIn("Missing required", missing.output)
        self.assertEqual("error", unexpected.status)
        self.assertIn("Unexpected arguments", unexpected.output)

    def test_truncates_tool_output(self):
        registry = ToolRegistry(max_output_chars=5)
        registry.register(
            ToolSpec(
                name="long",
                description="long output",
                parameters={"type": "object", "properties": {}},
                handler=lambda: "123456789",
            )
        )

        event = registry.execute_call(
            {"function": {"name": "long", "arguments": {}}}
        )

        self.assertEqual("success", event.status)
        self.assertTrue(event.output.startswith("12345"))
        self.assertIn("truncated", event.output)

    def test_web_search_is_optional(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            registry = build_default_registry()

        self.assertEqual(["calculator", "current_datetime"], registry.names())


if __name__ == "__main__":
    unittest.main()
