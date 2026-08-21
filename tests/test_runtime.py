import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from hooda_agents.client import (
    HTTPTransportError,
    OllamaClient,
    OllamaError,
)
from hooda_agents.config import Settings
from hooda_agents.memory import ConversationStore


class FakeTransport:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def __call__(self, url, payload, timeout):
        self.calls.append({"url": url, "payload": payload, "timeout": timeout})
        return self.response


class ConfigurationTests(unittest.TestCase):
    def test_loads_validated_environment(self):
        environment = {
            "OLLAMA_BASE_URL": "http://ollama:11434/",
            "OLLAMA_MODEL": "qwen3.5:9b",
            "OLLAMA_CONTEXT_LENGTH": "16384",
            "HOODA_THINK": "high",
            "HOODA_MEMORY": "off",
        }
        with mock.patch.dict(os.environ, environment, clear=True):
            settings = Settings.from_env()

        self.assertEqual("http://ollama:11434", settings.ollama_base_url)
        self.assertEqual("qwen3.5:9b", settings.model)
        self.assertEqual(16384, settings.context_length)
        self.assertEqual("high", settings.think)
        self.assertFalse(settings.memory_enabled)

    def test_rejects_invalid_thinking_mode(self):
        with mock.patch.dict(
            os.environ,
            {"HOODA_THINK": "sometimes"},
            clear=True,
        ):
            with self.assertRaisesRegex(ValueError, "HOODA_THINK"):
                Settings.from_env()


class OllamaClientTests(unittest.TestCase):
    def test_sends_native_agent_payload(self):
        transport = FakeTransport({"message": {"content": "hello"}})
        settings = Settings(
            ollama_base_url="http://localhost:11434",
            model="test-model",
            think=True,
            context_length=8192,
            request_timeout_seconds=12,
        )
        client = OllamaClient(settings, transport=transport)
        tools = [{"type": "function", "function": {"name": "calculator"}}]

        message = client.chat(
            [{"role": "user", "content": "hi"}],
            tools=tools,
        )

        self.assertEqual({"content": "hello"}, message)
        request = transport.calls[0]
        self.assertEqual(
            "http://localhost:11434/api/chat",
            request["url"],
        )
        self.assertEqual("test-model", request["payload"]["model"])
        self.assertEqual(tools, request["payload"]["tools"])
        self.assertTrue(request["payload"]["think"])
        self.assertEqual(8192, request["payload"]["options"]["num_ctx"])
        self.assertEqual(12, request["timeout"])

    def test_wraps_connection_errors_with_actionable_message(self):
        def failing_transport(url, payload, timeout):
            raise HTTPTransportError("connection refused")

        client = OllamaClient(Settings(), transport=failing_transport)

        with self.assertRaisesRegex(OllamaError, "Confirm that Ollama is running"):
            client.chat([{"role": "user", "content": "hi"}])


class MemoryTests(unittest.TestCase):
    def test_load_is_bounded_ordered_and_keeps_complete_turns(self):
        with TemporaryDirectory() as tmp:
            store = ConversationStore(Path(tmp) / "memory.db")
            store.append_exchange("s", "u1", "a1")
            store.append_exchange("s", "u2", "a2")

            messages = store.load("s", 3)

            self.assertEqual(
                [
                    {"role": "user", "content": "u2"},
                    {"role": "assistant", "content": "a2"},
                ],
                messages,
            )

    def test_clear_is_scoped_to_one_session(self):
        store = ConversationStore(":memory:")
        store.append_exchange("one", "u1", "a1")
        store.append_exchange("two", "u2", "a2")

        store.clear("one")

        self.assertEqual([], store.load("one", 10))
        self.assertEqual(2, len(store.load("two", 10)))


if __name__ == "__main__":
    unittest.main()
