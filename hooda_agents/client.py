"""Small, testable client for Ollama's native chat API."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import requests

from hooda_agents.config import Settings


class OllamaError(RuntimeError):
    """Raised when Ollama is unavailable or returns an invalid response."""


class OllamaClient:
    def __init__(
        self,
        settings: Settings,
        *,
        session: requests.Session | None = None,
    ) -> None:
        self.settings = settings
        self.session = session or requests.Session()

    def chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        tools: Sequence[Mapping[str, Any]] | None = None,
        format_schema: Mapping[str, Any] | str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.settings.model,
            "messages": list(messages),
            "stream": False,
            "think": self.settings.think,
            "options": {
                "temperature": self.settings.temperature,
                "num_ctx": self.settings.context_length,
            },
        }
        if tools:
            payload["tools"] = list(tools)
        if format_schema is not None:
            payload["format"] = format_schema

        url = f"{self.settings.ollama_base_url}/api/chat"
        try:
            response = self.session.post(
                url,
                json=payload,
                timeout=self.settings.request_timeout_seconds,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            detail = ""
            response_obj = getattr(exc, "response", None)
            if response_obj is not None:
                detail = f": {response_obj.text[:500]}"
            raise OllamaError(
                f"Ollama request failed at {url}{detail}. "
                "Confirm that Ollama is running and the configured model is installed."
            ) from exc

        try:
            data = response.json()
        except ValueError as exc:
            raise OllamaError("Ollama returned a non-JSON response") from exc

        message = data.get("message")
        if not isinstance(message, dict):
            raise OllamaError("Ollama response did not contain a message object")

        return message
