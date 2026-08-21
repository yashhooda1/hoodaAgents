"""Small, dependency-free client for Ollama's native chat API."""

from __future__ import annotations

import json
from typing import Any, Callable, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from hooda_agents.config import Settings


class HTTPTransportError(RuntimeError):
    """Raised when a JSON HTTP request fails."""


class OllamaError(RuntimeError):
    """Raised when Ollama is unavailable or returns an invalid response."""


JSONTransport = Callable[[str, Mapping[str, Any], float], dict[str, Any]]


def post_json(
    url: str,
    payload: Mapping[str, Any],
    timeout: float,
) -> dict[str, Any]:
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            raw_body = response.read()
    except HTTPError as exc:
        try:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
        except OSError:
            detail = str(exc)
        raise HTTPTransportError(f"HTTP {exc.code}: {detail}") from exc
    except (URLError, OSError) as exc:
        raise HTTPTransportError(str(exc)) from exc

    try:
        data = json.loads(raw_body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HTTPTransportError("server returned invalid JSON") from exc

    if not isinstance(data, dict):
        raise HTTPTransportError("server returned a non-object JSON response")
    return data


class OllamaClient:
    def __init__(
        self,
        settings: Settings,
        *,
        transport: JSONTransport = post_json,
    ) -> None:
        self.settings = settings
        self.transport = transport

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
            data = self.transport(
                url,
                payload,
                self.settings.request_timeout_seconds,
            )
        except HTTPTransportError as exc:
            raise OllamaError(
                f"Ollama request failed at {url}: {exc}. "
                "Confirm that Ollama is running and the configured model is installed."
            ) from exc

        message = data.get("message")
        if not isinstance(message, dict):
            raise OllamaError("Ollama response did not contain a message object")

        return message
