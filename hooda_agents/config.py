"""Runtime configuration for the native Ollama agent."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _positive_int(name: str, default: int) -> int:
    raw = os.getenv(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value


def _positive_float(name: str, default: float) -> float:
    raw = os.getenv(name, str(default))
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value


def _think_setting(raw: str) -> bool | str:
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    if normalized in {"low", "medium", "high", "max"}:
        return normalized
    raise ValueError(
        "HOODA_THINK must be true, false, low, medium, high, or max"
    )


@dataclass(frozen=True)
class Settings:
    """Validated settings loaded from environment variables."""

    ollama_base_url: str = "http://localhost:11434"
    model: str = "hoodarunner/hoodaAgents"
    temperature: float = 0.2
    context_length: int = 32768
    think: bool | str = True
    max_steps: int = 8
    history_messages: int = 24
    request_timeout_seconds: float = 180.0
    memory_path: Path = Path("~/.hoodaagents/memory.db")
    memory_enabled: bool = True
    max_prompt_chars: int = 20000
    max_tool_output_chars: int = 12000

    @classmethod
    def from_env(cls) -> "Settings":
        memory_enabled = os.getenv("HOODA_MEMORY", "on").strip().lower()
        if memory_enabled not in {"on", "off"}:
            raise ValueError("HOODA_MEMORY must be 'on' or 'off'")

        return cls(
            ollama_base_url=os.getenv(
                "OLLAMA_BASE_URL", "http://localhost:11434"
            ).rstrip("/"),
            model=os.getenv("OLLAMA_MODEL", "hoodarunner/hoodaAgents"),
            temperature=float(os.getenv("OLLAMA_TEMPERATURE", "0.2")),
            context_length=_positive_int("OLLAMA_CONTEXT_LENGTH", 32768),
            think=_think_setting(os.getenv("HOODA_THINK", "true")),
            max_steps=_positive_int("HOODA_MAX_STEPS", 8),
            history_messages=_positive_int("HOODA_HISTORY_MESSAGES", 24),
            request_timeout_seconds=_positive_float(
                "OLLAMA_TIMEOUT_SECONDS", 180.0
            ),
            memory_path=Path(
                os.getenv("HOODA_MEMORY_PATH", "~/.hoodaagents/memory.db")
            ).expanduser(),
            memory_enabled=memory_enabled == "on",
            max_prompt_chars=_positive_int("HOODA_MAX_PROMPT_CHARS", 20000),
            max_tool_output_chars=_positive_int(
                "HOODA_MAX_TOOL_OUTPUT_CHARS", 12000
            ),
        )
