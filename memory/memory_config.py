"""Compatibility helpers for local conversation memory."""

from __future__ import annotations

from pathlib import Path

from hooda_agents.config import Settings
from hooda_agents.memory import ConversationStore


def get_memory(path: str | Path | None = None) -> ConversationStore:
    settings = Settings.from_env()
    return ConversationStore(path or settings.memory_path)
