"""Local, bounded conversation memory backed by SQLite."""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Protocol


class MemoryStore(Protocol):
    def load(self, session_id: str, limit: int) -> list[dict[str, str]]: ...

    def append_exchange(
        self,
        session_id: str,
        user_content: str,
        assistant_content: str,
    ) -> None: ...

    def clear(self, session_id: str) -> None: ...


def _validate_session_id(session_id: str) -> str:
    normalized = session_id.strip()
    if not normalized:
        raise ValueError("session_id cannot be empty")
    if len(normalized) > 128:
        raise ValueError("session_id cannot exceed 128 characters")
    return normalized


class ConversationStore:
    """Thread-safe local memory that stores only final user/assistant exchanges."""

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).expanduser().parent.mkdir(parents=True, exist_ok=True)
            self.path = str(Path(self.path).expanduser())

        self._lock = threading.RLock()
        self._connection = sqlite3.connect(self.path, check_same_thread=False)
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self._connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_messages_session_id
            ON messages(session_id, id)
            """
        )
        self._connection.commit()

    def load(self, session_id: str, limit: int) -> list[dict[str, str]]:
        session_id = _validate_session_id(session_id)
        if limit <= 0:
            return []

        with self._lock:
            rows = self._connection.execute(
                """
                SELECT role, content
                FROM (
                    SELECT id, role, content
                    FROM messages
                    WHERE session_id = ?
                    ORDER BY id DESC
                    LIMIT ?
                )
                ORDER BY id ASC
                """,
                (session_id, limit),
            ).fetchall()

        return [{"role": role, "content": content} for role, content in rows]

    def append_exchange(
        self,
        session_id: str,
        user_content: str,
        assistant_content: str,
    ) -> None:
        session_id = _validate_session_id(session_id)
        with self._lock, self._connection:
            self._connection.executemany(
                """
                INSERT INTO messages(session_id, role, content)
                VALUES (?, ?, ?)
                """,
                [
                    (session_id, "user", user_content),
                    (session_id, "assistant", assistant_content),
                ],
            )

    def clear(self, session_id: str) -> None:
        session_id = _validate_session_id(session_id)
        with self._lock, self._connection:
            self._connection.execute(
                "DELETE FROM messages WHERE session_id = ?",
                (session_id,),
            )

    def close(self) -> None:
        with self._lock:
            self._connection.close()


class NullConversationStore:
    """Memory implementation used when local persistence is disabled."""

    def load(self, session_id: str, limit: int) -> list[dict[str, str]]:
        _validate_session_id(session_id)
        return []

    def append_exchange(
        self,
        session_id: str,
        user_content: str,
        assistant_content: str,
    ) -> None:
        _validate_session_id(session_id)

    def clear(self, session_id: str) -> None:
        _validate_session_id(session_id)
