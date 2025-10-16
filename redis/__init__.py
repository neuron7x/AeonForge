"""Redis client stub used for tests without external dependencies."""

from __future__ import annotations

from typing import Any, Dict

try:  # pragma: no cover - best effort import
    from infrastructure.service_stubs import ensure_stub_services

    ensure_stub_services()
except Exception:  # pragma: no cover - ignore stub bootstrap issues
    pass


class RedisError(Exception):
    """Fallback RedisError compatible with redis-py."""


class Redis:
    _store: Dict[str, bytes] = {}

    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0, **kwargs: Any) -> None:
        self.host = host
        self.port = port
        self.db = db

    def set(self, key: str, value: Any, ex: int | None = None) -> bool:
        if isinstance(value, str):
            stored = value.encode("utf-8")
        elif isinstance(value, bytes):
            stored = value
        else:
            stored = str(value).encode("utf-8")
        self._store[key] = stored
        return True

    def get(self, key: str) -> bytes | None:
        return self._store.get(key)

    def ping(self) -> bool:
        return True


__all__ = ["Redis", "RedisError"]

