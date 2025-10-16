"""Minimal stub of psycopg2 for the test environment."""

from __future__ import annotations

from typing import Any, Dict, Optional

try:  # pragma: no cover - best effort import
    from infrastructure.service_stubs import ensure_stub_services

    ensure_stub_services()
except Exception:  # pragma: no cover - ignore stub bootstrap issues
    pass


class OperationalError(Exception):
    """Fallback ``OperationalError`` compatible with psycopg2."""


class _FakeCursor:
    def __init__(self) -> None:
        self._last_query: Optional[str] = None

    def execute(self, query: str, params: Optional[Dict[str, Any]] = None) -> "_FakeCursor":
        self._last_query = query
        return self

    def fetchone(self) -> tuple[int]:
        return (1,)

    def close(self) -> None:  # pragma: no cover - trivial
        pass

    def __enter__(self) -> "_FakeCursor":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # pragma: no cover - trivial
        self.close()


class _FakeConnection:
    def __init__(self, dsn: Optional[str] = None, **kwargs: Any) -> None:
        self.dsn = dsn
        self.params = kwargs
        self.closed = False

    def cursor(self) -> _FakeCursor:
        return _FakeCursor()

    def close(self) -> None:
        self.closed = True

    def __enter__(self) -> "_FakeConnection":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def connect(dsn: Optional[str] = None, **kwargs: Any) -> _FakeConnection:
    return _FakeConnection(dsn, **kwargs)


__all__ = ["connect", "OperationalError"]

