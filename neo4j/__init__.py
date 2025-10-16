"""Neo4j driver stub for the local test environment."""

from __future__ import annotations

try:  # pragma: no cover - best effort import
    from infrastructure.service_stubs import ensure_stub_services

    ensure_stub_services()
except Exception:  # pragma: no cover - ignore stub bootstrap issues
    pass


class _FakeResult:
    def __init__(self, record: dict[str, int]) -> None:
        self._record = record

    def single(self) -> dict[str, int]:
        return self._record


class _FakeSession:
    def run(self, query: str, parameters=None) -> _FakeResult:
        return _FakeResult({"ok": 1})

    def close(self) -> None:  # pragma: no cover - trivial
        pass

    def __enter__(self) -> "_FakeSession":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class _FakeDriver:
    def __init__(self, uri: str, auth=None, **kwargs) -> None:
        self.uri = uri
        self.auth = auth

    def session(self, **kwargs) -> _FakeSession:
        return _FakeSession()

    def close(self) -> None:  # pragma: no cover - trivial
        pass

    def __enter__(self) -> "_FakeDriver":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class GraphDatabase:
    @staticmethod
    def driver(uri: str, auth=None, **kwargs) -> _FakeDriver:
        return _FakeDriver(uri, auth, **kwargs)


__all__ = ["GraphDatabase"]

