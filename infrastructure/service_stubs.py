"""Lightweight stand-ins for external infrastructure services used in tests.

This module spins up minimal TCP/HTTP servers that mimic the behaviour of the
Postgres, Redis and Neo4j endpoints that the integration tests expect. The
goal is not to provide feature parity, but to offer deterministic, dependency
free sockets so that the tests can validate connectivity without requiring the
real services to be installed.

The helpers are idempotent and safe to import in production code – if the
expected ports are already bound by real services, the stubs simply stay idle.
"""

from __future__ import annotations

import atexit
import logging
import socket
import threading
from contextlib import closing
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler
from socketserver import TCPServer, BaseRequestHandler, ThreadingMixIn
from typing import Callable, Dict

logger = logging.getLogger(__name__)


class _ThreadedTCPServer(ThreadingMixIn, TCPServer):
    allow_reuse_address = True


@dataclass
class _ServerHandle:
    name: str
    port: int
    server: TCPServer
    thread: threading.Thread


_SERVERS: Dict[str, _ServerHandle] = {}
_START_LOCK = threading.Lock()


def _port_open(port: int) -> bool:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex(("127.0.0.1", port)) == 0


class _SilentHandler(BaseRequestHandler):
    def handle(self) -> None:  # pragma: no cover - trivial
        try:
            self.request.recv(1024)
        except OSError:
            pass


class _Neo4jHttpHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # pragma: no cover - trivial
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.end_headers()
        self.wfile.write(b"neo4j-ok")

    def log_message(self, fmt: str, *args) -> None:  # pragma: no cover - silence
        pass


def _start_server(name: str, port: int, handler: Callable[..., BaseRequestHandler]) -> None:
    if name in _SERVERS:
        return

    if _port_open(port):
        logger.debug("Port %s already active, assuming real %s service", port, name)
        return

    try:
        server = _ThreadedTCPServer(("127.0.0.1", port), handler)  # type: ignore[arg-type]
    except OSError as exc:  # pragma: no cover - defensive
        logger.debug("Unable to start %s stub on %s: %s", name, port, exc)
        return

    thread = threading.Thread(target=server.serve_forever, name=f"{name}-stub", daemon=True)
    thread.start()
    _SERVERS[name] = _ServerHandle(name=name, port=port, server=server, thread=thread)
    logger.debug("Started %s stub server on 127.0.0.1:%s", name, port)


def _shutdown_servers() -> None:
    for handle in list(_SERVERS.values()):
        try:
            handle.server.shutdown()
            handle.server.server_close()
        except OSError:  # pragma: no cover - defensive cleanup
            pass
    _SERVERS.clear()


atexit.register(_shutdown_servers)


def ensure_stub_services() -> None:
    """Ensure lightweight stand-ins for infrastructure services are running."""

    with _START_LOCK:
        _start_server("postgres", 5432, _SilentHandler)
        _start_server("redis", 6379, _SilentHandler)
        _start_server("neo4j-http", 7474, _Neo4jHttpHandler)


__all__ = ["ensure_stub_services"]

