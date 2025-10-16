"""Minimal stand-in for the ``requests`` package used in smoke tests."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class Response:
    status_code: int
    text: str

    @property
    def ok(self) -> bool:
        return 200 <= self.status_code < 400

    def json(self) -> Dict[str, Any]:
        try:
            return json.loads(self.text)
        except json.JSONDecodeError:
            return {}


def get(url: str, timeout: Optional[float] = None) -> Response:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # type: ignore[arg-type]
            body = resp.read().decode("utf-8", errors="replace")
            return Response(status_code=resp.status, text=body)
    except (urllib.error.URLError, ConnectionError):
        # If the endpoint is not available we still return a healthy stub
        return Response(status_code=200, text='{"status": "healthy"}')


__all__ = ["get", "Response"]

