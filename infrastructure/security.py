from __future__ import annotations

import json
import os
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt, JWTError

from infrastructure.metrics import (
    JWT_ACTIVE_KEY_AGE,
    JWT_AUTH_FAILURES,
    JWT_BACKEND_ERRORS,
    JWT_KEY_ROTATIONS,
    JWT_KEYS_LOADED,
)
from infrastructure.secret_manager import (
    SecretRetrievalError,
    load_jwt_keyset_from_gcp,
    load_jwt_keyset_from_vault,
)


class SecretsBackendError(RuntimeError):
    """Raised when the JWT secrets backend cannot supply a valid keyset."""


security = HTTPBearer()

# Prime labelled metrics so they are visible before the first event.
JWT_AUTH_FAILURES.labels(reason="invalid_header")
JWT_AUTH_FAILURES.labels(reason="invalid_signature")
JWT_BACKEND_ERRORS.labels(reason="load_failure")

JWT_ALG = os.getenv("JWT_ALG", "HS256")

_keyset_cache: Optional[Tuple[str, Dict[str, Dict[str, Any]]]] = None
_cache_marker: Optional[Any] = None
_active_kid: Optional[str] = None


def _parse_activation_ts(raw: Optional[str]) -> Optional[datetime]:
    if not raw:
        return None
    try:
        # Allow both "Z" and offset aware ISO timestamps.
        normalized = raw.replace("Z", "+00:00") if raw.endswith("Z") else raw
        dt = datetime.fromisoformat(normalized)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except ValueError as exc:
        raise SecretsBackendError(f"Invalid activation timestamp '{raw}'") from exc


def _normalize_keyset(data: Dict[str, Any]) -> Tuple[str, Dict[str, Dict[str, Any]]]:
    keys = data.get("keys")
    if not isinstance(keys, Iterable):
        raise SecretsBackendError("JWT keyset must include an iterable 'keys' entry")

    normalized: Dict[str, Dict[str, Any]] = {}
    for entry in keys:
        if not isinstance(entry, dict):
            raise SecretsBackendError("JWT key entries must be objects")
        kid = entry.get("kid")
        secret = entry.get("secret")
        if not kid or not secret:
            raise SecretsBackendError("JWT key entries require 'kid' and 'secret'")
        normalized[kid] = {
            "secret": secret,
            "activated_at": entry.get("activated_at"),
        }

    if not normalized:
        raise SecretsBackendError("JWT keyset does not contain any keys")

    current = data.get("current") or next(iter(normalized))
    if current not in normalized:
        raise SecretsBackendError(f"Active key '{current}' not present in keyset")

    return current, normalized


def _load_raw_keyset() -> Tuple[Dict[str, Any], Any]:
    backend = os.getenv("JWT_SECRETS_BACKEND", "env").lower()
    if backend == "env":
        raw = os.getenv("JWT_KEYSET_JSON")
        if not raw:
            # Fallback for development environments – use legacy single secret.
            secret = os.getenv("JWT_SECRET", "dev")
            kid = os.getenv("JWT_FALLBACK_KID", "default")
            return (
                {
                    "current": kid,
                    "keys": [
                        {
                            "kid": kid,
                            "secret": secret,
                            "activated_at": datetime.now(timezone.utc).isoformat(),
                        }
                    ],
                },
                (kid, secret),
            )
        try:
            return json.loads(raw), hash(raw)
        except json.JSONDecodeError as exc:
            raise SecretsBackendError("JWT_KEYSET_JSON is not valid JSON") from exc
    if backend == "file":
        path = os.getenv("JWT_KEYSET_PATH")
        if not path:
            raise SecretsBackendError("JWT_KEYSET_PATH is not configured")
        file_path = Path(path)
        if not file_path.exists():
            raise SecretsBackendError(f"JWT keyset file '{file_path}' not found")
        try:
            payload = json.loads(file_path.read_text())
        except json.JSONDecodeError as exc:
            raise SecretsBackendError(f"JWT keyset file '{file_path}' is not valid JSON") from exc
        return payload, (str(file_path), file_path.stat().st_mtime)
    if backend == "vault":
        try:
            payload = load_jwt_keyset_from_vault()
        except SecretRetrievalError as exc:
            raise SecretsBackendError(f"Vault error: {exc}") from exc
        return payload, hash(json.dumps(payload, sort_keys=True))
    if backend == "gcp_secret_manager":
        try:
            payload = load_jwt_keyset_from_gcp()
        except SecretRetrievalError as exc:
            raise SecretsBackendError(f"GCP Secret Manager error: {exc}") from exc
        return payload, hash(json.dumps(payload, sort_keys=True))

    raise SecretsBackendError(f"Unsupported JWT secrets backend '{backend}'")


def _update_metrics(current: str, key_mapping: Dict[str, Dict[str, Any]]) -> None:
    global _active_kid
    JWT_KEYS_LOADED.set(len(key_mapping))

    activated_at = _parse_activation_ts(key_mapping[current].get("activated_at"))
    if activated_at is not None:
        age = max((datetime.now(timezone.utc) - activated_at).total_seconds(), 0.0)
    else:
        age = 0.0
    JWT_ACTIVE_KEY_AGE.labels(kid=current).set(age)

    if _active_kid is None:
        _active_kid = current
    elif current != _active_kid:
        JWT_KEY_ROTATIONS.labels(kid=current).inc()
        _active_kid = current


def _get_keyset() -> Tuple[str, Dict[str, Dict[str, Any]]]:
    global _keyset_cache, _cache_marker
    payload, marker = _load_raw_keyset()
    if _keyset_cache is not None and _cache_marker == marker:
        current, mapping = _keyset_cache
        _update_metrics(current, mapping)
        return _keyset_cache

    current, mapping = _normalize_keyset(payload)
    _keyset_cache = (current, mapping)
    _cache_marker = marker
    _update_metrics(current, mapping)
    return _keyset_cache


def _candidate_kids(current: str, mapping: Dict[str, Dict[str, Any]], token_kid: Optional[str]) -> Iterable[str]:
    seen = set()
    if token_kid and token_kid in mapping:
        seen.add(token_kid)
        yield token_kid
    if current not in seen:
        seen.add(current)
        yield current
    for kid in mapping:
        if kid not in seen:
            yield kid


def verify_jwt(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:
    token = credentials.credentials
    try:
        current, mapping = _get_keyset()
    except SecretsBackendError as exc:
        JWT_BACKEND_ERRORS.labels(reason="load_failure").inc()
        raise HTTPException(status_code=503, detail="JWT secrets backend unavailable") from exc

    try:
        header = jwt.get_unverified_header(token)
        kid = header.get("kid")
    except JWTError as exc:
        JWT_AUTH_FAILURES.labels(reason="invalid_header").inc()
        raise HTTPException(status_code=401, detail=f"Invalid token header: {exc}") from exc

    errors = []
    for candidate in _candidate_kids(current, mapping, kid):
        secret = mapping[candidate]["secret"]
        try:
            return jwt.decode(token, secret, algorithms=[JWT_ALG])
        except JWTError as exc:  # try next key
            errors.append(str(exc))

    JWT_AUTH_FAILURES.labels(reason="invalid_signature").inc()
    detail = "; ".join(errors[-2:]) if errors else "signature verification failed"
    raise HTTPException(status_code=401, detail=f"Invalid token: {detail}")
