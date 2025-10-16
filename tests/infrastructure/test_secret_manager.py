from __future__ import annotations

import json
from unittest import mock

import pytest

from infrastructure.secret_manager import (
    SecretRetrievalError,
    load_jwt_keyset_from_gcp,
    load_jwt_keyset_from_vault,
)


def test_vault_client_reads_kv(monkeypatch):
    response = mock.MagicMock()
    response.status_code = 200
    response.json.return_value = {"data": {"data": {"current": "kid", "keys": []}}}
    monkeypatch.setenv("VAULT_ADDR", "https://vault")
    monkeypatch.setenv("VAULT_TOKEN", "token")
    monkeypatch.setenv("VAULT_JWT_KEY_PATH", "secret/data/jwt")
    with mock.patch("infrastructure.secret_manager.requests.get", return_value=response):
        payload = load_jwt_keyset_from_vault()
    assert payload["current"] == "kid"


def test_vault_client_error(monkeypatch):
    response = mock.MagicMock()
    response.status_code = 404
    response.text = "not found"
    monkeypatch.setenv("VAULT_ADDR", "https://vault")
    monkeypatch.setenv("VAULT_TOKEN", "token")
    monkeypatch.setenv("VAULT_JWT_KEY_PATH", "secret/data/jwt")
    with mock.patch("infrastructure.secret_manager.requests.get", return_value=response):
        with pytest.raises(SecretRetrievalError):
            load_jwt_keyset_from_vault()


def test_gcp_secret_manager(monkeypatch):
    response = mock.MagicMock()
    response.status_code = 200
    response.json.return_value = {"payload": {"data": json.dumps({"keys": []})}}
    monkeypatch.setenv("GCP_PROJECT_ID", "proj")
    monkeypatch.setenv("GCP_JWT_SECRET_ID", "secret")
    monkeypatch.setenv("GCP_ACCESS_TOKEN", "token")
    with mock.patch("infrastructure.secret_manager.requests.get", return_value=response):
        payload = load_jwt_keyset_from_gcp()
    assert payload["keys"] == []


def test_gcp_secret_manager_error(monkeypatch):
    response = mock.MagicMock()
    response.status_code = 500
    response.text = "error"
    monkeypatch.setenv("GCP_PROJECT_ID", "proj")
    monkeypatch.setenv("GCP_JWT_SECRET_ID", "secret")
    monkeypatch.setenv("GCP_ACCESS_TOKEN", "token")
    with mock.patch("infrastructure.secret_manager.requests.get", return_value=response):
        with pytest.raises(SecretRetrievalError):
            load_jwt_keyset_from_gcp()
