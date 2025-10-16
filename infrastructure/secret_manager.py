from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict

import requests


class SecretRetrievalError(RuntimeError):
    pass


@dataclass
class VaultClient:
    address: str
    token: str
    namespace: str | None = None

    def _headers(self) -> Dict[str, str]:
        headers = {"X-Vault-Token": self.token}
        if self.namespace:
            headers["X-Vault-Namespace"] = self.namespace
        return headers

    def read(self, path: str) -> Dict[str, Any]:
        url = os.path.join(self.address.rstrip("/"), "v1", path.lstrip("/"))
        response = requests.get(url, headers=self._headers(), timeout=10)
        if response.status_code != 200:
            raise SecretRetrievalError(f"Vault returned {response.status_code}: {response.text}")
        payload = response.json()
        return payload.get("data", {}).get("data", payload.get("data", {}))


@dataclass
class GCPSecretManagerClient:
    project_id: str
    access_token: str

    def read(self, secret_id: str, version: str = "latest") -> str:
        url = (
            "https://secretmanager.googleapis.com/v1/"
            f"projects/{self.project_id}/secrets/{secret_id}/versions/{version}:access"
        )
        headers = {"Authorization": f"Bearer {self.access_token}"}
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            raise SecretRetrievalError(f"GCP Secret Manager returned {response.status_code}: {response.text}")
        data = response.json()
        return data["payload"]["data"]


def load_jwt_keyset_from_vault() -> Dict[str, Any]:
    address = os.getenv("VAULT_ADDR")
    token = os.getenv("VAULT_TOKEN")
    path = os.getenv("VAULT_JWT_KEY_PATH")
    namespace = os.getenv("VAULT_NAMESPACE")
    if not all([address, token, path]):
        raise SecretRetrievalError("Vault configuration incomplete")
    client = VaultClient(address=address, token=token, namespace=namespace)
    return client.read(path)


def load_jwt_keyset_from_gcp() -> Dict[str, Any]:
    project_id = os.getenv("GCP_PROJECT_ID")
    secret_id = os.getenv("GCP_JWT_SECRET_ID")
    access_token = os.getenv("GCP_ACCESS_TOKEN")
    if not all([project_id, secret_id, access_token]):
        raise SecretRetrievalError("GCP Secret Manager configuration incomplete")
    client = GCPSecretManagerClient(project_id=project_id, access_token=access_token)
    secret = client.read(secret_id)
    try:
        return json.loads(secret)
    except json.JSONDecodeError as exc:  # pragma: no cover - depends on remote
        raise SecretRetrievalError("Secret payload is not valid JSON") from exc
