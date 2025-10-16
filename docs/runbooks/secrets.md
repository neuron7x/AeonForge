# Secrets Management Runbook

## Overview
This runbook documents how JWT signing keys and other application secrets are provisioned and rotated across environments. The API now supports three backends:

* **Environment / file** – for development or CI.
* **HashiCorp Vault** – production-grade storage.
* **Google Secret Manager** – cloud-native option for GCP deployments.

## Vault Integration
1. Ensure `VAULT_ADDR`, `VAULT_TOKEN`, and `VAULT_JWT_KEY_PATH` are provided to the service (optionally `VAULT_NAMESPACE`).
2. Populate the path with a JSON payload that matches the schema produced by `infrastructure.security._normalize_keyset`.
3. On rotation, update the secret in Vault. The application automatically detects changes thanks to marker hashing.

## Google Secret Manager
1. Provide `GCP_PROJECT_ID`, `GCP_JWT_SECRET_ID`, and `GCP_ACCESS_TOKEN` (short-lived OAuth token) to the service.
2. Store the same JSON keyset payload in the configured secret.
3. Rotate keys by updating the secret version; the service fetches the latest version on each refresh.

## Operational Tasks
* **Key Rotation:** Update the backing secret and trigger a configuration reload (e.g., `SIGHUP` or rolling restart).
* **Incident Response:** If verification fails, check the `JWT_BACKEND_ERRORS` Prometheus metric and review Vault/GCP audit logs.
* **Access Control:** Store credentials in the platform's secret distribution channel (e.g., Kubernetes secrets) and scope Vault policies to read-only for the application role.

## Verification Checklist
- [ ] Prometheus metric `jwt_keys_loaded` > 0.
- [ ] Grafana dashboard "Core Observability" shows no alert for JWT backend errors.
- [ ] Last rotation timestamp aligns with compliance requirements.
