# JWT Secrets Rotation Runbook

This runbook documents the operational response for the JWT telemetry and alerts introduced for the CBC-Ω² API.

## Related Metrics

| Metric | Description | Expected Range |
| --- | --- | --- |
| `cbc_jwt_keys_loaded` | Number of JWT verification keys currently cached by the API. | ≥ 1 |
| `cbc_jwt_active_key_age_seconds` | Age of the active JWT key. | < 604,800 seconds (7 days) |
| `cbc_jwt_key_rotations_total` | Count of observed active key rotations. | Monotonic counter |
| `cbc_jwt_auth_failures_total` | JWT validation failures labelled by reason. | Should remain near 0 |
| `cbc_jwt_backend_errors_total` | Errors loading key material from the secrets backend. | Always 0 |

## Alert Playbooks

### `JWTSecretsBackendErrors`
1. **Triage**: Confirm recent deployments or infrastructure work affecting the secrets backend (e.g., Vault outages, missing Kubernetes secret).
2. **Immediate Action**: Restart a single API instance to attempt a fresh fetch. If the issue persists, roll back the last change that touched the secrets integration.
3. **Escalation**: Page the platform team if the backend remains unavailable for more than 10 minutes.
4. **Postmortem Data**: Capture `/metrics` output for `cbc_jwt_backend_errors_total` and any related logs.

### `JWTAuthFailureSpike`
1. **Triage**: Inspect application logs for repeated `401` responses and identify the offending client IDs.
2. **Immediate Action**: Invalidate tokens issued before the spike via the identity provider. Ensure the most recent signing key is active.
3. **Remediation**: Communicate with affected integrators, providing them with the current JWKS metadata.
4. **Follow-up**: Consider accelerating key rotation if the spike continues.

### `JWTKeyAgeExceeded`
1. **Triage**: Check `cbc_jwt_active_key_age_seconds` to confirm the key exceeded the 7-day SLA.
2. **Immediate Action**: Generate a new signing key in the secrets backend, update the `current` pointer, and ensure old keys remain for at least 24 hours for validation.
3. **Validation**: Confirm the counter `cbc_jwt_key_rotations_total` increments after the new key becomes active.
4. **Documentation**: Log the rotation time in the security change calendar.

## Verification Checklist After Rotation

- [ ] `/metrics` reports a new `kid` label on `cbc_jwt_active_key_age_seconds` with an age close to `0` seconds.
- [ ] `cbc_jwt_keys_loaded` reflects the updated number of keys.
- [ ] Legacy tokens continue to validate until their expiry.
- [ ] `cbc_jwt_backend_errors_total` remains `0`.

## Useful Commands

```bash
# Dump metrics locally
curl -s localhost:8000/metrics | grep "cbc_jwt"

# Validate keyset JSON schema
python -m json.tool <(kubectl get secret cbc-secrets -o jsonpath='{.data.jwt-keyset}' | base64 --decode)
```

## Escalation Contacts

- **On-call Platform Engineer** – Slack `#oncall-platform`
- **Security Operations** – PagerDuty schedule `secops-primary`
