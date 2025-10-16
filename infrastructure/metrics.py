from __future__ import annotations

from prometheus_client import Counter, Histogram, Gauge

REQUESTS_TOTAL = Counter('cbc_requests_total', 'Total API requests', ['endpoint', 'method'])
REQUEST_DURATION = Histogram('cbc_request_duration_seconds', 'Request duration', ['endpoint'])
ACTIVE_USERS = Gauge('cbc_active_users', 'Number of active users')
EOI_SCORE = Gauge('cbc_eoi_score', 'Current eOI score', ['user_id'])
DELEGATION_ACTIONS = Counter('cbc_delegation_actions_total', 'Delegation actions', ['task_type', 'autonomy_level'])
JWT_KEYS_LOADED = Gauge('cbc_jwt_keys_loaded', 'Number of JWT verification keys currently cached')
JWT_ACTIVE_KEY_AGE = Gauge('cbc_jwt_active_key_age_seconds', 'Age of the active JWT key in seconds', ['kid'])
JWT_KEY_ROTATIONS = Counter('cbc_jwt_key_rotations_total', 'Count of active JWT key rotations observed', ['kid'])
JWT_AUTH_FAILURES = Counter('cbc_jwt_auth_failures_total', 'Total JWT authentication failures', ['reason'])
JWT_BACKEND_ERRORS = Counter('cbc_jwt_backend_errors_total', 'Errors encountered when loading JWT secrets', ['reason'])
