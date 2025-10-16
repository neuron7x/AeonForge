# CBC-Ω² LITE Formalization

This note aligns the LITE formalization with the code paths deployed in the
repository.  The production API is implemented in
[`infrastructure/api.py`](../infrastructure/api.py) and depends on the biometric
and planning engines in [`core/biometric_engine.py`](../core/biometric_engine.py)
and [`core/pomdp_planner.py`](../core/pomdp_planner.py).  Behaviour is validated by
[`tests/test_biometric.py`](../tests/test_biometric.py) and
[`tests/test_pomdp.py`](../tests/test_pomdp.py).

## 1. Conceptual model

### 1.1 State definition
- **System state (`Σ`)** tracks biometric baselines, per-user beliefs, and
  planner statistics maintained inside the `CBCSystem` object that is created on
  application startup.【F:infrastructure/api.py†L59-L110】
- **Inputs (`I`)** are typed FastAPI models: `BiometricInput` captures HRV and
  sleep telemetry, while `TaskRequest` captures delegation context, matching the
  numeric ranges expected by the biometric engine.【F:infrastructure/api.py†L18-L58】
- **Outputs (`O`)** include JSON payloads returned by FastAPI endpoints and
  Prometheus metrics exposed at `/metrics` for observability.【F:infrastructure/api.py†L112-L175】
- **Transition (`δ`)** is realised through `BiometricEngine.add_sample` and
  `POMDPPlanner.update_belief`, which update baselines and beliefs after each
  request.【F:core/biometric_engine.py†L67-L132】【F:core/pomdp_planner.py†L175-L233】
- **Emission (`λ`)** covers the delegation plan produced by
  `POMDPPlanner.plan` along with reasoning strings assembled by the API layer to
  surface the current eOI band.【F:infrastructure/api.py†L177-L247】
- **Utility (`U`)** combines biometric overload risk (`eoi`), fatigue, and AI
  autonomy from the optimal action.  The categories map to engagement overload
  thresholds that can be recalibrated from ROC analysis via
  `BiometricEngine.calibrate_thresholds`.【F:infrastructure/api.py†L205-L238】【F:core/biometric_engine.py†L175-L237】

### 1.2 Architecture overview

- **Ingress & routing:** Kubernetes ingress routes `api.cbc-omega-squared.local`
  traffic to the `cbc-api` deployment, as defined in
  [`deploy/k8s/05-ingress.yaml`](../deploy/k8s/05-ingress.yaml) and
  [`deploy/k8s/04-service.yaml`](../deploy/k8s/04-service.yaml).  Two FastAPI
  replicas run behind the ClusterIP service per
  [`deploy/k8s/03-deployment.yaml`](../deploy/k8s/03-deployment.yaml).
- **Control plane:** Each pod initialises `CBCSystem`, which wires the
  biometric and POMDP engines, caches user beliefs, and keeps aggregate
  delegation counters.【F:infrastructure/api.py†L59-L110】
- **Configuration:** Runtime knobs (`NUM_PARTICLES`, `NUM_SIMULATIONS`,
  `MAX_DEPTH`) arrive through the ConfigMap, while JWT secrets flow through the
  Kubernetes Secret referenced in the deployment manifest.【F:deploy/k8s/03-deployment.yaml†L28-L60】
- **Observability:** Prometheus annotations on the pod expose request counters,
  latency histograms, active-user gauges, and delegation counters defined in
  [`infrastructure/metrics.py`](../infrastructure/metrics.py).【F:deploy/k8s/03-deployment.yaml†L16-L24】【F:infrastructure/metrics.py†L1-L8】
- **Security:** `verify_jwt` guards every write/read endpoint, ensuring biometrics
  and delegation plans are only returned to authenticated callers.【F:infrastructure/api.py†L122-L245】

## 2. System implementation

### 2.1 Biometric calibration

`BiometricEngine` maintains rolling baselines and computes the engagement
overload index (`eOI`) using robust z-scores, logistic regression, and PCA
features.  The engine exposes utilities to train models, optimise weights, and
calibrate thresholds from labelled datasets.【F:core/biometric_engine.py†L1-L237】
Tests confirm baseline pruning, category assignment, and ROC-based calibration
behave as expected under synthetic stress scenarios.【F:tests/test_biometric.py†L1-L53】

### 2.2 Delegation planning

`POMDPPlanner` approximates delegation policies with particle filtering and
Monte-Carlo tree search.  The API stores per-user beliefs and updates them with
observations returned after each task completion before planning the next action
with bounded depth search.【F:core/pomdp_planner.py†L175-L320】【F:infrastructure/api.py†L177-L245】
`tests/test_pomdp.py` validates that beliefs and policies remain within expected
bounds across initialise-plan-update cycles.【F:tests/test_pomdp.py†L1-L11】

### 2.3 FastAPI surface

The REST surface offers five endpoints:

| Endpoint | Purpose | Backing logic |
| --- | --- | --- |
| `GET /health` | Liveness probe for Kubernetes and smoke tests.【F:infrastructure/api.py†L112-L116】 | Returns service metadata without auth. |
| `GET /metrics` | Prometheus scrape target.【F:infrastructure/api.py†L118-L121】 | Delegates to `prometheus_client.generate_latest`. |
| `POST /biometric/submit` | Ingests biometric samples, recomputes eOI, updates baselines.【F:infrastructure/api.py†L123-L174】 | Calls `BiometricEngine.add_sample` and `compute_eoi`. |
| `POST /delegate/plan` | Produces delegation guidance, reasoning, and completion-time estimates.【F:infrastructure/api.py†L176-L238】 | Uses stored beliefs, updates with observations, and invokes `POMDPPlanner.plan`. |
| `GET /system/status` | Returns aggregate health metrics for dashboards.【F:infrastructure/api.py†L240-L247】 | Aggregates counts from `CBCSystem`. |

### 2.4 Continuous verification

Running `make test` executes the biometric, POMDP, and NeuroCrowd suites,
mirroring CI.  These tests collectively assert the mathematical assumptions of
the formal model and the contract of the FastAPI surface.【F:tests/test_biometric.py†L1-L53】【F:tests/test_pomdp.py†L1-L11】【F:tests/test_neurocrowd.py†L1-L360】

## 3. Validation checklist

- [x] Biometric baselines recomputed on every sample ingest (`tests/test_biometric.py`).
- [x] Delegation plan bounded between 0 and 1 autonomy (`tests/test_pomdp.py`).
- [x] Memory logging validated for high-eOI routing (`tests/test_neurocrowd.py`).
- [x] Kubernetes manifests kept in sync with API requirements (`deploy/k8s/*.yaml`).
