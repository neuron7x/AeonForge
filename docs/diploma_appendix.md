# Diploma Appendix — Reproducibility Checklist

This appendix captures the steps required to reproduce the experiments and
validation workflow referenced throughout the thesis.  The checklist mirrors the
current `main` branch of this repository.

## 1. Environment setup

- [ ] **Clone the repository**
  - `git clone https://github.com/<org>/aeonforge.git`
  - `cd aeonforge`
- [ ] **Python toolchain**
  - Use Python 3.10+ (tested with 3.11 via GitHub Actions runners).
  - Create an isolated environment: `python -m venv .venv && source .venv/bin/activate`.
- [ ] **Install dependencies**
  - Core + QA extras: `pip install -e .[qa]` (declared in `pyproject.toml`).
  - System packages (required for diagram exports and FastAPI telemetry) are
    listed in `requirements-dev.txt` and installed automatically by the command
    above.
- [ ] **Optional containers**
  - Build the deployment image via `make docker` (uses `deploy/Dockerfile`).

## 2. Configuration & secrets

- [ ] **Kubernetes manifests**
  - Apply ConfigMap and Secret templates under `deploy/k8s/` to provide
    `NUM_PARTICLES`, `NUM_SIMULATIONS`, `MAX_DEPTH`, and `JWT_SECRET` before
    applying `03-deployment.yaml`.【F:deploy/k8s/03-deployment.yaml†L28-L60】
- [ ] **Local development**
  - Export `JWT_SECRET` when running `uvicorn infrastructure.api:app` or `make run`
    to satisfy the authentication dependency.【F:infrastructure/api.py†L123-L201】

## 3. Datasets & synthetic generation

- [ ] **Biometric baselines**
  - Synthetic samples are generated on the fly inside
    `tests/test_biometric.py` using `BiometricSample`; no external dataset is
    required.【F:tests/test_biometric.py†L1-L53】
  - To mirror thesis plots, collect telemetry via `/biometric/submit` and export
    the resulting Prometheus time-series from `/metrics` (scraped by Prometheus).
- [ ] **Delegation traces**
  - Particle beliefs and policies are produced by `POMDPPlanner` during
    `tests/test_pomdp.py` and can be saved by running
    `pytest tests/test_pomdp.py -q` with `PYTEST_ADDOPTS="--durations=0"` for
    timing diagnostics.【F:tests/test_pomdp.py†L1-L11】
- [ ] **NeuroCrowd iterations**
  - NeuroCrowd replay data is emitted via the injected memory protocol in
    `tests/test_neurocrowd.py`; configure a semantic memory adapter that writes to
    disk if longitudinal logs are needed.【F:tests/test_neurocrowd.py†L123-L165】

## 4. Validation steps

- [ ] **Unit & integration tests**
  - `pytest` (full suite) — covers biometric calibration, POMDP policies, and
    NeuroCrowd routing.【F:tests/test_biometric.py†L1-L53】【F:tests/test_pomdp.py†L1-L11】【F:tests/test_neurocrowd.py†L1-L360】
- [ ] **Static analysis**
  - `ruff check .` and `mypy` (included in `make lint`) uphold the code-quality
    gate referenced in NeuroCrowd prompts.【F:agi_core/integration/neurocrowd.py†L187-L223】
- [ ] **API conformance**
  - `make run` then call `/health`, `/metrics`, `/biometric/submit`, and
    `/delegate/plan` with signed JWTs to verify the FastAPI surface and metric
    emission.【F:infrastructure/api.py†L112-L238】
- [ ] **Deployment rehearsal**
  - `kubectl apply -f deploy/k8s` (with namespace/ingress updates as needed) and
    confirm two pods reach `Ready` while Prometheus discovers the scrape target via
    the annotations.【F:deploy/k8s/03-deployment.yaml†L16-L60】

## 5. Artifacts

- [ ] **Architecture diagram** — regenerate via
  `CI=1 npx @mermaid-js/mermaid-cli -i docs/architecture.mmd -o docs/architecture.svg -p docs/puppeteer-config.json` to match the
  topology used in the evaluation.【F:docs/architecture.mmd†L1-L22】
- [ ] **Logs & metrics** — persist Prometheus snapshots and NeuroCrowd semantic
  memory exports for inclusion in the thesis appendix.

Completing every checkbox above reproduces the thesis-grade deployment, metrics,
and evaluation workflow for CBC-Ω².
