# AeonForge — Continuous Quality Pipeline

[![CI](https://github.com/neuron7x/aeonforge/actions/workflows/ci.yml/badge.svg)](https://github.com/neuron7x/aeonforge/actions/workflows/ci.yml)
[![PR Checks](https://github.com/neuron7x/aeonforge/actions/workflows/pr-checks.yml/badge.svg)](https://github.com/neuron7x/aeonforge/actions/workflows/pr-checks.yml)

Production-grade GitHub Actions setup for **full PR testing with coverage diff, security scans, static analysis and dockerized integration tests**.

## What’s inside

- **Ruff + Mypy + Bandit** (lint, type, security) on every push/PR
- **Pytest** with coverage reports and **coverage diff gate** (no regressions)
- **Dockerized integration tests** via `docker compose` (Postgres, Redis, Neo4j)
- Optional **API smoke** inside built container (`/health`) — enable by setting repo variable `API_SMOKE_CMD` (e.g. `uvicorn api:app --host 0.0.0.0 --port 8000`)
- **Auto‑labeler** for PRs based on file changes

## Quick start

1. Drop this folder into your repo (or unzip the release patch).
2. Commit and push to `develop` or open a PR → pipeline runs automatically.
3. To enable API smoke:
   - Go to **Settings → Secrets and variables → Actions → Variables**.
   - Add `API_SMOKE_CMD` with your startup command (example above).

## Developer tooling

```bash
pip install -r requirements-dev.txt
pytest -q --cov=. --cov-report=term-missing
ruff check .
mypy .
bandit -r .
```

## Integration test locally

```bash
docker compose -f docker-compose.ci.yml up -d
pytest -q tests/integration -m infra
docker compose -f docker-compose.ci.yml down -v
```

## Coverage non‑regression

The job `coverage_diff` runs tests on the PR **and** on the base branch,
then compares `coverage.xml` files. The build fails if PR coverage drops below base.

## Badges

- CI: `https://github.com/<owner>/<repo>/actions/workflows/ci.yml/badge.svg`
- PR Checks: `https://github.com/<owner>/<repo>/actions/workflows/pr-checks.yml/badge.svg`
