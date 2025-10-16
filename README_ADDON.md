# AeonForge

[![CI](https://github.com/neuron7x/aeonforge/actions/workflows/ci.yml/badge.svg)](https://github.com/neuron7x/aeonforge/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## What this pack adds

- ✅ Fixed CI YAML (previous syntax error on line 12)
- ✅ Unit tests matrix + coverage
- ✅ Coverage diff vs base branch (fail on decrease)
- ✅ Integration start with `docker-compose` (Neo4j/Redis/Postgres)
- ✅ API smoke (uvicorn + curl /health, configurable via `APP_MODULE` repo variable)
- ✅ PR auto-labeling by path

### Quick notes

- Set repository variable **APP_MODULE** if your ASGI app is not `api:app`.
- Integration stack is optional; if you don't use Neo4j/Redis/Postgres, the services will still boot harmlessly and tests can skip.

