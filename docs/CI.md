# CI/CD & QA Pack

This repository ships with:
- GitHub Actions CI (`.github/workflows/ci.yml`)
- Coverage diff gate (fails if PR coverage < base branch coverage)
- Integration tests via `docker-compose` (Postgres, Redis, Neo4j, app)
- Smoke check (`/health`) both via compose and unit-style
- Auto-labeler for PRs

## Local integration

```bash
docker compose -f docker-compose.integration.yml up -d
curl http://localhost:8000/health
docker compose -f docker-compose.integration.yml down -v
```

## Coverage diff locally (rough)

```bash
# base
git checkout main && pytest --cov=. --cov-report=xml && python .github/scripts/coverage_percent.py coverage.xml > /tmp/base.txt
# pr
git checkout - && pytest --cov=. --cov-report=xml && python .github/scripts/coverage_percent.py coverage.xml > /tmp/pr.txt
python .github/scripts/assert_coverage_diff.py "$(cat /tmp/base.txt)" "$(cat /tmp/pr.txt)"
```
