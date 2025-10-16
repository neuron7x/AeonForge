# AeonForge — CI/QA Pack

[![PR Quality Suite](https://github.com/neuron7x/aeonforge/actions/workflows/pr-suite.yml/badge.svg)](https://github.com/neuron7x/aeonforge/actions/workflows/pr-suite.yml)
[![Auto Labeler](https://github.com/neuron7x/aeonforge/actions/workflows/auto-labeler.yml/badge.svg)](https://github.com/neuron7x/aeonforge/actions/workflows/auto-labeler.yml)
[![Codecov](https://img.shields.io/codecov/c/github/neuron7x/aeonforge?logo=codecov)](https://app.codecov.io/gh/neuron7x/aeonforge)
![Python](https://img.shields.io/badge/python-3.11-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

Цей пакет додає **повне тестування PR** з покриттям, інтеграційні тести через `docker-compose` (Postgres/Redis/Neo4j), smoke-старт API (uvicorn + `/health`), авто-лейблинг PR та перевірку **coverage diff** (провал, якщо покриття PR < покриття базової гілки).

## Що входить
- `.github/workflows/pr-suite.yml` — основний конвеєр (lint, types, security, unit, coverage diff, integration, smoke)
- `.github/workflows/auto-labeler.yml` + `.github/labeler.yml` — автолейбли
- `docker-compose.integration.yml` — стек інтеграцій
- `.github/scripts/run-smoke.sh` — smoke (`uvicorn` + `curl /health`)
- `.github/scripts/wait_for_health.py` — очікування health сервісів
- `tests/integration/test_services.py` — перевірка з’єднання до Postgres/Redis/Neo4j
- Конфіги якості: `.coveragerc`, `pytest.ini`, `.ruff.toml`, `mypy.ini`, `.bandit`

## Використання
1. Скопіюй в корінь репозиторію (зберігаючи структуру).
2. У репо → **Settings → Actions → Variables** додай (за потреби):
   - `API_APP_MODULE` (за замовчуванням `api:app`)
3. Відкрий PR — конвеєр запуститься автоматично.

## Параметри
- **Coverage gate**: тестьовий джоб будує покриття для PR та **бази** (через `git worktree`) і валить PR, якщо `PR_coverage < Base_coverage`. Також `diff-cover` вимагає, щоб покриття **змінених рядків** було не нижче за базову загальну.
- **Smoke API**: модуль береться з `API_APP_MODULE` (наприклад, `package.api:app`). Порт — `8000`.

---

> Якщо вже маєш свої тести/налаштування — цей пакет не ламає існуюче, а додає стандартизований pipeline для PR.
