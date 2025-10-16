.PHONY: lint type bandit unit cov it-up it-down smoke

lint:
	python -m pip install -U pip
	pip install ruff
	ruff check .

type:
	pip install mypy
	mypy --ignore-missing-imports .

bandit:
	pip install bandit
	bandit -r . -q || true

unit:
	pip install -r requirements.txt || true
	pip install pytest pytest-cov
	pytest -q

cov:
	pip install -r requirements.txt || true
	pip install pytest pytest-cov
	pytest --cov=. --cov-report=term-missing --cov-report=xml

it-up:
	docker compose -f docker-compose.integration.yml up -d --wait

it-down:
	docker compose -f docker-compose.integration.yml down -v --remove-orphans

smoke:
	docker build -f Dockerfile.ci -t agi-core-x:ci .
	docker run -d -p 8000:8000 --name agi-core-x-ci --rm agi-core-x:ci bash -lc "uvicorn $$API_APP_MODULE --host 0.0.0.0 --port 8000"
	sleep 4
	curl -fsS http://localhost:8000/health | cat
	docker stop agi-core-x-ci >/dev/null
