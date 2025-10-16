# Minimal reproducible image for agi-core
FROM python:3.11-slim

WORKDIR /app
COPY . /app

# System deps (if needed later add: gcc, libgl1, etc.)
RUN pip install --no-cache-dir -e .[dev,viz]

# Default: run tests then a quick sanity run
CMD pytest -q && python -m agi_core.engine.runner --env tanh --iters 5 --horizon 5 --log-jsonl out/ci.jsonl
