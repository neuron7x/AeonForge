FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends     build-essential curl && rm -rf /var/lib/apt/lists/*

# Venv
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy whole repo (safe for generic projects)
COPY . /app

# Install deps if requirements exist; otherwise minimal runtime
RUN if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt;     else pip install --no-cache-dir fastapi uvicorn; fi

# Non-root
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3   CMD curl -fsS http://localhost:8000/health || exit 1

# APP_MODULE may be injected by CI/composer
ENV APP_MODULE="api:app"
CMD uvicorn ${APP_MODULE} --host 0.0.0.0 --port 8000 --workers 2
