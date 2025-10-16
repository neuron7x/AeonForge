# Stage 1 — builder
FROM python:3.11-slim as builder

WORKDIR /app
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc libpq-dev curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel && \
    pip wheel --no-cache-dir -r requirements.txt -w /wheels

# Stage 2 — runtime
FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1

RUN useradd -m -u 1000 botuser && \
    apt-get update && apt-get install -y --no-install-recommends \
    postgresql-client curl && rm -rf /var/lib/apt/lists/*

COPY --from=builder /wheels /wheels
RUN pip install --no-index --find-links=/wheels -r /wheels/requirements.txt

COPY . /app
RUN chown -R botuser:botuser /app

USER botuser
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
