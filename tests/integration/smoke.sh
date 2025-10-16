#!/usr/bin/env bash
set -euo pipefail
docker compose up -d --build
./wait_for_http.sh http://localhost:8000/health 60
curl -fsS http://localhost:8000/health && echo "OK"
docker compose down -v
