#!/usr/bin/env bash
set -euo pipefail
URL="${1:-http://localhost:8000/health}"
TIMEOUT="${2:-60}"
echo "Waiting up to ${TIMEOUT}s for ${URL} ..."
for i in $(seq 1 "${TIMEOUT}"); do
  if curl -fsS "${URL}" >/dev/null 2>&1; then
    echo "OK: ${URL} is up"
    exit 0
  fi
  sleep 1
done
echo "ERROR: ${URL} not reachable within ${TIMEOUT}s"
exit 1
