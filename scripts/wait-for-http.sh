#!/usr/bin/env bash
set -euo pipefail
URL="${1:-http://localhost:8000/health}"
ATTEMPTS="${2:-30}"

for i in $(seq 1 "$ATTEMPTS"); do
  if curl -fsS "$URL" >/dev/null 2>&1; then
    echo "Ready: $URL"
    exit 0
  fi
  echo "Waiting ($i/$ATTEMPTS): $URL"
  sleep 1
done

echo "ERROR: Service not ready: $URL" >&2
exit 1
