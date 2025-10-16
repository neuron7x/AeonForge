#!/usr/bin/env bash
set -euo pipefail
URL="${1:-http://localhost:8000/health}"
TIMEOUT="${2:-60}"
echo "⏳ Waiting for $URL (timeout ${TIMEOUT}s)"
end=$((SECONDS+TIMEOUT))
while (( SECONDS < end )); do
  if curl -fsS "$URL" >/dev/null 2>&1; then
    echo "✅ Service is up: $URL"
    exit 0
  fi
  sleep 2
done
echo "❌ Timed out waiting for $URL"
exit 1
