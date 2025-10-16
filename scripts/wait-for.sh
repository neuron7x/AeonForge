#!/usr/bin/env bash
# wait-for.sh host port timeout_seconds
set -e
HOST=${1:?host required}
PORT=${2:?port required}
TIMEOUT=${3:-60}

echo "Waiting for $HOST:$PORT up to ${TIMEOUT}s..."
for i in $(seq 1 $TIMEOUT); do
  if (echo > /dev/tcp/$HOST/$PORT) >/dev/null 2>&1; then
    echo "OK: $HOST:$PORT is reachable."
    exit 0
  fi
  sleep 1
done
echo "ERROR: $HOST:$PORT not reachable in ${TIMEOUT}s"
exit 1
