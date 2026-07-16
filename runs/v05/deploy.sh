#!/usr/bin/env bash
# Deploy/restart the v05 inference service for the 7810 recorder.
#   runs/v05/deploy.sh [PORT] [GPU]
# Stops whatever holds the port, starts the service detached (survives
# this shell), waits for /health, prints the endpoint. Logs append to
# runs/v05/serve.log. Detections are advisory — the agent tolerates
# this service being down.
set -euo pipefail
PORT="${1:-8100}"
GPU="${2:-2}"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"

OLD=$(lsof -t -i ":$PORT" 2>/dev/null || true)
if [ -n "$OLD" ]; then
  echo "stopping existing service on :$PORT (pid $OLD)"
  kill $OLD && sleep 2
fi

nohup uv run python -m runs.v05.serve --port "$PORT" --gpu "$GPU" \
  >> runs/v05/serve.log 2>&1 &
PID=$!
echo "started pid $PID (GPU $GPU), waiting for health..."

for i in $(seq 1 60); do
  if curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; then
    IP=$(hostname -I | awk '{print $1}')
    echo "READY: http://$IP:$PORT  (health: /health, inference: POST /infer)"
    curl -s "http://localhost:$PORT/health" | python3 -m json.tool
    exit 0
  fi
  sleep 2
done
echo "FAILED to become healthy in 120s — tail runs/v05/serve.log:" >&2
tail -20 runs/v05/serve.log >&2
exit 1
