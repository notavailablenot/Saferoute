#!/usr/bin/env bash
# Build the backend image and measure image size and container startup time (Sprint 2 metrics).
# Usage: bash scripts/measure_docker.sh        (run from the repo root, Docker Desktop running)
set -euo pipefail
mkdir -p reports
echo "== build =="
t0=$(date +%s.%N)
docker compose build api
t1=$(date +%s.%N)
size=$(docker image inspect saferoute-api:1.0 --format '{{.Size}}')
echo "== start =="
docker compose down >/dev/null 2>&1 || true
s0=$(date +%s.%N)
docker compose up -d api
until curl -fs http://127.0.0.1:8000/health | grep -q '"ready"'; do sleep 0.2; done
s1=$(date +%s.%N)
build_s=$(python3 -c "print($t1 - $t0)")
start_s=$(python3 -c "print($s1 - $s0)")
size_mb=$(python3 -c "print(round($size / 1e6, 1))")
docker compose ps
printf '{\n  "image": "saferoute-api:1.0",\n  "image_size_mb": %s,\n  "build_time_s": %.1f,\n  "startup_to_ready_s": %.2f\n}\n' \
  "$size_mb" "$build_s" "$start_s" | tee reports/docker_metrics.json
