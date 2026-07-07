#!/usr/bin/env bash
# Run on the Jetson (Linux aarch64 + CUDA), not on CPU-only Windows dev machines.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== Jetson max clocks (recommended before WCET profiling) ==="
if command -v jetson_clocks >/dev/null 2>&1; then
  sudo jetson_clocks
else
  echo "jetson_clocks not found — skip or install jetson-utils"
fi

python3 profile_jetson.py "$@"
