#!/usr/bin/env bash
# One-time setup ON the Jetson board (Linux aarch64 + CUDA).
# Run: bash scripts/setup_jetson.sh
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== Jetson device ==="
cat /proc/device-tree/model 2>/dev/null || echo "(model unknown)"
python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" \
  || echo "Install PyTorch for Jetson first (see NVIDIA Jetson PyTorch wheel)."

echo "=== Python deps ==="
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt

echo "=== Verify CUDA ==="
python3 -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available — install Jetson PyTorch wheel'
print('GPU:', torch.cuda.get_device_name(0))
"

echo "=== Smoke test (1 sample) ==="
python3 profile_jetson.py --max-samples 1 --skip-pareto --profile-batches 5

echo "Setup OK. Full run: bash scripts/run_jetson_profile.sh"
