"""Target-hardware detection for RTS timing reports (Jetson vs CPU/desktop GPU)."""

from __future__ import annotations

import platform
import subprocess
from pathlib import Path

import torch

from training.trainer import get_device


def detect_jetson() -> dict:
    """
    Read Linux device-tree model string (present on Jetson boards).

    Returns {"is_jetson": bool, "jetson_model": str | None}.
    """
    model_path = Path("/proc/device-tree/model")
    if not model_path.is_file():
        return {"is_jetson": False, "jetson_model": None}
    try:
        raw = model_path.read_bytes().decode("utf-8", errors="ignore").strip("\x00").strip()
    except OSError:
        return {"is_jetson": False, "jetson_model": None}
    low = raw.lower()
    is_jet = "jetson" in low or "tegra" in low
    return {"is_jetson": is_jet, "jetson_model": raw if is_jet else None}


def _jetpack_hint() -> str | None:
    """Best-effort JetPack / L4T version from dpkg (Jetson only)."""
    try:
        out = subprocess.run(
            ["dpkg-query", "--showformat=${Version}", "--show", "nvidia-l4t-core"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        ver = out.stdout.strip()
        return ver if ver else None
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None


def hardware_info() -> dict:
    """Structured hardware block for JSON reports and LaTeX table captions."""
    dev = get_device()
    jetson = detect_jetson()
    info: dict = {
        "device": str(dev),
        "cuda_available": torch.cuda.is_available(),
        "platform": platform.platform(),
        **jetson,
    }
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        try:
            props = torch.cuda.get_device_properties(0)
            info["gpu_total_memory_gb"] = round(props.total_memory / (1024**3), 2)
        except Exception:
            pass
    if jetson["is_jetson"]:
        l4t = _jetpack_hint()
        if l4t:
            info["l4t_core"] = l4t
        info["note"] = f"NVIDIA Jetson ({jetson['jetson_model']})"
    elif torch.cuda.is_available():
        info["note"] = "CUDA GPU (not Jetson); OK for dev, re-profile on Jetson for paper."
    else:
        info["note"] = "CPU-only; run profile_jetson.py on NVIDIA Jetson for paper WCET."
    return info


def require_cuda_for_profiling() -> torch.device:
    """Abort with a clear message if CUDA is unavailable (Windows laptop, etc.)."""
    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA not available on this machine.\n"
            "Copy the repo to your Jetson, then run:\n"
            "  bash scripts/run_jetson_profile.sh\n"
            "Or SSH in and run:\n"
            "  python profile_jetson.py"
        )
    return get_device()
