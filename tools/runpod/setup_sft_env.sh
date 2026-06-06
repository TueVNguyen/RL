#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root.
# This prepares a RunPod-style CUDA container for:
#   HF_HOME=/workspace/HF_HOME/ uv run --extra mcore python examples/run_sft.py

REPO_ROOT="${REPO_ROOT:-$(pwd)}"
HF_HOME="${HF_HOME:-/workspace/HF_HOME}"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-}"

cd "$REPO_ROOT"
mkdir -p "$HF_HOME"

if command -v apt-get >/dev/null 2>&1; then
  export DEBIAN_FRONTEND=noninteractive
  apt-get update
  apt-get install -y --no-install-recommends build-essential cmake ninja-build git curl libibverbs-dev
else
  echo "apt-get not found; install RDMA/Infiniband headers manually (libibverbs-dev)." >&2
fi

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

# Avoid the common RunPod CUDA 12.8 system library shadowing PyTorch cu129's nvJitLink.
install_nvjitlink_sitecustomize() {
  local venv_path="$1"
  local site_packages
  site_packages="$(find "$venv_path" -maxdepth 3 -type d -path '*/site-packages' | head -n 1 || true)"
  if [[ -z "$site_packages" ]]; then
    return 0
  fi

  cat >"$site_packages/sitecustomize.py" <<'PY'
"""Local startup fixes for RunPod CUDA/PyTorch wheel environments."""

from __future__ import annotations

import ctypes
from pathlib import Path


def _preload_torch_nvjitlink() -> None:
    site_packages = Path(__file__).resolve().parent
    nvjitlink = site_packages / "nvidia" / "nvjitlink" / "lib" / "libnvJitLink.so.12"
    if not nvjitlink.exists():
        return
    try:
        ctypes.CDLL(str(nvjitlink), mode=ctypes.RTLD_GLOBAL)
    except OSError:
        return


_preload_torch_nvjitlink()
PY
}

if [[ -z "$TORCH_CUDA_ARCH_LIST" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi --query-gpu=name --format=csv,noheader | grep -qi "H200"; then
    TORCH_CUDA_ARCH_LIST="9.0"
  else
    TORCH_CUDA_ARCH_LIST="12.0"
  fi
fi

export HF_HOME
export TORCH_CUDA_ARCH_LIST
export NEMO_RL_VENV_DIR="${NEMO_RL_VENV_DIR:-/tmp/nemo_rl_venvs}"
mkdir -p "$NEMO_RL_VENV_DIR"

uv sync --extra mcore
install_nvjitlink_sitecustomize "$REPO_ROOT/.venv"

uv run python - <<'PY'
import torch
print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"device_count={torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"capability={torch.cuda.get_device_capability(0)}")
PY

cat <<MSG

Setup complete.

Recommended smoke run:
  HF_HOME=$HF_HOME TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST NEMO_RL_VENV_DIR=$NEMO_RL_VENV_DIR NRL_FORCE_REBUILD_VENVS=true uv run --extra mcore python examples/run_sft.py

Notes:
  - H200 pods should use TORCH_CUDA_ARCH_LIST=9.0. Blackwell RTX PRO 6000 pods should use 12.0.
  - Because /workspace is shared across pods, keep generated Ray worker venvs on local disk: export NEMO_RL_VENV_DIR=/tmp/nemo_rl_venvs
  - Rebuild generated worker venvs when switching GPU/CUDA images: export NRL_FORCE_REBUILD_VENVS=true
  - Use a public/small model for smoke tests, e.g. HuggingFaceTB/SmolLM2-135M.
  - nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 is public, but full SFT OOMed on a single 95 GiB GPU in this config.
  - If you want gated models, run: huggingface-cli login
MSG
