# RunPod SFT Setup Notes

From the repo root on a fresh RunPod CUDA container:

```bash
bash tools/runpod/setup_sft_env.sh
HF_HOME=/workspace/HF_HOME TORCH_CUDA_ARCH_LIST=9.0 NEMO_RL_VENV_DIR=/tmp/nemo_rl_venvs NRL_FORCE_REBUILD_VENVS=true uv run --extra mcore python examples/run_sft.py
```

What the script handles:

- Installs `build-essential`, `cmake`, `ninja-build`, `git`, `curl`, and `libibverbs-dev`. `cmake` is required by `transformer-engine`; `libibverbs-dev` is required by `deep-ep` against NVSHMEM/Infiniband headers.
- Installs `uv` if missing.
- Runs `uv sync --extra mcore`.
- Adds a `sitecustomize.py` hook to preload PyTorch's wheel-bundled `libnvJitLink.so.12`. This avoids RunPod system CUDA libraries shadowing PyTorch cu129 libraries.
- Auto-selects `TORCH_CUDA_ARCH_LIST=9.0` for H200 and `12.0` otherwise. Override it if your GPU is different.
- Because RunPod can share `/workspace` across pod types, put generated Ray worker venvs on local disk with `NEMO_RL_VENV_DIR=/tmp/nemo_rl_venvs`. This avoids stale file handles while `uv` installs wheels. Use `NRL_FORCE_REBUILD_VENVS=true` when switching GPU/CUDA images so generated Ray worker venvs are rebuilt.
- Verifies `import torch` and CUDA visibility.

Megatron-Bridge grouped LoRA patch:

```bash
bash tools/runpod/apply_megatron_bridge_grouped_lora_patch.sh
```

This applies `tools/runpod/megatron_bridge_grouped_lora.patch` to `3rdparty/Megatron-Bridge-workspace/Megatron-Bridge`. The patch adds grouped MoE expert LoRA support while keeping grouped GEMM, plus HF PEFT adapter export for per-expert LoRA keys.

Model notes:

- The original default `meta-llama/Llama-3.2-1B` is gated and needs `huggingface-cli login` plus granted model access.
- `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` is public, but full SFT with this default config OOMed on one 95 GiB GPU.
- `HuggingFaceTB/SmolLM2-135M` is a practical public smoke-test model for verifying the command runs end to end.
