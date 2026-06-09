# Converters

This directory contains one-off export utilities for Megatron checkpoints.

## LoRA to Hugging Face

`examples/converters/convert_lora_to_hf.py` exports a Megatron LoRA adapter checkpoint into Hugging Face format. The script now imports `nemo_rl` before touching `megatron.bridge`, which ensures the local Megatron-LM checkout is added to `sys.path` automatically.

Example adapter-only export:

```bash
HF_HOME=/workspace/HF_HOME/ \
uv run --extra mcore python examples/converters/convert_lora_to_hf.py \
  --base-ckpt /workspace/HF_HOME/nemo_rl/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/iter_0000000/ \
  --adapter-ckpt results/sft-nemotron-lora-v3/step_1100/policy/weights/iter_0000000/ \
  --hf-model-name nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
  --hf-ckpt-path results/sft-nemotron-lora-v3/step_1100/adapter-weight \
  --adapter-only
```

Notes:

- Run the command from the repo root so relative paths resolve as expected.
- `HF_HOME` should point at the cache that already contains the base model checkpoint.
- Use `--adapter-only` when you want a PEFT adapter directory. Omit it to merge the adapter into the base model and export a full Hugging Face checkpoint.
