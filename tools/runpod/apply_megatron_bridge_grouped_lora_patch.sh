#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/workspace/src/RL}"
BRIDGE_DIR="${BRIDGE_DIR:-$ROOT_DIR/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge}"
PATCH_FILE="${PATCH_FILE:-$ROOT_DIR/tools/runpod/megatron_bridge_grouped_lora.patch}"

if [[ ! -d "$BRIDGE_DIR/.git" ]]; then
  echo "Megatron-Bridge git repo not found: $BRIDGE_DIR" >&2
  exit 1
fi

if [[ ! -f "$PATCH_FILE" ]]; then
  echo "Patch file not found: $PATCH_FILE" >&2
  exit 1
fi

cd "$BRIDGE_DIR"

if git apply --check "$PATCH_FILE"; then
  git apply "$PATCH_FILE"
  echo "Applied grouped LoRA patch to $BRIDGE_DIR"
else
  echo "Patch does not apply cleanly. Current Bridge source may already be patched or has drifted." >&2
  echo "Check with:" >&2
  echo "  cd $BRIDGE_DIR && git diff -- src/megatron/bridge/peft src/megatron/bridge/models/conversion/peft_bridge.py" >&2
  exit 1
fi
