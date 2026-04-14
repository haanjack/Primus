#!/bin/bash
#
# Kimi-K2.5-VL SFT Toy Model Smoke Test (Container Mode)
#
# Runs a 5-iteration smoke test using the local toy model at:
#   data/kimi_k25_vl_toy/  (7GB, 4 layers, 8 experts)
#
# Phase A: Converts toy HF checkpoint -> Megatron format (auto, first run only)
# Phase B: Runs 5 training iterations, verifies no crash and finite loss
#
# Usage:
#   bash examples/megatron_bridge/scripts/kimi_k25_vl_sft_toy_smoke.sh
#
# Prerequisites:
#   - data/kimi_k25_vl_toy/ must exist (toy HF model)
#   - 4x AMD MI300X GPUs required (default; override with GPUS_PER_NODE)
#   - No HF_TOKEN needed (uses local toy model)
#

set -e

# ===== Configuration =====
CONTAINER_IMAGE="${CONTAINER_IMAGE:-rocm/primus:v26.2}"
CONFIG_FILE="${CONFIG_FILE:-examples/megatron_bridge/configs/MI300X/kimi_k25_vl-sft-toy-smoke.yaml}"
WORKSPACE="${WORKSPACE:-.}"

# Data paths (must be absolute for Docker bind mounts)
DATA_PATH="$(realpath "${DATA_PATH:-./data}")"
TOY_MODEL_PATH="${DATA_PATH}/kimi_k25_vl_toy"
HF_HOME="${HF_HOME:-${DATA_PATH}/huggingface}"

# GPU selection (default: GPUs 0-3 to avoid conflicts with other users)
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0,1,2,3}"

# Optional
WANDB_API_KEY="${WANDB_API_KEY:-}"

# ===== Validation =====
if [ ! -d "$TOY_MODEL_PATH" ]; then
    echo "Error: Toy model not found at ${TOY_MODEL_PATH}"
    echo "Expected: data/kimi_k25_vl_toy/ (7GB, 4 layers, 8 experts)"
    exit 1
fi

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

# ===== Main =====
echo "========================================"
echo "Kimi-K2.5-VL SFT Toy Smoke Test"
echo "========================================"
echo "Container : $CONTAINER_IMAGE"
echo "Config    : $CONFIG_FILE"
echo "Toy model : $TOY_MODEL_PATH"
echo "========================================"
echo ""
echo "Phase A: toy HF model -> Megatron conversion (auto, ~2-5 min on first run)"
echo "Phase B: 5 training iterations (validates end-to-end pipeline)"
echo ""

cd "$WORKSPACE"

./primus-cli container \
    --image "$CONTAINER_IMAGE" \
    --env DATA_PATH="$DATA_PATH" \
    --env GPUS_PER_NODE="$GPUS_PER_NODE" \
    --env HIP_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES" \
    ${HF_HOME:+--env HF_HOME="$HF_HOME"} \
    ${WANDB_API_KEY:+--env WANDB_API_KEY="$WANDB_API_KEY"} \
    -- train posttrain --config "$CONFIG_FILE"

echo ""
echo "========================================"
echo "Smoke test complete. Check logs for:"
echo "  - No crash (all ranks exited 0)"
echo "  - Finite loss across 5 iterations"
echo "========================================"
