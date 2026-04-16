#!/bin/bash
#
# Kimi-K2.5-VL SFT Training with Primus-CLI (Container Mode)
#
# Example usage:
#   export HF_TOKEN="your_huggingface_token"
#   ./examples/megatron_bridge/scripts/kimi_k25_vl_sft_container.sh
#
# This script runs Kimi-K2.5-VL supervised fine-tuning (SFT) using primus-cli in container mode.
# The HuggingFace checkpoint is automatically converted to Megatron format on first run.
#

set -e

# ===== Configuration =====
# Container image
CONTAINER_IMAGE="${CONTAINER_IMAGE:-rocm/primus:v26.2}"

# Config file (use test variant for development)
CONFIG_FILE="${CONFIG_FILE:-examples/megatron_bridge/configs/MI300X/kimi_k25_vl-sft-test.yaml}"

# Workspace directory
WORKSPACE="${WORKSPACE:-.}"

# Environment variables required
HF_TOKEN="${HF_TOKEN:-}"
HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
WANDB_API_KEY="${WANDB_API_KEY:-}"

# Volume mounts
CONTAINER_MOUNTS="${CONTAINER_MOUNTS:-}"

# ===== Validation =====
if [ -z "$HF_TOKEN" ]; then
    echo "Error: HF_TOKEN environment variable is not set."
    echo "Please set it with: export HF_TOKEN=<your_huggingface_token>"
    exit 1
fi

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

if [ ! -d "$WORKSPACE" ]; then
    echo "Error: Workspace directory not found: $WORKSPACE"
    exit 1
fi

# ===== Main Execution =====
echo "========================================"
echo "Kimi-K2.5-VL SFT with Primus-CLI"
echo "========================================"
echo "Container Image: $CONTAINER_IMAGE"
echo "Config File: $CONFIG_FILE"
echo "Workspace: $WORKSPACE"
echo "HF_TOKEN: ${HF_TOKEN:0:10}... (hidden)"
echo "========================================"
echo ""

# Build the primus-cli command
PRIMUS_COMMAND="train posttrain --config $CONFIG_FILE"

# Build the container mount options
MOUNTS="-v $WORKSPACE:$WORKSPACE"
if [ -n "$CONTAINER_MOUNTS" ]; then
    MOUNTS="$MOUNTS $CONTAINER_MOUNTS"
fi

# If HF_HOME is set, mount it to cache models
if [ -n "$HF_HOME" ]; then
    MOUNTS="$MOUNTS -v $HF_HOME:$HF_HOME"
fi

# Execute primus-cli with container mode
# The -- separator tells primus-cli that everything after it is a primus command
cd "$WORKSPACE"
./primus-cli container \
    --image "$CONTAINER_IMAGE" \
    --env HF_TOKEN="$HF_TOKEN" \
    ${HF_HOME:+--env HF_HOME="$HF_HOME"} \
    ${WANDB_API_KEY:+--env WANDB_API_KEY="$WANDB_API_KEY"} \
    $MOUNTS \
    -- $PRIMUS_COMMAND

echo ""
echo "========================================"
echo "Training complete! Check logs for details."
echo "========================================"
