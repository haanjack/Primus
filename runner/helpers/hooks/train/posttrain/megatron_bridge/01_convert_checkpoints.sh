#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
set -euo pipefail

# Parse config to get HF model path and prepare checkpoint
LOG_INFO_RANK0 "[+] Preparing Megatron-Bridge checkpoint..."

# Find --config argument from command line
CONFIG_FILE=""
for ((i=1; i<=$#; i++)); do
    if [[ "${!i}" == "--config" ]]; then
        j=$((i+1))
        CONFIG_FILE="${!j}"
        break
    fi
done

if [[ -z "$CONFIG_FILE" ]]; then
    LOG_ERROR_RANK0 "[WARNING] No --config argument found, skipping checkpoint preparation"
    exit 0
fi

# Parse the complete config with all extends and nested configs
PRIMUS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../../.." && pwd)"
cd "$PRIMUS_ROOT"

# Convert CONFIG_FILE to absolute path
if [[ ! "$CONFIG_FILE" = /* ]]; then
    CONFIG_FILE="${PRIMUS_ROOT}/${CONFIG_FILE#./}"
fi

# Extract hf_path and pretrained_checkpoint from fully parsed config (single parse)
_PARSE_OUT=$(python3 -c "
import sys
sys.path.insert(0, '${PRIMUS_ROOT}')
from pathlib import Path
from primus.core.config.primus_config import load_primus_config, get_module_config

cfg = load_primus_config(Path('${CONFIG_FILE}'), None)
post_trainer = get_module_config(cfg, 'post_trainer')

if post_trainer is None or not hasattr(post_trainer, 'params'):
    print('CKPT_STATUS=missing')
    sys.exit(0)

params = post_trainer.params

# Check both flat (params.pretrained_checkpoint) and nested (params.checkpoint.pretrained_checkpoint)
ckpt = getattr(params, 'pretrained_checkpoint', 'unset')
if ckpt == 'unset' and hasattr(params, 'checkpoint'):
    ckpt = getattr(params.checkpoint, 'pretrained_checkpoint', 'unset')
ckpt_status = 'null' if ckpt is None else str(ckpt)
print(f'CKPT_STATUS={ckpt_status}')

hf_path = getattr(params, 'hf_path', '')
print(f'HF_PATH={hf_path}')
" 2>/dev/null || echo "CKPT_STATUS=error")

# Parse outputs
_CKPT_STATUS=$(echo "$_PARSE_OUT" | grep '^CKPT_STATUS=' | cut -d= -f2-)
HF_PATH=$(echo "$_PARSE_OUT" | grep '^HF_PATH=' | cut -d= -f2-)

LOG_DEBUG_RANK0 "pretrained_checkpoint: ${_CKPT_STATUS}"
LOG_DEBUG_RANK0 "HF_PATH captured: ${HF_PATH}"

# If pretrained_checkpoint is explicitly null, user wants random init — skip conversion
if [[ "$_CKPT_STATUS" == "null" ]]; then
    LOG_INFO_RANK0 "pretrained_checkpoint=null in config: training from random init, skipping conversion"
    exit 0
fi

LOG_DEBUG_RANK0 "HF_PATH captured: ${HF_PATH}"

if [[ -z "$HF_PATH" ]]; then
    LOG_ERROR_RANK0 "[WARNING] No hf_path found in config"
    LOG_ERROR_RANK0 "[WARNING] Assuming checkpoint already exists and conversion is not needed"
    exit 0
fi

# Set paths
DATA_PATH="${DATA_PATH:-${PRIMUS_ROOT}/data}"
HF_CACHE="${HF_HOME:-${DATA_PATH}/huggingface}/hub"
MEGATRON_PATH="${DATA_PATH}/megatron_checkpoints/$(basename "${HF_PATH}")"

LOG_INFO_RANK0 "HF Model: ${HF_PATH}"
LOG_INFO_RANK0 "HF Cache: ${HF_CACHE}"
LOG_INFO_RANK0 "Megatron Path: ${MEGATRON_PATH}"

# Check if HF checkpoint already downloaded
HF_MODEL_CACHE="${HF_CACHE}/models--$(echo "${HF_PATH}" | tr '/' '--')"
if [[ -d "$HF_MODEL_CACHE" ]]; then
    LOG_INFO_RANK0 "HF checkpoint already cached at ${HF_MODEL_CACHE}"
else
    LOG_INFO_RANK0 "HF checkpoint will be downloaded from ${HF_PATH}"
fi

# Check if Megatron checkpoint already exists
if [[ -d "$MEGATRON_PATH" ]]; then
    LOG_INFO_RANK0 "Megatron checkpoint already exists at ${MEGATRON_PATH}, skipping conversion"
    echo "extra.pretrained_checkpoint=${MEGATRON_PATH}"
    exit 0
fi

# Convert checkpoint (only on rank 0, others wait)
NODE_RANK="${NODE_RANK:-${RANK:-0}}"
LOCK_FILE="${MEGATRON_PATH}.converting.lock"
DONE_FILE="${MEGATRON_PATH}.done"

if [[ "$NODE_RANK" == "0" ]]; then
    # Rank 0: perform the conversion
    LOG_INFO_RANK0 "[+] Converting HF checkpoint to Megatron format..."
    mkdir -p "$(dirname "${MEGATRON_PATH}")"

    # Create lock file; clean it up if we exit unexpectedly
    touch "$LOCK_FILE"
    trap 'rm -f "$LOCK_FILE"' EXIT

    # Set up Python path for Megatron-Bridge
    export PYTHONPATH="${PRIMUS_ROOT}/third_party/Megatron-Bridge/src:${PRIMUS_ROOT}/third_party/Megatron-Bridge/3rdparty/Megatron-LM:${PYTHONPATH:-}"

    # Use memory-efficient conversion script with malloc_trim patch.
    # The standard convert_checkpoints.py OOM-kills on 1T+ models because glibc
    # keeps freed tensor pages in its malloc heap instead of returning them to the OS.
    CONVERT_SCRIPT="${PRIMUS_ROOT}/runner/helpers/hooks/train/posttrain/megatron_bridge/lib/convert_hf_to_megatron.py"

    python3 "${CONVERT_SCRIPT}" \
      --hf-model "${HF_PATH}" \
      --megatron-path "${MEGATRON_PATH}" \
      --trust-remote-code

    # Create done file and remove lock
    touch "$DONE_FILE"
    rm -f "$LOCK_FILE"
    trap - EXIT

    LOG_SUCCESS_RANK0 "Checkpoint prepared at ${MEGATRON_PATH}"
else
    # Other ranks: wait for rank 0 to complete
    LOG_INFO_RANK0 "[RANK ${NODE_RANK}] Waiting for rank 0 to complete checkpoint conversion..."

    # Wait for done file (with timeout)
    timeout=1800  # 30 minutes timeout (1T model conversion can take 10-20 min)
    elapsed=0
    while [[ ! -f "$DONE_FILE" ]] && [[ $elapsed -lt $timeout ]]; do
        sleep 5
        elapsed=$((elapsed + 5))
    done

    if [[ ! -f "$DONE_FILE" ]]; then
        echo "[RANK ${NODE_RANK}] Timeout waiting for checkpoint conversion"
        exit 1
    fi

    echo "[OK] [RANK ${NODE_RANK}] Checkpoint ready at ${MEGATRON_PATH}"
fi

echo "extra.pretrained_checkpoint=${MEGATRON_PATH}"
