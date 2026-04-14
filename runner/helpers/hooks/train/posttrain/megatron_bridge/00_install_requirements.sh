#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
set -euo pipefail

# Install Megatron-Bridge dependencies
echo "[+] Installing Megatron-Bridge dependencies..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Set up pip cache directory
PRIMUS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
DATA_PATH="${DATA_PATH:-${PRIMUS_ROOT}/data}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-${DATA_PATH}/pip_cache}"

echo "[INFO] Using pip cache: ${PIP_CACHE_DIR}"
mkdir -p "${PIP_CACHE_DIR}"

pip install --cache-dir="${PIP_CACHE_DIR}" "onnx==1.20.0rc1"
pip install --cache-dir="${PIP_CACHE_DIR}" -U nvidia-modelopt
pip install --cache-dir="${PIP_CACHE_DIR}" -U nvidia_resiliency_ext

pip install --cache-dir="${PIP_CACHE_DIR}" -U "datasets>=4.0.0" || { echo "[ERROR] Failed to upgrade datasets. Training will fail at data loading."; exit 1; }

pip install --cache-dir="${PIP_CACHE_DIR}" -r "${SCRIPT_DIR}/requirements-megatron-bridge.txt"

# Install emerging-optimizers (Muon optimizer, required by kimi_k25_vl recipe)
pip install --cache-dir="${PIP_CACHE_DIR}" -e "${PRIMUS_ROOT}/third_party/Emerging-Optimizers" 2>/dev/null || \
  pip install --cache-dir="${PIP_CACHE_DIR}" "emerging-optimizers" 2>/dev/null || \
  echo "[WARNING] Could not install emerging-optimizers; Muon optimizer will not be available"

echo "[OK] Megatron-Bridge dependencies installed"
