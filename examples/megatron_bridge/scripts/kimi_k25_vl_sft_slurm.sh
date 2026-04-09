#!/bin/bash
#
# Kimi-K2.5-VL SFT Training with Primus-CLI (SLURM Mode)
#
# This SLURM script runs Kimi-K2.5-VL supervised fine-tuning on multi-node cluster.
#
# Submit with:
#   sbatch examples/megatron_bridge/scripts/kimi_k25_vl_sft_slurm.sh
#
# Or for dry-run:
#   sbatch --dry-run examples/megatron_bridge/scripts/kimi_k25_vl_sft_slurm.sh
#

#SBATCH --job-name=kimi_k25_vl_sft
#SBATCH --account=${SLURM_ACCOUNT:-default}
#SBATCH --partition=${SLURM_PARTITION:-batch}
#SBATCH --nodes=${SLURM_NODES:-128}               # For full model training
#SBATCH --ntasks-per-node=8                        # 8 GPUs per node
#SBATCH --cpus-per-task=12                         # NUMA-aware CPU allocation
#SBATCH --gpus-per-node=8                          # 8 GPUs per node
#SBATCH --time=120:00:00                           # Max runtime: 120 hours
#SBATCH --output=${SLURM_LOG_DIR:-.}/logs/%x_%j.out
#SBATCH --error=${SLURM_LOG_DIR:-.}/logs/%x_%j.err
#SBATCH --exclusive=user
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=${SLURM_MAIL_USER:-}

set -e

# ===== Configuration =====
# Cluster setup
NODES_PER_JOB=${SLURM_NODES:-128}
GPUS_PER_NODE=8
TOTAL_GPUS=$((NODES_PER_JOB * GPUS_PER_NODE))

# Container image
CONTAINER_IMAGE="${CONTAINER_IMAGE:-rocm/primus:v26.2}"

# Config file
CONFIG_FILE="${CONFIG_FILE:-examples/megatron_bridge/configs/MI355X/kimi_k25_vl-sft.yaml}"

# Workspace
WORKSPACE="${WORKSPACE:-.}"

# Environment variables
HF_TOKEN="${HF_TOKEN:-}"
HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
WANDB_API_KEY="${WANDB_API_KEY:-}"

# Container mounts
CONTAINER_MOUNTS="${CONTAINER_MOUNTS:-}"

# ===== Validation =====
if [ -z "$HF_TOKEN" ]; then
    echo "Error: HF_TOKEN is not set. Set with: export HF_TOKEN=<token>"
    exit 1
fi

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

# ===== Logging Setup =====
SLURM_LOG_DIR=${SLURM_LOG_DIR:-.}/logs
mkdir -p "$SLURM_LOG_DIR"

# ===== Logging & Info =====
echo "========================================"
echo "Kimi-K2.5-VL SFT with Primus-CLI (SLURM)"
echo "========================================"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Nodes: $NODES_PER_JOB"
echo "Total GPUs: $TOTAL_GPUS (TP=2 × PP=16 × EP=32)"
echo "Container Image: $CONTAINER_IMAGE"
echo "Config File: $CONFIG_FILE"
echo "Workspace: $WORKSPACE"
echo "HF_TOKEN: ${HF_TOKEN:0:10}... (hidden)"
echo "========================================"
echo ""

# ===== Environment Setup =====
export NODES_PER_JOB
export GPUS_PER_NODE
export TOTAL_GPUS
export MASTER_ADDR="${SLURM_NODELIST%%,*}"
export MASTER_PORT=1234

# ===== Execute Training =====
# Build the primus-cli command
PRIMUS_COMMAND="train posttrain --config $CONFIG_FILE"

# Build container mount options
MOUNTS="-v $WORKSPACE:$WORKSPACE"
if [ -n "$CONTAINER_MOUNTS" ]; then
    MOUNTS="$MOUNTS $CONTAINER_MOUNTS"
fi

if [ -n "$HF_HOME" ]; then
    MOUNTS="$MOUNTS -v $HF_HOME:$HF_HOME"
fi

echo "Starting training on $NODES_PER_JOB nodes..."
echo ""

cd "$WORKSPACE"

# Execute with srun for SLURM multi-node coordination
srun --mpi=pmix \
    -N $NODES_PER_JOB \
    -n $((NODES_PER_JOB * GPUS_PER_NODE)) \
    --ntasks-per-node=$GPUS_PER_NODE \
    --container-image="$CONTAINER_IMAGE" \
    ${CONTAINER_MOUNTS:+--container-mounts=$(echo "$MOUNTS" | tr ' ' ',')} \
    --no-container-entrypoint \
    --no-container-remap-root \
    env HF_TOKEN="$HF_TOKEN" \
        ${HF_HOME:+HF_HOME="$HF_HOME"} \
        ${WANDB_API_KEY:+WANDB_API_KEY="$WANDB_API_KEY"} \
        MASTER_ADDR="$MASTER_ADDR" \
        MASTER_PORT="$MASTER_PORT" \
        ./primus-cli slurm srun -N $NODES_PER_JOB -- $PRIMUS_COMMAND

echo ""
echo "========================================"
echo "Training complete!"
echo "Check output: $SLURM_LOG_DIR/$SLURM_JOB_NAME/${SLURM_JOB_ID}.out"
echo "========================================"
