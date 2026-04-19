# Kimi-K2 Post-Training Guide

This guide covers post-training (SFT and LoRA fine-tuning) of **Kimi-K2** (1T MoE) using the **Megatron Bridge** backend within Primus. It also covers checkpoint conversion testing, smoke testing, and pretrain command structure.

---

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Quick Start — Smoke Test](#quick-start--smoke-test)
- [Pretrain](#pretrain)
- [Post-Training](#post-training)
  - [SFT (Full Fine-Tuning)](#sft-full-fine-tuning)
  - [LoRA](#lora)
- [Checkpoint Conversion](#checkpoint-conversion)
  - [Automated conversion via hook](#automated-conversion-via-hook-recommended)
  - [Standalone conversion](#standalone-conversion-no-training)
  - [Checkpoint Conversion Test](#checkpoint-conversion-test)
- [Running Modes](#running-modes)
  - [Container Mode](#container-mode)
  - [Slurm Mode](#slurm-mode)
  - [Direct Mode](#direct-mode)
- [Hardware Configurations](#hardware-configurations)
  - [MI300X](#mi300x)
  - [MI355X](#mi355x)
- [Customizing Parameters](#customizing-parameters)
- [Troubleshooting](#troubleshooting)

---

## Overview

Kimi-K2 is a 1T-parameter Mixture-of-Experts (MoE) language model. Post-training support in Primus uses a Primus-owned recipe (`primus.recipes.megatron_bridge.kimi.kimi_k2`) that wraps Megatron-Bridge's `_kimi_k2_common` base with SFT-specific defaults.

Two post-training methods are supported:

| Method | Memory | Speed | Use Case |
|--------|--------|-------|----------|
| **SFT** (`peft: "none"`) | High | Slower | Maximum adaptation, all weights updated |
| **LoRA** (`peft: lora`) | Lower | Faster | Parameter-efficient, adapter modules only |

At 1T scale, full production training requires **128 nodes × 8 GPUs = 1024 GPUs** (TP=2, PP=16, EP=32).

---

## Prerequisites

- AMD ROCm drivers (≥ 7.0)
- Docker (≥ 24.0) with ROCm support
- AMD Instinct GPUs (MI300X or MI355X)
- Pre-converted Megatron checkpoint (for SFT/LoRA; not required for smoke test)

```bash
# Verify setup
rocm-smi && docker --version
```

---

## Quick Start — Smoke Test

The smoke test runs 5 iterations from random initialization on a single node (no checkpoint required). It verifies that the Kimi-K2 recipe loads correctly and training produces a finite loss.

**MI300X:**
```bash
DATA_PATH=/path/to/data \
./primus-cli container --image rocm/primus:v26.2 \
  --volume /path/to/Primus:/workspace/Primus \
  -- train posttrain \
  --config examples/megatron_bridge/configs/MI300X/kimi_k2_sft_smoke_test.yaml
```

**MI355X:**
```bash
DATA_PATH=/path/to/data \
./primus-cli container --image rocm/primus:v26.2 \
  --volume /path/to/Primus:/workspace/Primus \
  -- train posttrain \
  --config examples/megatron_bridge/configs/MI355X/kimi_k2_sft_smoke_test.yaml
```

Expected: 5 iterations complete with finite loss, exit code 0.

> **Note — required volume mount**:
> `--volume /path/to/Primus:/workspace/Primus`: The container loads Primus
> Python source from `/workspace/Primus`. Mounting the Primus checkout here
> ensures the container uses the current code (including custom recipes under
> `primus/recipes/`, `third_party/` backends, and any config changes).

Key smoke test settings (`kimi_k2_sft_smoke_test.yaml`):
- `train_iters: 5`, `global_batch_size: 8`, `seq_length: 1024`
- `pretrained_checkpoint: null` (random init)
- `checkpoint.save: null` (no checkpoint saving)
- `moe_token_dispatcher_type: allgather` (required for EP=1)
- `cuda_graph_scope: null` (avoids enum KeyError in smoke test)

---

## Pretrain

Pretrain uses Megatron-Bridge's built-in `kimi_k2_pretrain_config`. A pretrain config is not bundled with Primus examples — refer to Megatron-Bridge documentation for pretrain YAML structure.

**Container mode:**
```bash
DATA_PATH=/path/to/data \
./primus-cli container --image rocm/primus:v26.2 \
  -- train pretrain --config <pretrain-config>.yaml
```

**Slurm mode (multi-node):**
```bash
./primus-cli slurm srun -N 128 -p gpu \
  -- container --image rocm/primus:v26.2 \
  -- train pretrain --config <pretrain-config>.yaml
```

**Direct mode:**
```bash
DATA_PATH=/path/to/data \
./primus-cli direct \
  -- train pretrain --config <pretrain-config>.yaml
```

---

## Post-Training

### SFT (Full Fine-Tuning)

Full supervised fine-tuning updates all model weights. Requires a pre-converted Megatron checkpoint.

**Container mode — MI300X:**
```bash
DATA_PATH=/path/to/data \
./primus-cli container --image rocm/primus:v26.2 \
  -- train posttrain \
  --config examples/megatron_bridge/configs/MI300X/kimi_k2_sft_posttrain.yaml
```

**Container mode — MI355X:**
```bash
DATA_PATH=/path/to/data \
./primus-cli container --image rocm/primus:v26.2 \
  -- train posttrain \
  --config examples/megatron_bridge/configs/MI355X/kimi_k2_sft_posttrain.yaml
```

**Slurm mode (128 nodes):**
```bash
./primus-cli slurm srun -N 128 -p gpu \
  -- container --image rocm/primus:v26.2 \
  -- train posttrain \
  --config examples/megatron_bridge/configs/MI300X/kimi_k2_sft_posttrain.yaml
```

**Direct mode:**
```bash
DATA_PATH=/path/to/data \
./primus-cli direct \
  -- train posttrain \
  --config examples/megatron_bridge/configs/MI300X/kimi_k2_sft_posttrain.yaml
```

Key SFT settings (`kimi_k2_sft_posttrain.yaml`):
- `peft: "none"` — all parameters updated
- `finetune_lr: 5.0e-6` — lower LR than pretrain
- `global_batch_size: 4096`, `seq_length: 4096`
- `checkpoint.finetune: true` — optimizer/RNG state not restored from checkpoint

### LoRA

LoRA trains only lightweight adapter matrices, significantly reducing memory and compute requirements compared to full SFT.

**Container mode — MI300X:**
```bash
DATA_PATH=/path/to/data \
./primus-cli container --image rocm/primus:v26.2 \
  -- train posttrain \
  --config examples/megatron_bridge/configs/MI300X/kimi_k2_lora_posttrain.yaml
```

**Container mode — MI355X:**
```bash
DATA_PATH=/path/to/data \
./primus-cli container --image rocm/primus:v26.2 \
  -- train posttrain \
  --config examples/megatron_bridge/configs/MI355X/kimi_k2_lora_posttrain.yaml
```

**Slurm mode (128 nodes):**
```bash
./primus-cli slurm srun -N 128 -p gpu \
  -- container --image rocm/primus:v26.2 \
  -- train posttrain \
  --config examples/megatron_bridge/configs/MI300X/kimi_k2_lora_posttrain.yaml
```

**Direct mode:**
```bash
DATA_PATH=/path/to/data \
./primus-cli direct \
  -- train posttrain \
  --config examples/megatron_bridge/configs/MI300X/kimi_k2_lora_posttrain.yaml
```

Key LoRA settings (`kimi_k2_lora_posttrain.yaml`):
- `peft: lora` — adapter modules only
- `finetune_lr: 1.0e-4` — higher LR than SFT
- `tensor_model_parallel_size: 1` — LoRA reduces activation memory, allows lower TP
- `global_batch_size: 256`

---

## Checkpoint Conversion

Before running SFT or LoRA with real weights, convert the HuggingFace checkpoint to Megatron format. Primus ships a memory-efficient conversion script at `runner/helpers/hooks/train/posttrain/megatron_bridge/lib/convert_hf_to_megatron.py` with the following optimizations for 1T-parameter models:

- **`malloc_trim` after every `gc.collect`**: Forces glibc to return freed pages to the OS during the clone-and-free save phase, preventing OOM that kills the process around the 50% mark on large models.
- **HF model freed after weight copy**: After `bridge.to_megatron_model()` copies all weights, the HF model tensors are deleted and `malloc_trim` is called, releasing ~half the working RAM before the Megatron save begins.
- **Patches for Kimi-K2 quirks**: Skips shared-expert validation when `n_shared_experts=0` (avoids division-by-zero), forces `HAVE_NVRX=False` to bypass the incompatible `nvidia_resiliency_ext` in the container.

### Automated conversion via hook (recommended)

Set `hf_path` in the conversion test config. The `01_convert_checkpoints.sh` hook runs automatically before training and:
1. Downloads the HF model (if not already cached under `$HF_HOME/hub/`)
2. Converts to Megatron format (rank 0 only; other ranks wait on a lock file)
3. Injects the converted path as `pretrained_checkpoint` for the training run

```bash
DATA_PATH=/path/to/data \
./primus-cli container --image rocm/primus:v26.2 \
  --volume /path/to/Primus:/workspace/Primus \
  -- train posttrain \
  --config examples/megatron_bridge/configs/MI300X/kimi_k2_hf2megatron_convert_test.yaml
```

The config has `hf_path: moonshotai/Kimi-K2-Instruct` pre-set. The converted checkpoint lands at `$DATA_PATH/megatron_checkpoints/Kimi-K2-Base/`.

### Standalone conversion (no training)

Run the conversion script directly inside the container when you only want the checkpoint without running a training step:

```bash
docker run --rm \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video \
  --security-opt seccomp=unconfined \
  --ipc=host \
  -e HF_HOME=/path/to/hf_cache \
  -e PYTHONPATH=/workspace/Primus/third_party/Megatron-Bridge/src:/workspace/Primus/third_party/Megatron-Bridge/3rdparty/Megatron-LM \
  -v /path/to/Primus:/workspace/Primus \
  -v /path/to/output:/output \
  rocm/primus:v26.2 \
  python3 /workspace/Primus/runner/helpers/hooks/train/posttrain/megatron_bridge/lib/convert_hf_to_megatron.py \
    --hf-model moonshotai/Kimi-K2-Instruct \
    --megatron-path /output/megatron_checkpoints/Kimi-K2-Base \
    --trust-remote-code
```

> **Note — RAM requirements**: Converting a 1T bf16 model requires approximately **2×model_size** of RAM peak (for the HF + Megatron copies). Kimi-K2-Base at bf16 is ~2TB; plan for a host with at least 3–4TB of available RAM. The `malloc_trim` patches significantly reduce peak usage compared to the stock conversion path but cannot eliminate the fundamental requirement.

### Checkpoint Conversion Test

After conversion, validate the checkpoint loads correctly and training is numerically stable with a 500-iteration single-node run:

**Container mode — MI300X:**
```bash
DATA_PATH=/path/to/data \
./primus-cli container --image rocm/primus:v26.2 \
  --volume /path/to/Primus:/workspace/Primus \
  -- train posttrain \
  --config examples/megatron_bridge/configs/MI300X/kimi_k2_hf2megatron_convert_test.yaml
```

**Container mode — MI355X:**
```bash
DATA_PATH=/path/to/data \
./primus-cli container --image rocm/primus:v26.2 \
  --volume /path/to/Primus:/workspace/Primus \
  -- train posttrain \
  --config examples/megatron_bridge/configs/MI355X/kimi_k2_hf2megatron_convert_test.yaml
```

**Slurm mode:**
```bash
./primus-cli slurm srun -N 1 -p gpu \
  -- container --image rocm/primus:v26.2 \
  -- train posttrain \
  --config examples/megatron_bridge/configs/MI300X/kimi_k2_hf2megatron_convert_test.yaml
```

**Direct mode:**
```bash
DATA_PATH=/path/to/data \
./primus-cli direct \
  -- train posttrain \
  --config examples/megatron_bridge/configs/MI300X/kimi_k2_hf2megatron_convert_test.yaml
```

Key conversion test settings (`kimi_k2_hf2megatron_convert_test.yaml`):
- `hf_path: moonshotai/Kimi-K2-Instruct` — triggers auto-download + conversion via hook
- `train_iters: 500` — long enough to catch numerical instability
- `EP=8, num_layers=2` — single-node constraints (384 experts across 8 GPUs, reduced layers to fit HBM)
- `checkpoint.finetune: true` — loads weights only, skips optimizer state
- `checkpoint.save: null` — set to a path when you want to keep intermediate checkpoints

---

## Running Modes

### Container Mode

Recommended for environment isolation. The container loads Primus from `/workspace/Primus` internally. Mount your working copy there so the container picks up local changes (custom recipes, config fixes, etc.).

```bash
DATA_PATH=/path/to/data \
./primus-cli container --image rocm/primus:v26.2 \
  --volume /path/to/Primus:/workspace/Primus \
  -- train posttrain --config <config>.yaml
```

- `--volume /path/to/Primus:/workspace/Primus`: replaces the container's bundled Primus source with your working copy, including `primus/`, `third_party/`, and all config files.
- Add `--volume /path/to/data:/path/to/data` if your data directory is outside the repo root.

### Slurm Mode

For multi-node cluster training via Slurm. Uses `srun` or `sbatch` to allocate nodes, then launches the container on each node.

```bash
# srun (interactive)
./primus-cli slurm srun -N 128 -p <partition> \
  -- container --image rocm/primus:v26.2 \
  -- train posttrain \
  --config examples/megatron_bridge/configs/MI300X/kimi_k2_sft_posttrain.yaml

# sbatch (batch submission)
./primus-cli slurm sbatch -N 128 -p <partition> \
  -- container --image rocm/primus:v26.2 \
  -- train posttrain \
  --config examples/megatron_bridge/configs/MI300X/kimi_k2_sft_posttrain.yaml
```

### Direct Mode

Runs training directly on the host or inside an existing container. Requires ROCm and all dependencies to be installed on the host.

```bash
./primus-cli direct \
  -- train posttrain --config <config>.yaml
```

---

## Hardware Configurations

### MI300X

| Config File | Method | TP | PP | EP | Layers | GBS | MBS | Seq | Iters |
|-------------|--------|----|----|-----|--------|-----|-----|-----|-------|
| `kimi_k2_sft_posttrain.yaml` | SFT | 2 | 16 | 32 | 61 | 4096 | 1 | 4096 | 150K |
| `kimi_k2_lora_posttrain.yaml` | LoRA | 1 | 16 | 32 | 61 | 256 | 1 | 4096 | 150K |
| `kimi_k2_hf2megatron_convert_test.yaml` | SFT | 1 | 1 | 8 | 2 | 8 | 1 | 4096 | 500 |
| `kimi_k2_sft_smoke_test.yaml` | SFT | 1 | 1 | 8 | 2 | 8 | 1 | 1024 | 5 |

All configs under `examples/megatron_bridge/configs/MI300X/`.

### MI355X

| Config File | Method | TP | PP | EP | Layers | GBS | MBS | Seq | Iters |
|-------------|--------|----|----|-----|--------|-----|-----|-----|-------|
| `kimi_k2_sft_posttrain.yaml` | SFT | 2 | 16 | 32 | 61 | 4096 | 1 | 4096 | 150K |
| `kimi_k2_lora_posttrain.yaml` | LoRA | 1 | 16 | 32 | 61 | 256 | 1 | 4096 | 150K |
| `kimi_k2_hf2megatron_convert_test.yaml` | SFT | 1 | 1 | 8 | 2 | 8 | 1 | 4096 | 500 |
| `kimi_k2_sft_smoke_test.yaml` | SFT | 1 | 1 | 8 | 2 | 8 | 1 | 1024 | 5 |

All configs under `examples/megatron_bridge/configs/MI355X/`.

**Legend:** TP = Tensor Parallelism, PP = Pipeline Parallelism, EP = Expert Parallelism, Layers = `num_layers` (reduced for single-node tests), GBS = Global Batch Size, MBS = Micro Batch Size, Seq = Sequence Length

---

## Customizing Parameters

### Parallelism

```yaml
model:
  tensor_model_parallel_size: 2    # GPUs for tensor parallelism
  pipeline_model_parallel_size: 16 # GPUs for pipeline parallelism
  expert_model_parallel_size: 32   # GPUs for MoE expert parallelism
  sequence_parallel: true          # Enable sequence parallelism (requires TP > 1)
```

### Training Hyperparameters

```yaml
train:
  train_iters: 150000
  global_batch_size: 4096
  micro_batch_size: 1
  manual_gc: true
  manual_gc_interval: 5

finetune_lr: 5.0e-6   # Peak LR for SFT
min_lr: 0.0

scheduler:
  lr_warmup_iters: 2000
  lr_decay_iters: 150000
```

### Fine-Tuning Method

```yaml
peft: "none"   # Full SFT — all parameters updated
# peft: lora   # LoRA — adapter modules only
```

### Checkpoint

```yaml
checkpoint:
  pretrained_checkpoint: /path/to/megatron/checkpoint
  finetune: true       # Load weights only, not optimizer state
  save_interval: 2000
```

### W&B Logging

```yaml
wandb_project: my-project
wandb_entity: my-team
wandb_exp_name: kimi-k2-sft-run1
```

---

## Troubleshooting

### Out of Memory (OOM)

1. Increase `tensor_model_parallel_size` (2 → 4 or 8)
2. Reduce `micro_batch_size` to 1
3. Enable recompute:
   ```yaml
   model:
     recompute_granularity: full
     recompute_method: uniform
     recompute_num_layers: 1
   ```
4. Reduce `seq_length` (4096 → 2048)

### MoE dispatcher: `allgather` vs `alltoall`

EP determines the dispatcher type:

```yaml
# EP=1 (all experts on one GPU — not used in practice)
model:
  moe_token_dispatcher_type: allgather
  moe_enable_deepep: false

# EP>1, single node (smoke test, conversion test: EP=8)
model:
  moe_token_dispatcher_type: alltoall
  moe_enable_deepep: false   # deepep disabled on single node

# EP>1, multi-node production (EP=32)
model:
  moe_token_dispatcher_type: alltoall
  moe_enable_deepep: true
```

The smoke test and conversion test use EP=8 + `alltoall` + `deepep: false`. Production SFT/LoRA use EP=32 + `alltoall` + `deepep: true`.

### `KeyError` in `CudaGraphScope` enum

Set `cuda_graph_scope: null` (not `""`) in smoke/test configs. The empty string triggers an enum lookup failure:

```yaml
model:
  cuda_graph_impl: "none"
  cuda_graph_scope: null
```

### Training Instability

1. Verify `finetune_lr` is low enough (SFT: 5e-6, LoRA: 1e-4)
2. Increase `lr_warmup_iters` (try 500–1000)
3. Confirm `checkpoint.finetune: true` so optimizer state is not restored from pretrain

---

**Need help?** Open an issue on [GitHub](https://github.com/AMD-AIG-AIMA/Primus/issues).
