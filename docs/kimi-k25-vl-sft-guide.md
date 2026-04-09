# Kimi-K2.5-VL Supervised Fine-Tuning Guide

This guide covers the complete workflow for fine-tuning the Kimi-K2.5-VL model using Primus-CLI, including checkpoint conversion, training, monitoring, and evaluation.

**Table of Contents:**
1. [Overview & Architecture](#overview--architecture)
2. [Prerequisites](#prerequisites)
3. [Checkpoint Conversion (HF → Megatron)](#checkpoint-conversion-hf--megatron)
4. [Training Execution](#training-execution)
5. [Monitoring & Logging](#monitoring--logging)
6. [Post-Training Checkpoint Export](#post-training-checkpoint-export)
7. [Inference & Evaluation](#inference--evaluation)
8. [Troubleshooting](#troubleshooting)

---

## Overview & Architecture

### Model Background

**Kimi-K2.5-VL** is a multimodal large language model by Moonshot AI with the following characteristics:

- **Total Parameters**: ~1 Trillion (1T)
- **Active Parameters**: ~27B (very sparse due to MoE)
- **Architecture**: Mixture-of-Experts (MoE) with 384 experts
- **Modality**: Vision-Language model (processes both text and images)
- **Key Features**:
  - Multi-Latent Attention (MLA) for efficient attention
  - Multi-Token Prediction (MTP) for faster generation
  - 61 transformer layers with 7,168 hidden dimensions
  - Rotary positional embeddings with YaRN scaling (4K → extended context)
  - Device-limited routing for stable multi-node training

### Training Architecture in Primus

The Primus-CLI integration uses the following stack:

```
User Command (primus-cli)
    ↓
+─────────────────────────────────────────────+
│ Primus Runtime (CLI orchestration)          │
│ - Config loading & extends resolution       │
│ - Environment setup (distributed training)  │
│ - Hook execution (pre/post training)        │
+─────────────────────────────────────────────+
    ↓
+─────────────────────────────────────────────+
│ Megatron-Bridge Backend                     │
│ - Checkpoint conversion hooks                │
│ - Model loading (HF → Megatron)             │
│ - Distributed training coordination         │
+─────────────────────────────────────────────+
    ↓
+─────────────────────────────────────────────+
│ Training Engine                              │
│ - Megatron-LM training loop                 │
│ - Tensor/Pipeline/Expert parallelism        │
│ - Checkpoint save/load                      │
+─────────────────────────────────────────────+
```

### Parallelism Strategy

For **Full Kimi-K2.5-VL SFT**:

| Strategy | Value | Explanation |
|----------|-------|-------------|
| Tensor Parallel (TP) | 2 | Partition weight matrices across 2 GPUs |
| Pipeline Parallel (PP) | 16 | Split model into 16 stages across depth |
| Expert Parallel (EP) | 32 | Distribute 384 experts across 32 GPUs |
| **Total GPUs** | **1,024** | TP×PP×EP = 2×16×32 |
| **Node Count** | **128** | 1,024 GPUs ÷ 8 GPUs/node |
| **TP Group Size** | 2 | TP communication overhead |
| **PP Stage Size** | 16 | Pipeline bubble (mitigated with careful schedule) |

**Batch Size Decomposition** (Global Batch = 4,096):
- Gradient Accumulation Steps = 4,096 / (1,024 × 1 micro-batch size) = 4 steps per gradient update

---

## Prerequisites

### Required Resources

**Hardware**:
- Minimum 1,024 GPUs (128 nodes with 8 GPUs each)
- AMD MI355X or MI300X recommended (or equivalent)
- 512 GB total GPU memory (0.5 TB per GPU)
- High-speed interconnect (InfiniBand preferred for multi-node)

**Storage**:
- **Model Checkpoint**: ~2TB (full Kimi-K2.5 HuggingFace model)
- **Converted Megatron Checkpoint**: ~2TB
- **Training Outputs**: ~500GB-1TB (logs, checkpoints, intermediate saves)

**Time Estimate**:
- Full model conversion: 2-4 hours (multi-GPU, multi-node)
- Training (150K iterations at 4K batch size): 1-2 weeks (depending on cluster performance)

### Required Credentials & Setup

1. **HuggingFace Token**:
   ```bash
   export HF_TOKEN="hf_xxxxxxxxxxxxx"
   ```
   Get token from https://huggingface.co/settings/tokens

2. **HuggingFace Home** (optional, for caching):
   ```bash
   export HF_HOME=/path/to/hf_cache  # Default: ~/.cache/huggingface
   ```

3. **Weights & Biases** (optional, for experiment tracking):
   ```bash
   export WANDB_API_KEY="your_wandb_key"
   ```

4. **Cluster Access**:
   - SLURM account and partition
   - Container runtime (Podman/Docker)
   - Network access to HuggingFace Hub

### Software Dependencies

All dependencies are included in the Primus container image. If running locally:

```bash
pip install -r requirements.txt

# Additional for checkpoint conversion:
pip install -r requirements-jax.txt  # If using JAX for conversion
```

**Key Package Versions**:
- Python 3.10+
- PyTorch 2.0+
- Transformers 4.40+
- Megatron-LM (from third_party/Megatron-Bridge)

---

## Checkpoint Conversion (HF → Megatron)

### Automatic Conversion (Recommended)

When you run `primus-cli train posttrain` with a Kimi model, the checkpoint conversion happens **automatically** during setup:

1. **Rank 0 Process** downloads the HF model and converts it
2. **Other Ranks** wait for conversion to complete (via lock files)
3. **Result** is saved to `data/megatron_checkpoints/`
4. **Training** begins with the converted checkpoint

This is handled by the pre-training hook:
```
runner/helpers/hooks/train/posttrain/megatron_bridge/01_convert_checkpoints.sh
```

**No manual steps required** — just ensure `HF_TOKEN` is set and let Primus handle it.

### Manual Multi-GPU Conversion (Advanced)

If you need to pre-convert the checkpoint separately:

#### Step 1: Prepare Conversion Environment

```bash
cd /path/to/Primus
export HF_TOKEN="hf_xxxxxxxxxxxxx"
export CHECKPOINT_DIR="/path/to/checkpoints"  # Destination
mkdir -p "$CHECKPOINT_DIR"
```

#### Step 2: Run Multi-GPU Conversion

For full Kimi-K2.5-VL (TP=8, EP=8, PP=4 parallelism for conversion):

```bash
# On a machine with 64+ GPUs or use srun for multi-node
torchrun --nproc_per_node=8 \
  third_party/Megatron-Bridge/examples/conversion/convert_checkpoints_multi_gpu.py import \
    --hf-model moonshotai/Kimi-K2.5 \
    --megatron-path "$CHECKPOINT_DIR/kimi_k25_vl_megatron" \
    --tp 8 \
    --ep 8 \
    --pp 4 \
    --trust_remote_code
```

**Parameters**:
- `--tp 8`: Tensor parallelism for conversion
- `--ep 8`: Expert parallelism for conversion
- `--pp 4`: Pipeline parallelism for conversion
- `--trust_remote_code`: Required for proprietary model code

#### Step 3: Verify Conversion

```bash
# Check output structure
ls -lh "$CHECKPOINT_DIR/kimi_k25_vl_megatron/"

# Expected output:
# - metadata.json
# - iter_0000000/
#   ├── mp_rank_00/
#   ├── mp_rank_01/
#   └── ... (more ranks)
```

#### Step 4: Point Config to Converted Checkpoint

In your experiment config (e.g., `kimi_k25_vl-sft.yaml`), set:

```yaml
modules:
  post_trainer:
    overrides:
      pretrained_checkpoint: /path/to/kimi_k25_vl_megatron/iter_0000000
```

### Output Structure & Naming

**Auto-converted checkpoints** are saved to:
```
data/megatron_checkpoints/
├── kimi_k25_vl/
│   ├── metadata.json
│   └── iter_0000000/
```

**Manually converted** (if specified):
```
/your/checkpoint_dir/kimi_k25_vl_megatron/
├── metadata.json
└── iter_0000000/
```

Each saved checkpoint contains:
- Model weights in TP/PP/EP distributed format
- Optimizer state (if available)
- Metadata about model architecture and parallelism

---

## Training Execution

### Option 1: Container Mode (Recommended for Single Node)

Use Docker/Podman container for isolated environment:

```bash
# Set up environment variables
export HF_TOKEN="hf_xxxxxxxxxxxxx"
export CONFIG_FILE="examples/megatron_bridge/configs/MI355X/kimi_k25_vl-sft-test.yaml"

# Run training script
./examples/megatron_bridge/scripts/kimi_k25_vl_sft_container.sh
```

**What the script does**:
1. Validates HF_TOKEN is set
2. Mounts workspace and cache directories
3. Starts container with Primus image
4. Runs `primus-cli train posttrain` inside container
5. Handles checkpoint conversion automatically

**Customize via environment variables**:

```bash
# Use full model config instead of test
export CONFIG_FILE="examples/megatron_bridge/configs/MI355X/kimi_k25_vl-sft.yaml"

# Specify container image
export CONTAINER_IMAGE="rocm/primus:latest"

# Add extra volume mounts
export CONTAINER_MOUNTS="-v /data:/data -v /scratch:/scratch"

# Run
./examples/megatron_bridge/scripts/kimi_k25_vl_sft_container.sh
```

### Option 2: SLURM Mode (Recommended for Multi-Node)

Submit to SLURM cluster for multi-node training:

```bash
# Set environment variables
export HF_TOKEN="hf_xxxxxxxxxxxxx"
export SLURM_NODES=128
export SLURM_ACCOUNT="your_account"
export SLURM_PARTITION="batch"

# Submit job
sbatch examples/megatron_bridge/scripts/kimi_k25_vl_sft_slurm.sh

# Monitor job
squeue -j <job_id>
tail -f slurm_logs/<jobname>_<jobid>.out
```

**Customize SLURM parameters**:

Edit the script header or pass via environment:

```bash
# Reduce node count for testing
export SLURM_NODES=4
sbatch examples/megatron_bridge/scripts/kimi_k25_vl_sft_slurm.sh

# Change partition
export SLURM_PARTITION="debug"
sbatch examples/megatron_bridge/scripts/kimi_k25_vl_sft_slurm.sh

# Request specific time limit
export SLURM_TIMELIMIT=48:00:00
sbatch examples/megatron_bridge/scripts/kimi_k25_vl_sft_slurm.sh
```

**SLURM Job Script** customization:
Edit `examples/megatron_bridge/scripts/kimi_k25_vl_sft_slurm.sh`:

```bash
#SBATCH --nodes=128           # Change number of nodes
#SBATCH --ntasks-per-node=8   # 8 GPUs per node (don't change if 8 GPUs/node)
#SBATCH --time=120:00:00      # Max runtime hours
#SBATCH --mail-user=you@domain.com  # Email notifications
```

### Option 3: Direct Mode (For Development/Testing)

Run directly without container on current machine:

```bash
# Single node, 8 GPUs
./primus-cli direct train posttrain \
  --config examples/megatron_bridge/configs/MI355X/kimi_k25_vl-sft-test.yaml

# Multi-node with environment variables
export NNODES=4
export GPUS_PER_NODE=8
export NODE_RANK=0
export MASTER_ADDR=host1
export MASTER_PORT=1234

./primus-cli direct train posttrain \
  --config examples/megatron_bridge/configs/MI355X/kimi_k25_vl-sft.yaml
```

### Configuration Selection

**For Testing/Development** (quick iteration, 1 node):
```bash
CONFIG="examples/megatron_bridge/configs/MI355X/kimi_k25_vl-sft-test.yaml"
# TP=1, PP=1, EP=1 | 500 iterations | batch size 8
# Runs in < 30 minutes on 1 node with 8 GPUs
```

**For Full-Scale Training** (128 nodes):
```bash
CONFIG="examples/megatron_bridge/configs/MI355X/kimi_k25_vl-sft.yaml"
# TP=2, PP=16, EP=32 | 150K iterations | batch size 4,096
# Requires 1,024 GPUs, estimated 1-2 weeks training time
```

**For CI/Development** (8-16 GPUs):
Create a custom config with reduced parallelism:
```yaml
# examples/megatron_bridge/configs/MI355X/kimi_k25_vl-sft-small.yaml
extends: "kimi_k25_vl-sft.yaml"
modules:
  post_trainer:
    overrides:
      tensor_model_parallel_size: 1
      pipeline_model_parallel_size: 1
      expert_model_parallel_size: 1
      train_iters: 100
      global_batch_size: 16
      save_interval: 50
```

---

## Monitoring & Logging

### Real-Time Training Metrics

**Tensorboard** (automatic):
```bash
# In separate terminal
tensorboard --logdir output/exp_pretrain/tensorboard/

# Open browser: http://localhost:6006
```

**Displayed Metrics**:
- Learning rate schedule
- Loss curves (training + validation)
- Throughput (tokens/sec)
- GPU memory usage
- Gradient norms

### Logs Location

**Training Logs**:
```
output/exp_pretrain/logs/
├── rank_0.log          # Rank 0 (main) detailed logs
├── rank_1.log
└── ...

output/exp_pretrain/
├── tensorboard/        # Tensorboard data
└── checkpoints/        # Saved model checkpoints
    ├── iter_0000002000/
    ├── iter_0000004000/
    ...
```

**Checkpoint Contents**:
```
iter_XXXXXXXXX/
├── mp_rank_00/         # Tensor parallel rank 0
│   ├── optimizer.pt
│   └── model_states.pt
├── mp_rank_01/
├── ...
├── latest
├── latest_checkpointed_iteration.txt
```

### Weights & Biases Integration (Optional)

Enable WandB tracking for experiment management:

1. **Set WandB credentials**:
   ```bash
   export WANDB_API_KEY="your_key"
   ```

2. **Configure in experiment YAML**:
   ```yaml
   modules:
     post_trainer:
       overrides:
         wandb_project: "kimi-finetuning"
         wandb_entity: "your_team"
         wandb_exp_name: "kimi_k25_sft_run1"
   ```

3. **Access dashboard**:
   - Go to https://wandb.ai/your_team/kimi-finetuning
   - View live training curves, compare runs, log hyperparameters

### Common Metrics to Monitor

| Metric | Healthy Range | Action if Out of Range |
|--------|--------------|----------------------|
| **Loss** | Decreasing over time | Check data, learning rate |
| **Learning Rate** | Follows schedule (warmup→peak→decay) | Verify lr_decay_iters |
| **GPU Memory** | 75-90% utilized | Increase batch size or reduce micro_batch_size |
| **Throughput** | >1000 tokens/sec (TP=2, PP=16) | Check for comm bottleneck |
| **Gradient Norm** | Stable, no extreme spikes | May indicate data issue |
| **Perplexity** | Decreasing | Check loss, data quality |

---

## Post-Training Checkpoint Export

After training completes, convert the Megatron checkpoint back to HuggingFace format for inference/deployment.

### Export Megatron → HuggingFace

#### Automatic Export (Recommended)

The conversion can be done automatically using the same checkpoint conversion tool:

```bash
export HF_TOKEN="hf_xxxxxxxxxxxxx"
export MEGATRON_CKPT="output/exp_pretrain/checkpoints/iter_0150000"
export HF_OUTPUT_DIR="./kimi_k25_vl_sft_ckpt"

# Single GPU export (works for any parallelism)
python3 third_party/Megatron-Bridge/examples/conversion/convert_checkpoints.py export \
  --hf-model moonshotai/Kimi-K2.5 \
  --megatron-path "$MEGATRON_CKPT" \
  --hf-path "$HF_OUTPUT_DIR" \
  --trust_remote_code
```

**Parameters**:
- `--hf-model`: Original HF model (for config)
- `--megatron-path`: Trained Megatron checkpoint directory
- `--hf-path`: Output directory for HF-format checkpoint

#### Multi-GPU Export (For Large Checkpoints)

If conversion is slow or memory-constrained:

```bash
torchrun --nproc_per_node=8 \
  third_party/Megatron-Bridge/examples/conversion/convert_checkpoints_multi_gpu.py export \
    --hf-model moonshotai/Kimi-K2.5 \
    --megatron-path "$MEGATRON_CKPT" \
    --hf-path "$HF_OUTPUT_DIR" \
    --tp 2 --ep 8  # Match or coarsen from training parallelism
```

### Verify Exported Checkpoint

```bash
# Check structure
ls -lh "$HF_OUTPUT_DIR/"

# Expected:
# - config.json
# - generation_config.json
# - model.safetensors (or pytorch_model.bin)
# - tokenizer.model (or tokenizer files)
# - special_tokens_map.json
# - tokenizer_config.json

# Test loading in Python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("$HF_OUTPUT_DIR", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("$HF_OUTPUT_DIR", trust_remote_code=True)
print("✓ Checkpoint exports successfully!")
```

### Optional: Merge LoRA Adapters

If you used LoRA fine-tuning (not standard for full SFT), merge adapters:

```bash
# Requires peft library
from peft import AutoPeftModelForCausalLM

model = AutoPeftModelForCausalLM.from_pretrained("$HF_OUTPUT_DIR")
merged_model = model.merge_and_unload()
merged_model.save_pretrained("$HF_OUTPUT_DIR/merged")
```

### Push to HuggingFace Hub (Optional)

Share your fine-tuned model publicly:

```bash
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("$HF_OUTPUT_DIR", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("$HF_OUTPUT_DIR")

# Push to Hub
model.push_to_hub("your_org/kimi_k25_vl_sft")
tokenizer.push_to_hub("your_org/kimi_k25_vl_sft")
```

---

## Inference & Evaluation

### Generate from Fine-Tuned Model

#### Using HuggingFace Transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load fine-tuned model
model = AutoModelForCausalLM.from_pretrained(
    "./kimi_k25_vl_sft_ckpt",
    trust_remote_code=True,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(
    "./kimi_k25_vl_sft_ckpt",
    trust_remote_code=True
)

# Generate text
prompt = "Explain machine learning in simple terms:"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(
    **inputs,
    max_length=512,
    temperature=0.7,
    top_p=0.95,
    num_return_sequences=1
)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

#### Using vLLM (For Faster Inference)

```bash
# Start vLLM server
python -m vllm.entrypoints.openai.api_server \
  --model ./kimi_k25_vl_sft_ckpt \
  --trust-remote-code \
  --tensor-parallel-size 2  # Adjust to your setup
```

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="token-abc123"
)

response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Explain quantum computing"}]
)
print(response.choices[0].message.content)
```

### Evaluation on Benchmarks

#### MMLU (Massive Multi-task Language Understanding)

```bash
# Using lm-evaluation-harness
python -m lm_eval \
  --model hf-causal-experimental \
  --model_args pretrained="./kimi_k25_vl_sft_ckpt",trust_remote_code=True \
  --tasks mmlu \
  --device cuda:0 \
  --batch_size 4 \
  --num_fewshot 5
```

#### Custom Task Evaluation

```python
# Evaluate on your custom test set
import json
from transformers import pipeline

generator = pipeline("text-generation", model="./kimi_k25_vl_sft_ckpt")

with open("test_questions.jsonl") as f:
    correct = 0
    for line in f:
        q = json.loads(line)
        pred = generator(q["prompt"], max_length=256)[0]["generated_text"]
        if q["expected"] in pred:
            correct += 1

acc = correct / 100 * 100
print(f"Accuracy: {acc:.2f}%")
```

### Vision-Language Evaluation (For VLM Capabilities)

```python
from PIL import Image
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "./kimi_k25_vl_sft_ckpt",
    trust_remote_code=True,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(
    "./kimi_k25_vl_sft_ckpt"
)

# Load and process image
image = Image.open("sample.jpg")
prompt = "Describe this image in detail:"

# Process through model's vision encoder
inputs = tokenizer(prompt, images=[image], return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_length=256)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

---

## Troubleshooting

### Checkpoint Conversion Issues

#### "Model not found on HuggingFace Hub"

**Symptom**: Error during auto-conversion: `Could not find model moonshotai/Kimi-K2.5`

**Solutions**:
1. Verify HF_TOKEN is set and valid
2. Check internet connectivity to HF Hub
3. Use manual conversion with pre-downloaded model

#### "Out of Memory during conversion"

**Symptom**: OOM on rank 0 during HF→Megatron conversion

**Solutions**:
1. Reduce TP/EP for conversion: `--tp 4 --ep 4 --pp 2`
2. Enable gradient checkpointing: `--gradient_checkpointing`
3. Use multi-node for conversion: `srun -N 4 torchrun...`

#### "Checkpoint validation failed"

**Symptom**: Error: `metadata.json mismatch` or `Shape mismatch`

**Solutions**:
1. Verify source model matches config
2. Ensure correct TP/EP/PP used
3. Check disk space for output

---

### Training Execution Issues

#### NCCL Timeout During Training

**Symptom**: `NCCL operation timed out` after N iterations

**Solutions**:
1. Increase timeout in config:
   ```yaml
   distributed_timeout_minutes: 120  # Increase from default 60
   ```

2. Check network connectivity
   ```bash
   # Verify NCCL IB connection
   /opt/nccl_tests/build/all_reduce_perf -b 8 -e 1G -f 2 -g <ngpus>
   ```

3. Reduce communication overlap
   ```yaml
   tp_comm_overlap: false
   overlap_grad_reduce: false
   ```

#### OOM During Training

**Symptom**: CUDA out of memory after N iterations

**Solutions**:
1. Reduce micro_batch_size
2. Enable recompute (already enabled by default)
3. Reduce sequence_length
4. Increase expert_model_parallel_size (EP)

```yaml
# Example fix
micro_batch_size: 1  # Already minimal
seq_length: 2048     # Reduce from 4096
recompute_granularity: full  # Verify this is set
```

#### Loss Not Decreasing

**Symptom**: Loss plateaus or increases over time

**Solutions**:
1. Check learning rate schedule
   ```bash
   # Inspect in tensorboard
   # Verify warmup → peak → decay progression
   ```

2. Verify data loading
   ```bash
   # Check first batch
   python -c "
   from transformers import AutoTokenizer
   tok = AutoTokenizer.from_pretrained('moonshotai/Kimi-K2.5')
   print(tok.decode(tok.encode('hello world')))
   "
   ```

3. Reduce learning rate if diverging
   ```yaml
   finetune_lr: 1.0e-4  # From 3.0e-4
   ```

#### Multi-Node Synchronization Error

**Symptom**: "rank 0 finished but rank 1-7 still running"

**Solutions**:
1. Check MASTER_ADDR/PORT are correct
   ```bash
   srun env | grep MASTER
   ```

2. Ensure all nodes can ping each other
   ```bash
   srun -N 2 ping -c 1 $MASTER_ADDR
   ```

3. Increase ranked watchdog timeout
   ```yaml
   distributed_timeout_minutes: 120
   ```

---

### Monitoring & Logging Issues

#### Tensorboard Not Updating

**Symptom**: Tensorboard dashboard shows no new data

**Solutions**:
1. Check tensorboard directory exists
   ```bash
   ls -la output/exp_pretrain/tensorboard/
   ```

2. Verify training is writing events
   ```bash
   find output/exp_pretrain/tensorboard -name "events*" -mmin 1
   ```

3. Restart tensorboard server
   ```bash
   # Kill old process
   pkill -f tensorboard
   # Start new
   tensorboard --logdir output/exp_pretrain/tensorboard/
   ```

#### Log Files Not Being Generated

**Symptom**: No rank_*.log files in output directory

**Solutions**:
1. Check stderr_sink_level setting
   ```yaml
   stderr_sink_level: DEBUG  # Allow debug messages
   ```

2. Verify output directory permissions
   ```bash
   ls -ld output/exp_pretrain/logs/
   chmod 755 output/exp_pretrain/logs/
   ```

3. Check for log file locks
   ```bash
   lsof output/exp_pretrain/logs/rank_0.log
   ```

---

### Checkpoint Export Issues

#### "Config mismatch" during export

**Symptom**: Error during HF export: `Config for model failed validation`

**Solutions**:
1. Use exact same HF model during export
   ```bash
   --hf-model moonshotai/Kimi-K2.5  # Must match import source
   ```

2. Verify checkpoint architecture matches config
   ```python
   import json
   with open("megatron_ckpt/metadata.json") as f:
       meta = json.load(f)
   print(meta["model_name"])
   ```

#### Exported model doesn't generate correctly

**Symptom**: Generation produces nonsense or incomplete text

**Solutions**:
1. Verify tokenizer consistency
   ```python
   from transformers import AutoTokenizer
   tok = AutoTokenizer.from_pretrained("./exported_ckpt")
   print(len(tok))  # Should have >100k tokens
   ```

2. Check generation config
   ```python
   from transformers import AutoConfig
   cfg = AutoConfig.from_pretrained("./exported_ckpt")
   print(cfg.to_dict().keys())
   ```

3. Test with original model first
   ```python
   orig = AutoModel.from_pretrained("moonshotai/Kimi-K2.5")
   # Compare outputs
   ```

---

### Getting Help

If issues persist:

1. **Check Logs**: Review `output/exp_pretrain/logs/rank_0.log` for detailed error traces

2. **Verify Environment**:
   ```bash
   # Run diagnostic
   echo "HF_TOKEN: ${HF_TOKEN:0:10}..."
   echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
   nvidia-smi
   python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')"
   ```

3. **Test Incrementally**:
   - Start with test config (1 node, 500 iters)
   - Verify checkpoint conversion separately
   - Scale to full training progressively

4. **Community Resources**:
   - Primus Issues: https://github.com/rocm/Primus/issues
   - Megatron-LM Discussions: Megatron GitHub
   - Kimi Model: https://huggingface.co/moonshotai/Kimi-K2.5

---

## Summary: Complete Workflow

### Quick Start (5 minutes)

```bash
# 1. Set credentials
export HF_TOKEN="hf_xxxxx"

# 2. Test on single node
./primus-cli direct train posttrain \
  --config examples/megatron_bridge/configs/MI355X/kimi_k25_vl-sft-test.yaml

# 3. Monitor (in another terminal)
tensorboard --logdir output/*/tensorboard/
```

### Production Workflow (Full Training)

```bash
# 1. Submit to cluster
sbatch examples/megatron_bridge/scripts/kimi_k25_vl_sft_slurm.sh

# 2. Monitor progress
squeue -j <job_id>
tail -f logs/kimi_k25_vl_sft_<jobid>.out

# 3. After training completes
python third_party/Megatron-Bridge/examples/conversion/convert_checkpoints.py export \
  --hf-model moonshotai/Kimi-K2.5 \
  --megatron-path output/*/checkpoints/iter_0150000 \
  --hf-path ./kimi_k25_vl_sft_ckpt

# 4. Test inference
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained('./kimi_k25_vl_sft_ckpt', trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained('./kimi_k25_vl_sft_ckpt')
# Generate from model...
"
```

---

## Additional Resources

- **Kimi Model Card**: https://huggingface.co/moonshotai/Kimi-K2.5
- **Megatron-Bridge Docs**: third_party/Megatron-Bridge/README.md
- **Primus Documentation**: docs/README.md
- **Megatron-LM**: https://github.com/NVIDIA/Megatron-LM
