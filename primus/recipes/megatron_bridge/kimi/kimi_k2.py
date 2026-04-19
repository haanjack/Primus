###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Primus-owned SFT recipe for Kimi-K2 (1T MoE).

Megatron-Bridge provides kimi_k2_pretrain_config but no finetune flavor.
This module adds kimi_k2_finetune_config for full supervised fine-tuning
(no PEFT), built on top of kimi_k2_pretrain_config from Megatron-Bridge.

To add further Kimi-K2 recipes (e.g., LoRA once supported), extend this file
and export from primus/recipes/megatron_bridge/kimi/__init__.py.
"""

from typing import Optional

from megatron.bridge.recipes.kimi.kimi_k2 import kimi_k2_pretrain_config
from megatron.bridge.training.config import ConfigContainer


def kimi_k2_finetune_config(
    optimizer_type: str = "muon",
    pretrained_checkpoint: Optional[str] = None,
    finetune_lr: float = 5e-6,
    wandb_project: Optional[str] = None,
    wandb_entity: Optional[str] = None,
    wandb_exp_name: Optional[str] = None,
) -> ConfigContainer:
    """Full SFT config for Kimi-K2 (1T MoE). No PEFT — all weights updated.

    Recommended parallelism for 128-node (1024 GPU) cluster:
        TP=2, PP=16, EP=32

    Built on kimi_k2_pretrain_config with finetune-specific overrides:
    - Much lower default LR (5e-6 vs 3e-4) via ``finetune_lr``
    - checkpoint.finetune=True so optimizer/RNG state is not restored
    - Shorter default training run (1k iters vs 1M)
    - Optional W&B logging fields

    Args:
        optimizer_type: 'adam' or 'muon' (default).
        pretrained_checkpoint: Path to a converted Megatron checkpoint to
            fine-tune from. If None, trains from random init (smoke test).
        finetune_lr: Peak learning rate. Defaults to 5e-6.
        wandb_project: W&B project name (optional).
        wandb_entity: W&B entity/team name (optional).
        wandb_exp_name: W&B experiment name (optional).
    """
    cfg: ConfigContainer = kimi_k2_pretrain_config(optimizer_type=optimizer_type)

    # Lower LR for fine-tuning (pretrain uses 3e-4)
    cfg.optimizer.lr = finetune_lr
    cfg.optimizer.min_lr = 0.0

    # Shorter default run for fine-tuning
    cfg.train.train_iters = 1_000
    cfg.scheduler.lr_warmup_iters = 200
    cfg.scheduler.lr_decay_iters = 1_000

    # Finetune-specific checkpoint settings
    cfg.checkpoint.finetune = True
    cfg.checkpoint.pretrained_checkpoint = pretrained_checkpoint

    # W&B logging (applied only when at least one field is set)
    if wandb_project or wandb_entity or wandb_exp_name:
        cfg.logger.wandb_project = wandb_project
        cfg.logger.wandb_entity = wandb_entity
        cfg.logger.wandb_exp_name = wandb_exp_name

    return cfg
