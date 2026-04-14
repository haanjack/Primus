###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Unit tests for Kimi-K2.5-VL config files.

Validates that all kimi_k25_vl YAML configs load without error and produce
the expected module structure. Catches typos, missing keys, and invalid YAML
before running on the remote GPU server.
"""

from pathlib import Path

import pytest

from primus.core.config.primus_config import get_module_config, load_primus_config

EXAMPLES_DIR = Path(__file__).parents[4] / "examples" / "megatron_bridge" / "configs"

KIMI_CONFIGS = [
    EXAMPLES_DIR / "MI300X" / "kimi_k25_vl-sft.yaml",
    EXAMPLES_DIR / "MI300X" / "kimi_k25_vl-sft-test.yaml",
    EXAMPLES_DIR / "MI300X" / "kimi_k25_vl-sft-toy-smoke.yaml",
    EXAMPLES_DIR / "MI355X" / "kimi_k25_vl-sft.yaml",
    EXAMPLES_DIR / "MI355X" / "kimi_k25_vl-sft-test.yaml",
]


@pytest.mark.parametrize("config_path", KIMI_CONFIGS, ids=lambda p: p.name)
def test_config_loads_without_error(config_path):
    """Config parses to a SimpleNamespace without raising."""
    cfg = load_primus_config(config_path, cli_args=None)
    assert cfg is not None


@pytest.mark.parametrize("config_path", KIMI_CONFIGS, ids=lambda p: p.name)
def test_config_has_post_trainer_module(config_path):
    """Each config must have a post_trainer module."""
    cfg = load_primus_config(config_path, cli_args=None)
    post_trainer = get_module_config(cfg, "post_trainer")
    assert post_trainer is not None, f"post_trainer module missing in {config_path.name}"


@pytest.mark.parametrize("config_path", KIMI_CONFIGS, ids=lambda p: p.name)
def test_config_uses_megatron_bridge_framework(config_path):
    """All kimi configs must use the megatron_bridge framework."""
    cfg = load_primus_config(config_path, cli_args=None)
    post_trainer = get_module_config(cfg, "post_trainer")
    assert hasattr(post_trainer, "framework"), "post_trainer.framework missing"
    assert post_trainer.framework == "megatron_bridge"


def test_toy_smoke_config_has_hf_path_override():
    """Toy smoke config must override hf_path to the local toy model path."""
    config_path = EXAMPLES_DIR / "MI300X" / "kimi_k25_vl-sft-toy-smoke.yaml"
    cfg = load_primus_config(config_path, cli_args=None)
    post_trainer = get_module_config(cfg, "post_trainer")
    assert hasattr(post_trainer, "params"), "post_trainer.params missing"
    assert hasattr(post_trainer.params, "hf_path"), "hf_path not in params"
    assert "kimi_k25_vl_toy" in post_trainer.params.hf_path, (
        f"hf_path should point to toy model, got: {post_trainer.params.hf_path}"
    )


def test_toy_smoke_config_ep1_dispatcher_settings():
    """Toy smoke config must use allgather dispatcher (not alltoall/deepep) for EP=1."""
    config_path = EXAMPLES_DIR / "MI300X" / "kimi_k25_vl-sft-toy-smoke.yaml"
    cfg = load_primus_config(config_path, cli_args=None)
    post_trainer = get_module_config(cfg, "post_trainer")
    params = post_trainer.params
    # Check nested model overrides (these are what _merge_dict_to_dataclass uses)
    assert params.model.expert_model_parallel_size == 1, "Smoke test should use EP=1"
    assert params.model.moe_token_dispatcher_type == "allgather", (
        "EP=1 requires allgather dispatcher, not alltoall"
    )
    assert params.model.moe_flex_dispatcher_backend == "allgather", (
        "EP=1 requires allgather backend, not deepep"
    )
    assert params.model.moe_enable_deepep is False, "deepep requires EP>1"


def test_toy_smoke_config_minimal_iters():
    """Toy smoke config must use 5 training iterations (not 500 or 150000)."""
    config_path = EXAMPLES_DIR / "MI300X" / "kimi_k25_vl-sft-toy-smoke.yaml"
    cfg = load_primus_config(config_path, cli_args=None)
    post_trainer = get_module_config(cfg, "post_trainer")
    # train_iters is now under the nested train: sub-config
    assert post_trainer.params.train.train_iters == 5


def test_toy_smoke_config_nested_model_overrides():
    """Verify nested overrides exist in parsed config (pre-ConfigContainer).

    These nested dicts are what _merge_dict_to_dataclass uses to set fields
    on ConfigContainer.model, ConfigContainer.train, etc. Flat top-level keys
    would be silently ignored.
    """
    config_path = EXAMPLES_DIR / "MI300X" / "kimi_k25_vl-sft-toy-smoke.yaml"
    cfg = load_primus_config(config_path, cli_args=None)
    post_trainer = get_module_config(cfg, "post_trainer")
    params = post_trainer.params

    # model overrides must be nested under params.model
    assert hasattr(params, "model"), "model overrides missing from params"
    assert params.model.tensor_model_parallel_size == 1
    assert params.model.pipeline_model_parallel_size == 1
    assert params.model.expert_model_parallel_size == 1

    # train overrides must be nested under params.train
    assert hasattr(params, "train"), "train overrides missing from params"
    assert params.train.train_iters == 5
    assert params.train.global_batch_size == 4

    # checkpoint overrides must be nested under params.checkpoint
    assert hasattr(params, "checkpoint"), "checkpoint overrides missing from params"

    # scheduler overrides must be nested under params.scheduler
    assert hasattr(params, "scheduler"), "scheduler overrides missing from params"
    assert params.scheduler.lr_warmup_iters == 0
