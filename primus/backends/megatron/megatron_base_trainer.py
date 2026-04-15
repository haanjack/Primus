###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from primus.backends.megatron.training.global_vars import set_primus_global_variables
from primus.core.trainer.base_trainer import BaseTrainer
from primus.modules.module_utils import log_rank_0


class MegatronBaseTrainer(BaseTrainer):
    """Base trainer for Megatron-LM, handles parse_args patching."""

    def setup(self):
        """Setup Megatron runtime: set global vars and patch parse_args."""
        set_primus_global_variables(self.backend_args)
        self._set_megatron_global_variables()
        self._patch_parse_args()

    def init(self):
        """Initialize Megatron training components."""
        log_rank_0("Initializing Megatron training...")
        # log_dict_aligned("Backend arguments", self.backend_args)

    def _set_megatron_global_variables(self):
        """Initialize Megatron's own global variables.

        In Megatron-LM >= 0.18, ``parse_and_validate_args()`` calls
        ``validate_args()`` then ``set_global_variables()`` to populate
        ``megatron.training.global_vars``.  Primus bypasses that path by
        calling ``pretrain()`` directly, so we replicate the same
        initialization here.
        """
        from megatron.training.arguments import validate_args  # type: ignore
        from megatron.training.global_vars import set_global_variables  # type: ignore

        log_rank_0("Validating Megatron args and setting global variables")
        validate_args(self.backend_args)
        set_global_variables(self.backend_args)

    def _patch_parse_args(self):
        """Patch Megatron's parse_args to return pre-configured Primus arguments."""
        import megatron.training.arguments as megatron_args  # type: ignore
        import megatron.training.initialize as megatron_init  # type: ignore

        log_rank_0("Patching Megatron-LM parse_args()")

        patched_parse_args = lambda *args, **kwargs: (
            log_rank_0("parse_args() called; returning Primus arguments") or self.backend_args
        )

        megatron_args.parse_args = patched_parse_args
        megatron_init.parse_args = patched_parse_args

        log_rank_0(f"Patched parse_args(); Primus provided {len(vars(self.backend_args))} arguments")
