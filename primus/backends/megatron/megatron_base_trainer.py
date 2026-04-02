###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import os

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

    def cleanup(self, on_error: bool = False):
        """Megatron cleanup: optional fast exit.

        * ``PRIMUS_EXIT_FAST=1`` — after ``super().cleanup()``, skip Python
          shutdown and call ``os._exit(0)`` directly. Saves ~20 s of Python
          interpreter shutdown / torchrun reaper lag at the cost of any
          ``atexit`` handlers below us. OFF by default.

        When both are off this method just delegates to ``super().cleanup()``,
        identical to the previous behavior.
        """
        super().cleanup(on_error=on_error)

        exit_fast = os.environ.get("PRIMUS_EXIT_FAST", "0") == "1"

        if exit_fast and not on_error:
            log_rank_0("[MegatronBaseTrainer] PRIMUS_EXIT_FAST=1 -> os._exit(0)")
            # Flush stdout/stderr so the final log lines are not lost.
            try:
                import sys

                sys.stdout.flush()
                sys.stderr.flush()
            except Exception:  # pragma: no cover
                pass
            os._exit(0)

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

        original_validate_args = megatron_args.validate_args

        def _run_modified_validate(ori_code, new_code, *args, **kwargs):
            before_patch_validate = megatron_args.validate_args
            before_init_validate = megatron_init.validate_args
            try:
                megatron_args.validate_args = original_validate_args
                megatron_init.validate_args = original_validate_args
                validated_args = validate_args_modified(*args, **kwargs, ori_code=ori_code, new_code=new_code)
                if validated_args is None:
                    validated_args = args[0] if args else kwargs.get("args", None)
                return validated_args
            finally:
                # Always restore wrapper references to avoid leaking global monkey patches.
                megatron_args.validate_args = before_patch_validate
                megatron_init.validate_args = before_init_validate

        def patched_validate_args(*args, **kwargs):
            parsed_args = args[0] if args else kwargs.get("args", None)
            if parsed_args is None:
                return original_validate_args(*args, **kwargs)

            if getattr(parsed_args, "decoder_pipeline_manual_split_list", None) is not None:
                ori_code = "if args.decoder_first_pipeline_num_layers is None and args.decoder_last_pipeline_num_layers is None:"
                new_code = (
                    "if args.decoder_pipeline_manual_split_list is None and " + ori_code.split("if ")[-1]
                )
                validated_args = _run_modified_validate(ori_code, new_code, *args, **kwargs)
            elif getattr(parsed_args, "fp4", None) is not None:
                ori_code = """raise ValueError("--fp4-format requires Transformer Engine >= 2.7.0.dev0 for NVFP4BlockScaling support.")"""
                new_code = """pass"""
                validated_args = _run_modified_validate(ori_code, new_code, *args, **kwargs)
            else:
                validated_args = original_validate_args(*args, **kwargs)

            import torch

            _target = getattr(self.backend_args, "target_gpu", "auto")
            if _target == "auto":
                _target = "amd" if getattr(torch.version, "hip", None) else "nvidia"

            if _target == "amd":
                log_rank_0("validate_args() called; validating on ROCm")
                validate_args_on_rocm(validated_args)
            else:
                log_rank_0("validate_args() called; non-ROCm target, skipping ROCm-specific validation")
            return validated_args


        megatron_args.parse_args = patched_parse_args
        megatron_init.parse_args = patched_parse_args

        log_rank_0(f"Patched parse_args(); Primus provided {len(vars(self.backend_args))} arguments")
