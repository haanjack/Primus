#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""
Memory-efficient HF -> Megatron checkpoint conversion.

Patches the LOW_MEMORY_SAVE clone loop to call malloc_trim(0) periodically,
forcing glibc to return freed pages to the OS immediately. Without this,
freed tensor pages accumulate in the malloc heap and the kernel OOM-kills
the process around the 50% mark on 1T+ models.

Also patches:
- TransformerConfig: skip shared expert validation when n_shared_experts=0
- Checkpoint generation: avoid modulo-by-zero when num_moe_shared_experts=0
- CheckpointConfig: set async_strategy='mcore' to avoid nvidia_resiliency_ext
  import failure (the container has an incompatible version)
"""
import argparse
import ctypes
import gc
import sys
import time

_libc = ctypes.CDLL("libc.so.6")


def malloc_trim():
    try:
        _libc.malloc_trim(0)
    except Exception:
        pass


def log_mem(tag: str) -> None:
    try:
        with open("/proc/meminfo") as f:
            info = {k.strip(): v.strip() for k, v in
                    (line.split(":", 1) for line in f if ":" in line)}
        total      = int(info["MemTotal"].split()[0])    // 1024
        avail      = int(info["MemAvailable"].split()[0]) // 1024
        swap_total = int(info["SwapTotal"].split()[0])   // 1024
        swap_free  = int(info["SwapFree"].split()[0])    // 1024
        ts = time.strftime("%H:%M:%S")
        print(
            f"[mem {ts}] {tag}: "
            f"RAM used={total - avail:,} MB / {total:,} MB  "
            f"(avail={avail:,} MB)  "
            f"Swap used={swap_total - swap_free:,} MB / {swap_total:,} MB",
            flush=True,
        )
    except Exception as e:
        print(f"[mem] {tag}: could not read /proc/meminfo: {e}", flush=True)


def patch_moe_shared_expert_validation():
    """
    Patch Megatron's transformer_config.py to skip the shared expert intermediate size
    validation when n_shared_experts=0. The validation formula
    (moe_shared_expert_intermediate_size == n_shared_experts * ffn_size) causes a
    division by zero error when n_shared_experts=0.
    """
    try:
        from megatron.core.transformer.transformer_config import TransformerConfig

        orig_post_init = TransformerConfig.__post_init__

        def patched_post_init(self):
            moe_shared = getattr(self, 'moe_shared_expert_intermediate_size', None)
            n_shared = getattr(self, 'num_moe_shared_experts', None) or getattr(self, 'n_shared_experts', 0)

            if n_shared == 0 and moe_shared is not None:
                self._orig_moe_shared = moe_shared
                self.moe_shared_expert_intermediate_size = None
                try:
                    orig_post_init(self)
                finally:
                    self.moe_shared_expert_intermediate_size = self._orig_moe_shared
                    delattr(self, '_orig_moe_shared')
            else:
                orig_post_init(self)

        TransformerConfig.__post_init__ = patched_post_init
        print("[convert] Patched TransformerConfig: skip MoE shared expert validation when n_shared_experts=0", flush=True)
    except Exception as e:
        print(f"[convert] Could not patch TransformerConfig (continuing anyway): {e}", flush=True)


def patch_checkpoint_generation():
    """
    Patch Megatron-Bridge's checkpoint generation to avoid modulo-by-zero when
    num_moe_shared_experts=0.
    """
    try:
        import megatron.bridge.training.checkpointing as ckpt

        orig_generate_model = ckpt._generate_model_state_dict

        def patched_generate_model(model, prefix='', keep_vars=False, **kwargs):
            if hasattr(model, 'config'):
                cfg = model.config
                tc = getattr(cfg, 'text_config', cfg)
                n_shared = getattr(tc, 'n_shared_experts', 0) or getattr(tc, 'num_moe_shared_experts', 0)

                if n_shared == 0:
                    if hasattr(tc, 'n_shared_experts'):
                        tc._orig_n_shared = tc.n_shared_experts
                        tc.n_shared_experts = 1
                    if hasattr(tc, 'num_moe_shared_experts'):
                        tc._orig_num_shared = tc.num_moe_shared_experts
                        tc.num_moe_shared_experts = 1
                    try:
                        return orig_generate_model(model, prefix, keep_vars, **kwargs)
                    finally:
                        if hasattr(tc, '_orig_n_shared'):
                            tc.n_shared_experts = tc._orig_n_shared
                            delattr(tc, '_orig_n_shared')
                        if hasattr(tc, '_orig_num_shared'):
                            tc.num_moe_shared_experts = tc._orig_num_shared
                            delattr(tc, '_orig_num_shared')

            return orig_generate_model(model, prefix, keep_vars, **kwargs)

        ckpt._generate_model_state_dict = patched_generate_model
        print("[convert] Patched checkpoint generation: avoid modulo-by-zero for shared experts", flush=True)
    except Exception as e:
        print(f"[convert] Could not patch checkpoint generation (continuing anyway): {e}", flush=True)


def patch_low_memory_save():
    """
    Monkey-patch the LOW_MEMORY_SAVE clone loop in model_load_save.py to call
    malloc_trim(0) every gc.collect(). This forces glibc to return freed mmap
    pages to the OS during the clone-and-free phase, preventing OOM on large
    models where param storage is freed incrementally.
    """
    import megatron.bridge.training.model_load_save as mls

    orig_save = mls.save_megatron_model

    def patched_save(model, path, *, low_memory_save=False, **kwargs):
        if not low_memory_save:
            return orig_save(model, path, low_memory_save=low_memory_save, **kwargs)

        orig_gc_collect = gc.collect
        call_count = [0]

        def patched_gc_collect(*args, **kw):
            result = orig_gc_collect(*args, **kw)
            call_count[0] += 1
            malloc_trim()
            log_mem(f"gc+trim #{call_count[0]}")
            return result

        gc.collect = patched_gc_collect
        try:
            return orig_save(model, path, low_memory_save=True, **kwargs)
        finally:
            gc.collect = orig_gc_collect

    mls.save_megatron_model = patched_save
    print("[convert] Patched save_megatron_model: malloc_trim on every gc.collect", flush=True)


def patch_async_strategy():
    """
    Force HAVE_NVRX=False in the torch dist checkpoint strategy module.

    The container has nvidia_resiliency_ext installed but with an incompatible
    API: `_get_write_results_queue` (with underscore) instead of
    `get_write_results_queue`. The module-level import at line 55 succeeds
    (HAVE_NVRX=True), but the deeper import in get_async_strategy() fails.

    TorchDistSaveShardedStrategy.save() hardcodes
    `strategy = "nvrx" if HAVE_NVRX else "mcore"` (line 648), ignoring the
    config's async_strategy entirely. Setting HAVE_NVRX=False forces it to
    use the mcore async path which works without nvidia_resiliency_ext.
    """
    try:
        import megatron.core.dist_checkpointing.strategies.torch as torch_strat
        torch_strat.HAVE_NVRX = False
        print("[convert] Patched torch strategy: HAVE_NVRX=False (force mcore async path)", flush=True)
    except Exception as e:
        print(f"[convert] Could not patch HAVE_NVRX (continuing anyway): {e}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Convert HF checkpoint to Megatron format")
    parser.add_argument("--hf-model", required=True)
    parser.add_argument("--megatron-path", required=True)
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    from megatron.bridge import AutoBridge

    kwargs = {}
    if args.trust_remote_code:
        kwargs["trust_remote_code"] = True

    patch_moe_shared_expert_validation()
    patch_checkpoint_generation()
    patch_low_memory_save()
    patch_async_strategy()
    log_mem("start")

    print(f"[convert] Loading HF model: {args.hf_model}", flush=True)
    bridge = AutoBridge.from_hf_pretrained(args.hf_model, **kwargs)
    log_mem("after HF model load")

    print("[convert] Converting to Megatron model...", flush=True)
    megatron_model = bridge.to_megatron_model(wrap_with_ddp=False, use_cpu_initialization=True)
    log_mem("after to_megatron_model")

    # Free HF model before save -- copied weights now live in megatron_model
    print("[convert] Freeing HF model from memory...", flush=True)
    del bridge.hf_pretrained
    bridge.hf_pretrained = None
    gc.collect()
    malloc_trim()
    log_mem("after free+trim")

    print("[convert] Starting checkpoint save...", flush=True)

    hf_tokenizer_kwargs = {}
    if hasattr(bridge._model_bridge, "get_hf_tokenizer_kwargs"):
        hf_tokenizer_kwargs = bridge._model_bridge.get_hf_tokenizer_kwargs()
    if kwargs.get("trust_remote_code"):
        if hf_tokenizer_kwargs is None:
            hf_tokenizer_kwargs = {}
        hf_tokenizer_kwargs.setdefault("trust_remote_code", True)

    bridge.save_megatron_model(
        megatron_model,
        args.megatron_path,
        hf_tokenizer_path=args.hf_model,
        hf_tokenizer_kwargs=hf_tokenizer_kwargs,
        low_memory_save=True,
    )
    log_mem("after save_megatron_model")

    print(f"[convert] Done. Checkpoint saved to: {args.megatron_path}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
