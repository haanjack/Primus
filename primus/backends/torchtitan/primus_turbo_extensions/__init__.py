import importlib.util

if importlib.util.find_spec("primus_turbo") is not None:
    import primus.backends.torchtitan.primus_turbo_extensions.primus_turbo_converter
