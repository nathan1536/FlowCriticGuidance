
try:
    from .adroit import AdroitEnv
except Exception:
    # Optional envs can fail if system deps (e.g., mujoco_py + gcc) are missing.
    AdroitEnv = None
# from .dexart import DexArtEnv # require sapien==2.2.1
try:
    from .metaworld import MetaWorldEnv, MetaWorldEnv2D
except Exception:
    MetaWorldEnv = None
    MetaWorldEnv2D = None
try:
    from .robotwin import *  # require sapien==3.0.0b1
except Exception:
    pass


