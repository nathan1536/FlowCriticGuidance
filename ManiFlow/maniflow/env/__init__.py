
try:
    from .adroit import AdroitEnv
except Exception:
    # Optional envs can fail if system deps (e.g., mujoco_py + gcc) are missing.
    AdroitEnv = None
# from .dexart import DexArtEnv # require sapien==2.2.1
from .metaworld import MetaWorldEnv, MetaWorldEnv2D
try:
    from .robotwin import *  # require sapien==3.0.0b1
except Exception:
    pass


