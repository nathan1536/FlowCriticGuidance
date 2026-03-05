try:
    from .metaworld_wrapper import MetaWorldEnv
except Exception:
    # Optional 3D wrapper depends on pytorch3d and other heavy deps.
    MetaWorldEnv = None

from .metaworld_wrapper_2d import MetaWorldEnv2D

try:
    from .sb3_metaworld_state_env import SB3MetaWorldStateEnv
except Exception:
    SB3MetaWorldStateEnv = None