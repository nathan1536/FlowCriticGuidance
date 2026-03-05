import os
import sys
from pathlib import Path
import random
from typing import Optional, Tuple, Any, Dict

# --- Vendored gym + mj_envs + mjrl ---
_this_file = Path(__file__).resolve()
for _p in _this_file.parents:
    _vendored_gym = _p / "third_party" / "gym-0.21.0"
    if (_vendored_gym / "gym" / "__init__.py").exists():
        sys.path.insert(0, str(_vendored_gym))
        # mj_envs and mjrl live under third_party/rrl-dependencies
        _rrl_deps = _p / "third_party" / "rrl-dependencies"
        for _subpkg in ("mj_envs", "mjrl"):
            _pkg_path = _rrl_deps / _subpkg
            if _pkg_path.exists() and str(_pkg_path) not in sys.path:
                sys.path.insert(0, str(_pkg_path))
        break

# --- MuJoCo runtime (mujoco-py) ---
_mujoco_bin = os.path.expanduser("~/.mujoco/mujoco210/bin")
if os.path.isdir(_mujoco_bin):
    _ld = os.environ.get("LD_LIBRARY_PATH", "")
    if _mujoco_bin not in _ld.split(":"):
        os.environ["LD_LIBRARY_PATH"] = f"{_ld}:{_mujoco_bin}" if _ld else _mujoco_bin

import gym  # noqa: E402
import numpy as np  # noqa: E402
import mj_envs  # noqa: E402, F401  -- triggers env registration


ADROIT_TASKS = {
    "door": {"env_id": "door-v0", "max_episode_steps": 200},
    "hammer": {"env_id": "hammer-v0", "max_episode_steps": 200},
    "pen": {"env_id": "pen-v0", "max_episode_steps": 100},
    "relocate": {"env_id": "relocate-v0", "max_episode_steps": 200},
}

REWARD_RESCALE = {
    "door": 1 / 20,
    "hammer": 1 / 100,
    "pen": 1 / 50,
    "relocate": 1 / 30,
}


class SB3AdroitStateEnv(gym.Env):
    """
    Lightweight Adroit single-task env for Stable-Baselines3.

    - Uses the native (state) observation directly (no rendering / point-cloud).
    - Gym API (v0.21 style): step -> (obs, reward, done, info), reset -> obs
    - Maps info['goal_achieved'] -> info['success'] for callback compatibility
    """

    metadata = {"render.modes": ["rgb_array"]}

    def __init__(
        self,
        task_name: str,
        seed: Optional[int] = None,
        reward_rescale: bool = False,
        store_sim_states: bool = False,
        sim_state_buffer_size: int = 0,
    ):
        super().__init__()

        task_key = task_name.replace("-v0", "")
        if task_key not in ADROIT_TASKS:
            raise ValueError(
                f"Unknown Adroit task '{task_name}'. "
                f"Valid tasks: {list(ADROIT_TASKS.keys())}"
            )

        task_cfg = ADROIT_TASKS[task_key]
        self.task_name = task_key
        self.episode_length = task_cfg["max_episode_steps"]
        self._max_episode_steps = self.episode_length

        self.env = gym.make(task_cfg["env_id"])

        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

        self.reward_rescale_factor = REWARD_RESCALE[task_key] if reward_rescale else 1.0

        # Sim state ring buffer for image rendering at replay buffer save time.
        # Stays in sync with SB3's replay buffer (both increment once per step).
        self._store_sim_states = store_sim_states
        self._sim_state_buffer_size = sim_state_buffer_size
        self._sim_states = [None] * sim_state_buffer_size if store_sim_states else []
        self._sim_state_pos = 0

        self.cur_step = 0
        if seed is not None:
            self.seed(seed)

    def reset(self):
        obs = self.env.reset()
        self.cur_step = 0
        return np.asarray(obs, dtype=np.float32)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        # Capture sim state BEFORE stepping (corresponds to current obs/replay entry)
        if self._store_sim_states:
            self._sim_states[self._sim_state_pos] = self.env.get_env_state()
            self._sim_state_pos = (self._sim_state_pos + 1) % self._sim_state_buffer_size

        obs, reward, done, info = self.env.step(action)
        self.cur_step += 1

        reward = float(reward) * self.reward_rescale_factor
        done = bool(done) or (self.cur_step >= self.episode_length)

        # Normalise success key: Adroit uses 'goal_achieved', SAC eval expects 'success'
        info["success"] = bool(info.get("goal_achieved", False))

        return np.asarray(obs, dtype=np.float32), reward, done, info

    def seed(self, seed: Optional[int] = None):
        if seed is None:
            return
        random.seed(seed)
        np.random.seed(seed)
        try:
            self.env.seed(seed)
        except Exception:
            pass
        try:
            self.action_space.seed(seed)
        except Exception:
            pass

    def render(self, mode: str = "rgb_array"):
        try:
            return self.env.render(mode=mode)
        except TypeError:
            return self.env.render()

    def close(self):
        try:
            self.env.close()
        except Exception:
            pass
