import os
import sys
from pathlib import Path
import random
from typing import Optional, Tuple, Any, Dict

# --- Prefer vendored OpenAI gym (required by MetaWorld) ---
# Your conda env uses gymnasium, but MetaWorld expects classic gym.Env.
_this_file = Path(__file__).resolve()
for _p in _this_file.parents:
    _vendored_gym = _p / "third_party" / "gym-0.21.0"
    if (_vendored_gym / "gym" / "__init__.py").exists():
        sys.path.insert(0, str(_vendored_gym))
        break

# --- MuJoCo runtime (mujoco-py) ---
# mujoco-py requires MuJoCo's bin directory to be present in LD_LIBRARY_PATH.
_mujoco_bin = os.path.expanduser("~/.mujoco/mujoco210/bin")
if os.path.isdir(_mujoco_bin):
    _ld = os.environ.get("LD_LIBRARY_PATH", "")
    if _mujoco_bin not in _ld.split(":"):
        os.environ["LD_LIBRARY_PATH"] = f"{_ld}:{_mujoco_bin}" if _ld else _mujoco_bin

import gym  # noqa: E402
import numpy as np
import metaworld


class SB3MetaWorldStateEnv(gym.Env):
    """
    Lightweight MetaWorld single-task env for Stable-Baselines3.

    - Uses MetaWorld's native (state) observation directly (no rendering / point-cloud).
    - Gym API (v0.21 style): step -> (obs, reward, done, info), reset -> obs
    - Optional grasp bonus for manipulation tasks
    """

    metadata = {"render.modes": ["rgb_array"]}
    
    # Tasks that benefit from grasp reward shaping
    GRASP_TASKS = {
        'bin-picking', 'assembly', 'disassemble', 'hammer',
        'hand-insert', 'peg-insert-side', 'pick-place', 'pick-place-wall',
        'shelf-place', 'stick-pull', 'stick-push', 'handle-pull',
    }

    def __init__(
        self,
        task_name: str,
        episode_length: int = 200,
        seed: Optional[int] = None,
        grasp_reward_bonus: float = 0.5,
    ):
        super().__init__()

        if "-v2" not in task_name:
            task_name = task_name + "-v2-goal-observable"

        self.task_name = task_name
        self.episode_length = episode_length
        self._max_episode_steps = episode_length

        self.env = metaworld.envs.ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[task_name]()
        # Allow randomized initial conditions (False = randomize each reset)
        if hasattr(self.env, "_freeze_rand_vec"):
            self.env._freeze_rand_vec = False

        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

        # Grasp reward shaping for hard manipulation tasks
        task_base = task_name.replace('-v2-goal-observable', '').replace('-v2', '')
        self.use_grasp_bonus = task_base in self.GRASP_TASKS
        self.grasp_reward_bonus = grasp_reward_bonus if self.use_grasp_bonus else 0.0
        
        self.cur_step = 0
        if seed is not None:
            self.seed(seed)

    def reset(self):
        obs = self.env.reset()
        self.cur_step = 0
        return np.asarray(obs, dtype=np.float32)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        obs, reward, done, info = self.env.step(action)
        self.cur_step += 1
        done = bool(done) or (self.cur_step >= self.episode_length)
        
        # Add grasp reward bonus for hard manipulation tasks
        # This encourages the agent to actually close the gripper and grasp objects
        if self.use_grasp_bonus and self.grasp_reward_bonus > 0:
            grasp_success = info.get('grasp_success', False)
            grasp_reward = info.get('grasp_reward', 0.0)
            if grasp_success:
                reward += self.grasp_reward_bonus
            elif isinstance(grasp_reward, (int, float)) and grasp_reward > 0.5:
                # Partial bonus for good grasp attempt
                reward += self.grasp_reward_bonus * 0.5 * grasp_reward
        
        return np.asarray(obs, dtype=np.float32), float(reward), done, info

    def seed(self, seed: Optional[int] = None):
        if seed is None:
            return
        random.seed(seed)
        np.random.seed(seed)
        try:
            self.env.seed(seed)
        except Exception:
            # Some MetaWorld env variants may not expose seed()
            pass
        try:
            self.action_space.seed(seed)
        except Exception:
            pass

    def render(self, mode: str = "rgb_array"):
        # MetaWorld uses MuJoCo; render signature depends on version.
        try:
            return self.env.render(mode=mode)
        except TypeError:
            return self.env.render()

    def close(self):
        try:
            self.env.close()
        except Exception:
            pass

