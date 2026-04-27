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
        store_sim_states: bool = False,
        sim_state_buffer_size: int = 0,
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

        # Ring buffers for sim-state-based image rendering
        self._store_sim_states = store_sim_states
        self._sim_state_buffer_size = sim_state_buffer_size
        _buf = sim_state_buffer_size if sim_state_buffer_size > 0 else 0
        self._buf_pos = 0

        if store_sim_states and _buf > 0:
            nq = self.env.sim.model.nq
            nv = self.env.sim.model.nv
            self._sim_qpos      = np.zeros((_buf, nq), dtype=np.float64)
            self._sim_qvel      = np.zeros((_buf, nv), dtype=np.float64)
            self._next_sim_qpos = np.zeros((_buf, nq), dtype=np.float64)
            self._next_sim_qvel = np.zeros((_buf, nv), dtype=np.float64)
        else:
            self._sim_qpos = self._sim_qvel = None
            self._next_sim_qpos = self._next_sim_qvel = None

        self._success_flags = np.zeros(_buf, dtype=np.float32) if _buf > 0 else None
        # agent_pos = [eef_pos(3), finger_right(3), finger_left(3)] stored per step
        self._agent_pos_buf      = np.zeros((_buf, 9), dtype=np.float32) if _buf > 0 else None
        self._next_agent_pos_buf = np.zeros((_buf, 9), dtype=np.float32) if _buf > 0 else None
        self._store_agent_pos = _buf > 0
        self._post_reset_robot_state: Optional[np.ndarray] = None  # cached after reset()

        self.cur_step = 0
        if seed is not None:
            self.seed(seed)

    def reset(self):
        obs = self.env.reset()
        self.cur_step = 0
        if self._store_agent_pos:
            self._post_reset_robot_state = self._get_robot_state()
        return np.asarray(obs, dtype=np.float32)

    def _get_robot_state(self) -> np.ndarray:
        """[eef_pos(3), finger_right(3), finger_left(3)] — matches MetaWorldEnv2D.get_robot_state()."""
        eef_pos = self.env.get_endeff_pos()
        finger_right = self.env._get_site_pos('rightEndEffector')
        finger_left  = self.env._get_site_pos('leftEndEffector')
        return np.concatenate([eef_pos, finger_right, finger_left]).astype(np.float32)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        pos = self._buf_pos

        if self._store_sim_states:
            self._sim_qpos[pos] = self.env.sim.data.qpos
            self._sim_qvel[pos] = self.env.sim.data.qvel
        if self._store_agent_pos:
            if self._post_reset_robot_state is not None:
                self._agent_pos_buf[pos] = self._post_reset_robot_state
                self._post_reset_robot_state = None
            else:
                self._agent_pos_buf[pos] = self._get_robot_state()

        obs, reward, done, info = self.env.step(action)
        self.cur_step += 1
        done = bool(done) or (self.cur_step >= self.episode_length)

        if self._store_sim_states:
            self._next_sim_qpos[pos] = self.env.sim.data.qpos
            self._next_sim_qvel[pos] = self.env.sim.data.qvel
        if self._store_agent_pos:
            self._next_agent_pos_buf[pos] = self._get_robot_state()

        # Add grasp reward bonus for hard manipulation tasks
        if self.use_grasp_bonus and self.grasp_reward_bonus > 0:
            grasp_success = info.get('grasp_success', False)
            grasp_reward = info.get('grasp_reward', 0.0)
            if grasp_success:
                reward += self.grasp_reward_bonus
            elif isinstance(grasp_reward, (int, float)) and grasp_reward > 0.5:
                reward += self.grasp_reward_bonus * 0.5 * grasp_reward

        info["success"] = bool(info.get("success", False))

        if self._success_flags is not None:
            self._success_flags[pos] = 1.0 if info["success"] else 0.0

        if self._sim_state_buffer_size > 0:
            self._buf_pos = (pos + 1) % self._sim_state_buffer_size

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

