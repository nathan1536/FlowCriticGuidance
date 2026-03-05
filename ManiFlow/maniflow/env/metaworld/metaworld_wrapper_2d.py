import gym
import numpy as np
import metaworld

from gym import spaces
from termcolor import cprint


class MetaWorldEnv2D(gym.Env):
    """
    2D-only MetaWorld wrapper.

    Differences vs `MetaWorldEnv` in `metaworld_wrapper.py`:
    - No point cloud generation
    - No depth generation
    - Only returns RGB image + robot state + full_state
    """

    metadata = {"render.modes": ["rgb_array"], "video.frames_per_second": 10}

    def __init__(
        self,
        task_name: str,
        device: str = "cuda:0",
        image_size: int = 128,
        camera_name: str = "corner2",
    ):
        super().__init__()

        if "-v2" not in task_name:
            task_name = task_name + "-v2-goal-observable"

        self.task_name = task_name
        self.camera_name = camera_name
        self.image_size = int(image_size)

        self.env = metaworld.envs.ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[task_name]()
        # allow randomized initial conditions
        if hasattr(self.env, "_freeze_rand_vec"):
            self.env._freeze_rand_vec = False

        # Match the 3D wrapper's camera defaults (best effort).
        try:
            self.env.sim.model.cam_pos[2] = [0.6, 0.295, 0.8]
            self.env.sim.model.vis.map.znear = 0.1
            self.env.sim.model.vis.map.zfar = 1.5
        except Exception:
            pass

        # Render device selection (MuJoCo device_id)
        # Accept "cuda:0" or ["cuda:0", ...] / "['cuda:0', ...]" and pick the first.
        device_str = device
        if isinstance(device, (list, tuple)):
            device_str = device[0] if len(device) > 0 else "cuda:0"
        device_str = str(device_str)
        if device_str.startswith("[") and device_str.endswith("]"):
            device_str = device_str[1:-1]
            device_str = device_str.split(",")[0].strip().strip("'").strip('"')
        self.device_id = int(device_str.split(":")[-1])
        cprint(f"[MetaWorldEnv2D] camera={self.camera_name} image_size={self.image_size} device_id={self.device_id}", "cyan")

        self.episode_length = self._max_episode_steps = 200
        self.action_space = self.env.action_space
        self.obs_sensor_dim = self.get_robot_state().shape[0]

        # Note: MetaWorld goal-observable state obs is typically 20-D.
        # Keep it generic by reading from env.observation_space.
        full_state_shape = getattr(self.env.observation_space, "shape", (20,))

        self.observation_space = spaces.Dict(
            {
                "image": spaces.Box(
                    low=0,
                    high=255,
                    shape=(3, self.image_size, self.image_size),
                    dtype=np.float32,
                ),
                "agent_pos": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.obs_sensor_dim,),
                    dtype=np.float32,
                ),
                "full_state": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=tuple(full_state_shape),
                    dtype=np.float32,
                ),
            }
        )

        self.cur_step = 0

    def get_robot_state(self):
        eef_pos = self.env.get_endeff_pos()
        finger_right, finger_left = (
            self.env._get_site_pos("rightEndEffector"),
            self.env._get_site_pos("leftEndEffector"),
        )
        return np.concatenate([eef_pos, finger_right, finger_left]).astype(np.float32)

    def get_rgb(self):
        # cam names: ('topview', 'corner', 'corner2', 'corner3', 'behindGripper', 'gripperPOV')
        img = self.env.sim.render(
            width=self.image_size,
            height=self.image_size,
            camera_name=self.camera_name,
            device_id=self.device_id,
        )
        return img

    def step(self, action: np.ndarray):
        raw_state, reward, done, env_info = self.env.step(action)
        self.cur_step += 1

        obs_pixels = self.get_rgb()
        robot_state = self.get_robot_state()

        # channel-first
        if obs_pixels.shape[0] != 3:
            obs_pixels = obs_pixels.transpose(2, 0, 1)

        obs_dict = {
            "image": obs_pixels,
            "agent_pos": robot_state,
            "full_state": np.asarray(raw_state, dtype=np.float32),
        }

        done = bool(done) or (self.cur_step >= self.episode_length)
        return obs_dict, float(reward), done, env_info

    def reset(self):
        self.env.reset()
        try:
            self.env.reset_model()
        except Exception:
            pass
        raw_obs = self.env.reset()
        self.cur_step = 0

        obs_pixels = self.get_rgb()
        robot_state = self.get_robot_state()

        if obs_pixels.shape[0] != 3:
            obs_pixels = obs_pixels.transpose(2, 0, 1)

        obs_dict = {
            "image": obs_pixels,
            "agent_pos": robot_state,
            "full_state": np.asarray(raw_obs, dtype=np.float32),
        }
        return obs_dict

    def render(self, mode="rgb_array"):
        return self.get_rgb()

    def close(self):
        try:
            self.env.close()
        except Exception:
            pass


