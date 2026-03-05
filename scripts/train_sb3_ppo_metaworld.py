#!/usr/bin/env python3
"""
Single-task PPO training on MetaWorld using vendored Stable-Baselines3.

Includes an improved Q(s,a) critic training mode with:
- Contrastive negative action sampling (teaches action-sensitivity)
- Experience replay buffer (maintains diversity after PPO converges)
- Margin ranking loss (Q(s,a_good) > Q(s,a_bad) + margin)
- Mini-batch training for stability

Example:
  python scripts/train_sb3_ppo_metaworld.py --task reach --total-timesteps 2000000 --n-envs 8 --train-q-critic

  # With improved contrastive training (recommended):
  python scripts/train_sb3_ppo_metaworld.py --task reach --total-timesteps 5000000 --n-envs 8 \
      --train-q-critic --q-critic-epochs 50 --q-critic-neg-samples 4 --tensorboard
"""

import argparse
import os
import sys
from pathlib import Path
import importlib.util

import numpy as np
from typing import Optional
import torch.nn as nn


def _acgd_qcritic_state_action_dims(obs_space, act_space) -> tuple[int, int]:
    """
    Infer (state_dim, action_dim) for a simple MLP Q(s,a) critic.
    Only supports flat Box observations and Box actions (MetaWorld).
    """
    obs_shape = getattr(obs_space, "shape", None)
    act_shape = getattr(act_space, "shape", None)
    if obs_shape is None or act_shape is None or len(obs_shape) != 1 or len(act_shape) != 1:
        raise ValueError(f"Unsupported spaces for Q-critic: obs_shape={obs_shape} act_shape={act_shape}")
    return int(obs_shape[0]), int(act_shape[0])


def _build_acgd_qcritic(state_dim: int, action_dim: int, hidden_dim: int):
    """
    Construct the Q(s,a) critic network.
    Architecture must match build_qcritic() in workspace for loading.
    """


    return nn.Sequential(
        nn.Linear(state_dim + action_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, 1),
    )


def _ensure_sb3_importable(repo_root: Path) -> None:
    """
    Prefer system-installed stable_baselines3. If missing, fall back to vendored copy:
    third_party/dexart-release/stable_baselines3
    """
    try:
        import stable_baselines3  # noqa: F401
        return
    except Exception:
        pass

    vendored_parent = repo_root / "third_party" / "dexart-release"
    sys.path.insert(0, str(vendored_parent))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task", type=str, default="bin-picking", help="MetaWorld task name (e.g., assembly, basketball, ...)")
    p.add_argument("--total-timesteps", type=int, default=20_000_000)
    p.add_argument("--n-envs", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="auto", help="SB3 device: auto|cpu|cuda")
    p.add_argument("--log-dir", type=str, default="runs/sb3_metaworld_ppo")
    p.add_argument("--save-dir", type=str, default="runs/sb3_metaworld_ppo/models_v3")
    p.add_argument("--monitor-dir", type=str, default="", help="If set, write Monitor CSVs for reward curves.")
    p.add_argument("--tensorboard", action="store_true", help="Enable tensorboard logging (requires tensorboard)")
    p.add_argument("--eval-episodes", type=int, default=20)
    p.add_argument("--eval-freq", type=int, default=200_000, help="Evaluate every N timesteps during training (0 disables).")
    p.add_argument(
        "--best-metric",
        type=str,
        default="success_rate",
        choices=["success_rate", "mean_reward"],
        help="Metric used to decide the best checkpoint.",
    )
    p.add_argument(
        "--save-critic",
        action="store_true",
        help="Also save critic/value-function weights to *.pt alongside saved models.",
    )
    p.add_argument("--mujoco-bin", type=str, default="~/.mujoco/mujoco210/bin", help="MuJoCo bin dir (for mujoco-py LD_LIBRARY_PATH)")
    p.add_argument(
        "--track-success-horizon",
        action="store_true",
        help="Track step-of-first-success within each episode during training (MetaWorld info['success']).",
    )
    p.add_argument(
        "--train-q-critic",
        action="store_true",
        help="Train an auxiliary Q(s,a) critic from PPO rollout buffer returns and save it as expert_critic_q.pt.",
    )
    p.add_argument("--q-critic-lr", type=float, default=3e-4)
    p.add_argument("--q-critic-epochs", type=int, default=50, help="Gradient epochs per rollout (default 50; was 10).")
    p.add_argument("--q-critic-hidden-dim", type=int, default=256)
    p.add_argument(
        "--q-critic-save-name",
        type=str,
        default="expert_critic_q.pt",
        help="Filename (under save_dir/task) for the exported Q-critic weights.",
    )
    # ── Contrastive Q-critic training options ──
    p.add_argument(
        "--q-critic-neg-samples",
        type=int,
        default=6,
        help="Number of negative (perturbed/random) actions per positive sample. "
             "Higher = stronger action-sensitivity signal. 0 = disable contrastive training.",
    )
    p.add_argument(
        "--q-critic-neg-noise-std",
        type=float,
        default=0.3,
        help="Std of Gaussian noise added to on-policy actions for negative samples.",
    )
    p.add_argument(
        "--q-critic-random-neg-ratio",
        type=float,
        default=0.5,
        help="Fraction of negative samples that are purely random (vs noisy perturbations).",
    )
    p.add_argument(
        "--q-critic-replay-size",
        type=int,
        default=200_000,
        help="Replay buffer size for Q-critic training (stores past rollouts for diversity).",
    )
    p.add_argument(
        "--q-critic-mini-batch-size",
        type=int,
        default=2048,
        help="Mini-batch size for Q-critic training.",
    )
    p.add_argument(
        "--q-critic-margin",
        type=float,
        default=5.0,
        help="Margin for ranking loss: Q(s,a_good) > Q(s,a_bad) + margin.",
    )

    p.add_argument("--q-critic-gamma", type=float, default=0.99, help="Discount factor for TD targets.")
    p.add_argument("--q-critic-tau", type=float, default=0.005, help="Polyak averaging rate for target network.")
    p.add_argument(
        "--q-critic-margin-weight",
        type=float,
        default=5.0,
        help="Weight of margin ranking loss relative to TD loss.",
    )
    p.add_argument(
        "--expert-data-path",
        type=str,
        default="ManiFlow/data",
        help="Path to directory containing metaworld_<task>_expert.zarr files. "
             "Expert states are mixed into Q-critic training so the critic "
             "generalizes to the student's (expert-dataset) state distribution.",
    )
    p.add_argument(
        "--q-critic-expert-ratio",
        type=float,
        default=0.3,
        help="Fraction of each mini-batch that comes from expert demonstrations "
             "(vs PPO replay buffer). 0.3 = 30%% expert, 70%% PPO.",
    )
    return p.parse_args()

def _import_from_path(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module {module_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _ensure_ld_library_path_on_process_start(mujoco_bin: str) -> None:
    """
    Some native libs (mujoco-py dependencies) are resolved by the dynamic loader
    using LD_LIBRARY_PATH as of *process start*. In that case, mutating
    os.environ inside Python is too late.
    """
    if not mujoco_bin or not os.path.isdir(mujoco_bin):
        return
    ld = os.environ.get("LD_LIBRARY_PATH", "")
    if mujoco_bin in ld.split(":"):
        return
    if os.environ.get("MANIFLOW_LD_REEXEC", "") == "1":
        return

    new_env = os.environ.copy()
    new_env["LD_LIBRARY_PATH"] = f"{ld}:{mujoco_bin}" if ld else mujoco_bin
    new_env["MANIFLOW_LD_REEXEC"] = "1"
    os.execvpe(sys.executable, [sys.executable] + sys.argv, new_env)


def main():
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[1]

    # mujoco-py requires MuJoCo's bin directory on LD_LIBRARY_PATH.
    mujoco_bin = os.path.expanduser(args.mujoco_bin)
    _ensure_ld_library_path_on_process_start(mujoco_bin)

    # Ensure MetaWorld imports the correct OpenAI gym (vendored) instead of any gymnasium shim.
    sys.path.insert(0, str(repo_root / "third_party" / "gym-0.21.0"))
    sys.path.insert(0, str(repo_root / "third_party" / "Metaworld"))

    _ensure_sb3_importable(repo_root)

    # Import our env without importing `maniflow.env` (which has optional deps like dm_env)
    # by loading the file directly.
    env_mod = _import_from_path(
        "sb3_metaworld_state_env",
        repo_root / "ManiFlow" / "maniflow" / "env" / "metaworld" / "sb3_metaworld_state_env.py",
    )
    SB3MetaWorldStateEnv = env_mod.SB3MetaWorldStateEnv

    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback, CallbackList
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
    import numpy as np
    import torch

    log_dir = repo_root / args.log_dir / args.task
    save_dir = repo_root / args.save_dir / args.task
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    def save_critic(model: "PPO", path: Path) -> None:
        """
        Save critic/value-function related weights for later inspection.
        Note: value prediction still depends on the feature extractor and mlp_extractor.
        """
        policy = model.policy
        payload = {
            "policy_class": policy.__class__.__name__,
            "device": str(model.device),
            "state_dict": {
                # feature extractor (usually has no params for MlpPolicy, but keep it generic)
                "features_extractor": getattr(policy, "features_extractor", None).state_dict()
                if getattr(policy, "features_extractor", None) is not None
                else {},
                # shared MLP extractor contains vf/pi nets for MlpPolicy
                "mlp_extractor": getattr(policy, "mlp_extractor", None).state_dict()
                if getattr(policy, "mlp_extractor", None) is not None
                else {},
                # value head
                "value_net": getattr(policy, "value_net", None).state_dict()
                if getattr(policy, "value_net", None) is not None
                else {},
            },
        }
        torch.save(payload, str(path))

    tensorboard_log = None
    if args.tensorboard:
        try:
            import tensorboard  # noqa: F401
            tensorboard_log = str(log_dir)
        except Exception:
            print("[Warn] --tensorboard set but tensorboard is not installed; disabling tensorboard logging.")

    def make_env(rank: int):
        def _init():
            env = SB3MetaWorldStateEnv(task_name=args.task, seed=args.seed + rank)
            if args.monitor_dir:
                monitor_root = repo_root / args.monitor_dir / args.task
                os.makedirs(monitor_root, exist_ok=True)
                env = Monitor(env, filename=str(monitor_root / f"monitor_rank{rank}.csv"))
            return env

        return _init

    if args.n_envs <= 1:
        vec_env = DummyVecEnv([make_env(0)])
    else:
        vec_env = SubprocVecEnv([make_env(i) for i in range(args.n_envs)])
    vec_env = VecMonitor(vec_env)

    # Separate eval env (always 1 env for deterministic episodic eval)
    eval_env = DummyVecEnv([make_env(10_000)])

    class SuccessHorizonCallback(BaseCallback):
        """
        Tracks per-episode:
        - episode length (in env steps)
        - first step index where info['success'] becomes True (if any)

        Works with VecEnv: maintains per-env counters.
        """

        def __init__(self):
            super().__init__()
            self.episode_lengths = []
            self.success_steps = []  # first-success step within episode (1-indexed)
            self._steps_in_ep = None
            self._first_success_step = None

        def _on_training_start(self) -> None:
            n_envs = int(getattr(self.training_env, "num_envs", 1))
            self._steps_in_ep = np.zeros(n_envs, dtype=np.int32)
            self._first_success_step = np.full(n_envs, -1, dtype=np.int32)

        def _on_step(self) -> bool:
            infos = self.locals.get("infos", [])
            dones = self.locals.get("dones", [])

            # VecEnv steps: one step per active env each call
            for i in range(len(dones)):
                self._steps_in_ep[i] += 1
                info = infos[i] if i < len(infos) else {}
                if self._first_success_step[i] < 0 and isinstance(info, dict) and bool(info.get("success", False)):
                    # 1-indexed step count within the episode
                    self._first_success_step[i] = int(self._steps_in_ep[i])

                if bool(dones[i]):
                    self.episode_lengths.append(int(self._steps_in_ep[i]))
                    if self._first_success_step[i] >= 0:
                        self.success_steps.append(int(self._first_success_step[i]))
                    self._steps_in_ep[i] = 0
                    self._first_success_step[i] = -1

            return True

    success_cb = SuccessHorizonCallback() if args.track_success_horizon else None

    class PeriodicEvalBestCallback(BaseCallback):
        """
        Every `eval_freq` timesteps:
        - runs evaluation for N episodes
        - computes mean_reward + success_rate (MetaWorld info['success'])
        - saves best model to best_model.zip according to selected metric
        """

        def __init__(
            self,
            eval_env,
            eval_freq: int,
            n_eval_episodes: int,
            save_dir: Path,
            best_metric: str = "success_rate",
            q_critic: Optional["torch.nn.Module"] = None,
            q_critic_name: str = "expert_critic_q.pt",
            verbose: int = 1,
        ):
            super().__init__(verbose=verbose)
            self.eval_env = eval_env
            self.eval_freq = int(eval_freq)
            self.n_eval_episodes = int(n_eval_episodes)
            self.save_dir = Path(save_dir)
            self.best_metric = best_metric
            self.best_value = -float("inf")
            self.last_eval_t = 0

            # Store the Q-critic reference
            self.q_critic = q_critic
            self.q_critic_name = q_critic_name

        def _init_callback(self) -> None:
            os.makedirs(self.save_dir, exist_ok=True)

        def _evaluate(self):
            episode_rewards = []
            episode_success = []
            success_key_missing_steps = 0
            success_key_total_steps = 0

            for _ in range(self.n_eval_episodes):
                obs = self.eval_env.reset()
                done = np.array([False])
                ep_rew = 0.0
                ep_succ = False
                while not bool(done[0]):
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, done, infos = self.eval_env.step(action)
                    ep_rew += float(reward[0])
                    info0 = infos[0] if isinstance(infos, (list, tuple)) and len(infos) > 0 else {}
                    if isinstance(info0, dict):
                        success_key_total_steps += 1
                        if "success" not in info0:
                            success_key_missing_steps += 1
                        elif bool(info0.get("success", False)):
                            ep_succ = True
                episode_rewards.append(ep_rew)
                episode_success.append(1.0 if ep_succ else 0.0)

            mean_reward = float(np.mean(episode_rewards)) if len(episode_rewards) else 0.0
            success_rate = float(np.mean(episode_success)) if len(episode_success) else 0.0
            all_missing_success = (success_key_total_steps > 0 and success_key_missing_steps == success_key_total_steps)
            return mean_reward, success_rate, all_missing_success

        def _on_step(self) -> bool:
            if self.eval_freq <= 0:
                return True
            if (self.num_timesteps - self.last_eval_t) < self.eval_freq:
                return True

            self.last_eval_t = int(self.num_timesteps)
            mean_reward, success_rate, all_missing_success = self._evaluate()
                        # Share success rate with Q-critic callback so it can gate best-critic saves
            if q_cb is not None:
                q_cb._last_eval_success_rate = success_rate

            if self.verbose:
                if all_missing_success:
                    print(
                        "[Eval Warn] `info['success']` was missing for all eval steps. "
                        "Success-rate may be meaningless for this env variant."
                    )
                print(
                    f"[Periodic Eval] t={self.num_timesteps} episodes={self.n_eval_episodes} "
                    f"mean_reward={mean_reward:.3f} success_rate={success_rate:.3f}"
                )

            # Log to SB3 logger (will show up in TensorBoard if tensorboard_log is enabled).
            # We also log mean reward for convenience.
            self.logger.record("eval/mean_reward", float(mean_reward))
            self.logger.record("eval/success_rate", float(success_rate))
            # Force an immediate write at the evaluation timestep so the curve aligns with eval points.
            self.logger.dump(self.num_timesteps)

            value = success_rate if self.best_metric == "success_rate" else mean_reward
            if value >= self.best_value:
                self.best_value = float(value)
                best_path = self.save_dir / "best_model.zip"
                self.model.save(str(best_path))
                if args.save_critic:
                    critic_path = self.save_dir / "best_critic.pt"
                    save_critic(self.model, critic_path)
                if self.q_critic is not None:
                    q_path = self.save_dir / self.q_critic_name
                    # Ensure we save the state_dict, not the object
                    torch.save(self.q_critic.state_dict(), str(q_path))
                    if self.verbose:
                        print(f"[Best Model] Saved Q-Critic to {q_path}")
                if self.verbose:
                    print(f"[Best Model] metric={self.best_metric} value={value:.3f} saved={best_path}")
            return True



    # ══════════════════════════════════════════════════════════════════
    # Improved Q(s,a) critic training with contrastive negative sampling.
    #
    # WHY: Standard PPO only provides (s, a_good, R) tuples. After PPO
    # converges, all actions in the buffer are similar, so the critic
    # learns V(s) — a FLAT landscape over actions. ACGD then has no
    # gradient to follow.
    #
    # FIX: For each real (s, a, R), we synthesize negative examples:
    #   1. Noisy perturbations:  (s, a+ε, R_low)  — teaches local sensitivity
    #   2. Random actions:       (s, a_rand, 0)    — teaches global contrast
    #   3. Margin ranking loss:  Q(s,a) > Q(s,a_bad) + margin
    #   4. Replay buffer:        stores past rollouts for diversity
    # ══════════════════════════════════════════════════════════════════
    class TrainQCriticCallback(BaseCallback):
        """
        Contrastive Q(s,a) critic trained alongside PPO.
        Produces action-sensitive Q-landscapes suitable for ACGD.

        Now supports mixing expert demonstration data (from zarr) into
        training so the critic generalizes to the student's state distribution.
        """

        def __init__(
                            self,
            q_critic: "torch.nn.Module",
            q_optimizer: "torch.optim.Optimizer",
            q_epochs: int = 50,
            neg_samples: int = 4,
            neg_noise_std: float = 0.3,
            random_neg_ratio: float = 0.5,
            replay_size: int = 200_000,
            mini_batch_size: int = 2048,
            margin: float = 10.0,
            margin_weight: float = 1.0,
            gamma: float = 0.99,
            target_update_freq: int = 5,
            tau: float = 0.005,
            expert_data_path: Optional[str] = None,
            task_name: Optional[str] = None,
            expert_ratio: float = 0.3,
            save_dir: Optional[Path] = None,
            verbose: int = 0,
        ):
            super().__init__(verbose=verbose)
            self.q_critic = q_critic
            self.q_optimizer = q_optimizer
            self.q_epochs = int(q_epochs)
            self.neg_samples = int(neg_samples)
            self.neg_noise_std = float(neg_noise_std)
            self.random_neg_ratio = float(random_neg_ratio)
            self.replay_size = int(replay_size)
            self.mini_batch_size = int(mini_batch_size)
            self.margin = float(margin)
            self.margin_weight = float(margin_weight)
            self.gamma = float(gamma)
            self.target_update_freq = int(target_update_freq)
            self.tau = float(tau)
            self.expert_ratio = float(expert_ratio)
            self.mse_fn = torch.nn.MSELoss()
            self.rank_fn = torch.nn.MarginRankingLoss(margin=self.margin)

            # Target network for stable TD targets
            import copy
            self.q_target = copy.deepcopy(q_critic)
            self.q_target.requires_grad_(False)

            # Best-critic-quality tracking
            self.save_dir = save_dir
            self._best_q_gap_vs_noisy = -float("inf")

            # ── Load expert demonstration data from zarr ──────────────
            self._expert_obs = None
            self._expert_actions = None
            if expert_data_path and task_name:
                self._load_expert_data(expert_data_path, task_name)

            # Replay buffer (now stores transitions: s, a, r, s', done)
            self._replay_obs = None
            self._replay_actions = None
            self._replay_rewards = None
            self._replay_next_obs = None
            self._replay_dones = None
            self._replay_ptr = 0
            self._replay_filled = 0
            self._rollout_count = 0

        def _load_expert_data(self, data_path: str, task_name: str):
            """Load full_state and action from the expert zarr dataset."""
            import zarr
            zarr_path = Path(data_path) / f"metaworld_{task_name}_expert.zarr"
            if not zarr_path.exists():
                print(f"[Q-Critic] WARNING: Expert zarr not found at {zarr_path}. "
                      f"Critic will train on PPO data only (may cause OOD issues).")
                return

            root = zarr.open(str(zarr_path), mode="r")
            full_state = np.array(root["data"]["full_state"])  # (N, state_dim)
            action = np.array(root["data"]["action"])          # (N, action_dim)

            self._expert_obs = torch.tensor(full_state, dtype=torch.float32)
            self._expert_actions = torch.tensor(action, dtype=torch.float32)
            print(
                f"[Q-Critic] Loaded expert data: {full_state.shape[0]} steps from {zarr_path}\n"
                f"           state_dim={full_state.shape[1]}, action_dim={action.shape[1]}"
            )

        def _soft_update_target(self):
            """Polyak averaging: θ_target ← τ·θ + (1-τ)·θ_target"""
            for p_target, p in zip(self.q_target.parameters(), self.q_critic.parameters()):
                p_target.data.mul_(1.0 - self.tau).add_(p.data * self.tau)

        def _add_to_replay(self, obs, actions, rewards, next_obs, dones):
            """Circular-buffer insertion of transition data."""
            device = obs.device
            n = obs.shape[0]

            if self._replay_obs is None:
                self._replay_obs = torch.zeros(self.replay_size, obs.shape[1], device=device)
                self._replay_actions = torch.zeros(self.replay_size, actions.shape[1], device=device)
                self._replay_rewards = torch.zeros(self.replay_size, 1, device=device)
                self._replay_next_obs = torch.zeros(self.replay_size, obs.shape[1], device=device)
                self._replay_dones = torch.zeros(self.replay_size, 1, device=device)

            for start in range(0, n, self.replay_size):
                end = min(start + self.replay_size, n)
                chunk = end - start
                space = self.replay_size - self._replay_ptr
                n_fit = min(chunk, space)
                idx_s = self._replay_ptr
                idx_e = idx_s + n_fit

                self._replay_obs[idx_s:idx_e] = obs[start:start + n_fit]
                self._replay_actions[idx_s:idx_e] = actions[start:start + n_fit]
                self._replay_rewards[idx_s:idx_e] = rewards[start:start + n_fit]
                self._replay_next_obs[idx_s:idx_e] = next_obs[start:start + n_fit]
                self._replay_dones[idx_s:idx_e] = dones[start:start + n_fit]

                self._replay_ptr = (self._replay_ptr + n_fit) % self.replay_size
                self._replay_filled += n_fit

                remaining = chunk - n_fit
                if remaining > 0:
                    self._replay_obs[:remaining] = obs[start + n_fit:start + n_fit + remaining]
                    self._replay_actions[:remaining] = actions[start + n_fit:start + n_fit + remaining]
                    self._replay_rewards[:remaining] = rewards[start + n_fit:start + n_fit + remaining]
                    self._replay_next_obs[:remaining] = next_obs[start + n_fit:start + n_fit + remaining]
                    self._replay_dones[:remaining] = dones[start + n_fit:start + n_fit + remaining]
                    self._replay_ptr = remaining
                    self._replay_filled += remaining

        def _get_replay_data(self):
            """Return all valid transition data in the replay buffer."""
            n_valid = min(self._replay_filled, self.replay_size)
            return (
                self._replay_obs[:n_valid],
                self._replay_actions[:n_valid],
                self._replay_rewards[:n_valid],
                self._replay_next_obs[:n_valid],
                self._replay_dones[:n_valid],
            )

        def _on_step(self) -> bool:
            return True

        def _on_rollout_end(self) -> None:
            self._rollout_count += 1
            buffer = self.model.rollout_buffer
            device = self.model.device

            # ── Extract transition data from rollout buffer ──────────
            obs_all = torch.as_tensor(buffer.observations, device=device)
            actions_all = torch.as_tensor(buffer.actions, device=device)
            rewards_all = torch.as_tensor(buffer.rewards, device=device)

            n_steps, n_envs = obs_all.shape[0], obs_all.shape[1]

            obs = obs_all[:-1].reshape(-1, obs_all.shape[-1])
            next_obs = obs_all[1:].reshape(-1, obs_all.shape[-1])
            actions = actions_all[:-1].reshape(-1, actions_all.shape[-1])
            rewards = rewards_all[:-1].reshape(-1, 1)

            ep_starts = torch.as_tensor(buffer.episode_starts, device=device)
            dones = ep_starts[1:].reshape(-1, 1).float()

            valid_mask = (dones.squeeze(-1) == 0)
            obs = obs[valid_mask]
            next_obs = next_obs[valid_mask]
            actions = actions[valid_mask]
            rewards = rewards[valid_mask]
            dones = dones[valid_mask]
            # ── Add transitions to replay buffer ─────────────────────
            self._add_to_replay(
                obs.detach(), actions.detach(), rewards.detach(),
                next_obs.detach(), dones.detach()
            )
            rep_obs, rep_actions, rep_rewards, rep_next_obs, rep_dones = self._get_replay_data()
            n_pos = rep_obs.shape[0]
            act_dim = actions.shape[1]

            # ── Pre-compute π(s') for ALL replay next-obs (ONCE) ─────
            # Avoids calling model.predict() inside every mini-batch.
            with torch.no_grad():
                policy = self.model.policy
                policy.set_training_mode(False)

                # Batch through policy in chunks to avoid OOM
                all_next_actions = torch.empty(n_pos, act_dim, device=device)
                precompute_batch = 4096
                for start in range(0, n_pos, precompute_batch):
                    end = min(start + precompute_batch, n_pos)
                    chunk_obs = rep_next_obs[start:end]
                    # Use policy.forward() directly on tensors — no numpy round-trip
                    dist = policy.get_distribution(policy.extract_features(chunk_obs))
                    all_next_actions[start:end] = dist.mode()

                # Pre-compute Q̂(s', π(s')) with target network
                all_q_next = torch.empty(n_pos, 1, device=device)
                for start in range(0, n_pos, precompute_batch):
                    end = min(start + precompute_batch, n_pos)
                    next_sa = torch.cat([rep_next_obs[start:end], all_next_actions[start:end]], dim=-1)
                    all_q_next[start:end] = self.q_target(next_sa)

                # Pre-compute all TD targets
                all_td_targets = rep_rewards + self.gamma * all_q_next 
            # ── Move expert data to device (once) ────────────────────
            has_expert = self._expert_obs is not None
            if has_expert:
                expert_obs_dev = self._expert_obs.to(device)
                expert_act_dev = self._expert_actions.to(device)
                n_expert = expert_obs_dev.shape[0]

            # ── Train Q-critic with TD loss + contrastive ranking ────
            self.q_critic.train()
            epoch_td_sum = 0.0
            epoch_rank_sum = 0.0
            epoch_expert_rank_sum = 0.0
            total_batches = 0

            for _ in range(max(1, self.q_epochs)):
                perm = torch.randperm(n_pos, device=device)

                for i in range(0, n_pos, self.mini_batch_size):
                    idx = perm[i:i + self.mini_batch_size]
                    b_obs = rep_obs[idx]
                    b_act = rep_actions[idx]
                    b_td_target = all_td_targets[idx]
                    bs = b_obs.shape[0]

                    # Q(s, a)
                    sa = torch.cat([b_obs, b_act], dim=-1)
                    q_pred = self.q_critic(sa)

                    # TD loss: ||Q(s,a) - [r + γQ̂(s', π(s'))]||²
                    loss_td = self.mse_fn(q_pred, b_td_target)

                    # ── Margin ranking loss on PPO data (contrastive) ──
                    loss_rank = torch.tensor(0.0, device=device)
                    if self.neg_samples > 0:
                        n_neg = min(bs, 256)
                        neg_idx = torch.randint(0, bs, (n_neg,), device=device)

                        n_noisy_neg = int(n_neg * (1.0 - self.random_neg_ratio))
                        n_rand_neg = n_neg - n_noisy_neg

                        neg_actions = torch.empty(n_neg, act_dim, device=device)
                        if n_noisy_neg > 0:
                            neg_actions[:n_noisy_neg] = b_act[neg_idx[:n_noisy_neg]] + \
                                torch.randn(n_noisy_neg, act_dim, device=device) * self.neg_noise_std
                        if n_rand_neg > 0:
                            neg_actions[n_noisy_neg:] = torch.rand(n_rand_neg, act_dim, device=device) * 2 - 1

                        neg_sa = torch.cat([b_obs[neg_idx], neg_actions], dim=-1)
                        q_neg = self.q_critic(neg_sa)

                        pos_sample = q_pred[neg_idx]
                        target = torch.ones(n_neg, device=device)
                        loss_rank = self.rank_fn(
                            pos_sample.squeeze(-1), q_neg.squeeze(-1), target
                        )

                    # ── Expert-state ranking loss (bridges distribution gap) ──
                    # Sample expert (s, a) pairs and enforce Q(s_expert, a_expert)
                    # > Q(s_expert, a_noisy/random) + margin on EXPERT states
                    loss_expert_rank = torch.tensor(0.0, device=device)
                    if has_expert and self.neg_samples > 0:
                        # Sample a batch from expert data
                        n_exp_batch = max(int(bs * self.expert_ratio), 32)
                        exp_idx = torch.randint(0, n_expert, (n_exp_batch,), device=device)
                        exp_obs = expert_obs_dev[exp_idx]
                        exp_act = expert_act_dev[exp_idx]

                        # Q(s_expert, a_expert)
                        exp_sa = torch.cat([exp_obs, exp_act], dim=-1)
                        q_exp_pos = self.q_critic(exp_sa)

                        # Generate negative actions on expert states
                        n_exp_noisy = int(n_exp_batch * (1.0 - self.random_neg_ratio))
                        n_exp_rand = n_exp_batch - n_exp_noisy

                        exp_neg_actions = torch.empty(n_exp_batch, act_dim, device=device)
                        if n_exp_noisy > 0:
                            exp_neg_actions[:n_exp_noisy] = exp_act[:n_exp_noisy] + \
                                torch.randn(n_exp_noisy, act_dim, device=device) * self.neg_noise_std
                        if n_exp_rand > 0:
                            exp_neg_actions[n_exp_noisy:] = torch.rand(n_exp_rand, act_dim, device=device) * 2 - 1

                        exp_neg_sa = torch.cat([exp_obs, exp_neg_actions], dim=-1)
                        q_exp_neg = self.q_critic(exp_neg_sa)

                        exp_target = torch.ones(n_exp_batch, device=device)
                        loss_expert_rank = self.rank_fn(
                            q_exp_pos.squeeze(-1), q_exp_neg.squeeze(-1), exp_target
                        )

                    loss = (loss_td
                            + self.margin_weight * loss_rank
                            + self.margin_weight * loss_expert_rank)

                    self.q_optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    self.q_optimizer.step()

                    epoch_td_sum += loss_td.item()
                    epoch_rank_sum += loss_rank.item()
                    epoch_expert_rank_sum += loss_expert_rank.item()
                    total_batches += 1

            # ── Soft-update target network ───────────────────────────
            self._soft_update_target()

            avg_td = epoch_td_sum / max(total_batches, 1)
            avg_rank = epoch_rank_sum / max(total_batches, 1)
            avg_expert_rank = epoch_expert_rank_sum / max(total_batches, 1)

            # ── Logging ──────────────────────────────────────────────
            self.logger.record("q_critic/td_loss", avg_td)
            self.logger.record("q_critic/rank_loss", avg_rank)
            self.logger.record("q_critic/expert_rank_loss", avg_expert_rank)
            self.logger.record("q_critic/replay_filled", min(self._replay_filled, self.replay_size))
            self.logger.record("q_critic/rollout_count", self._rollout_count)

            # Q-value gap diagnostic on PPO states
            n_diag = min(200, n_pos)
            with torch.no_grad():
                diag_obs = rep_obs[:n_diag]
                diag_act = rep_actions[:n_diag]
                q_onpol = self.q_critic(torch.cat([diag_obs, diag_act], dim=-1)).mean().item()
                q_rand = self.q_critic(
                    torch.cat([diag_obs, torch.rand(n_diag, act_dim, device=device) * 2 - 1], dim=-1)
                ).mean().item()
                q_noisy = self.q_critic(
                    torch.cat([diag_obs, diag_act + torch.randn(n_diag, act_dim, device=device) * 0.3], dim=-1)
                ).mean().item()

            q_gap_vs_noisy = q_onpol - q_noisy
            q_gap_vs_random = q_onpol - q_rand

            self.logger.record("q_critic/q_onpolicy", q_onpol)
            self.logger.record("q_critic/q_random_action", q_rand)
            self.logger.record("q_critic/q_noisy_action", q_noisy)
            self.logger.record("q_critic/q_gap_vs_random", q_gap_vs_random)
            self.logger.record("q_critic/q_gap_vs_noisy", q_gap_vs_noisy)

            # ── Expert-state Q-value gap diagnostic (THIS IS WHAT ACGD SEES) ──
            exp_q_expert = 0.0
            exp_q_noisy = 0.0
            exp_q_rand = 0.0
            exp_gap_vs_noisy = 0.0
            exp_gap_vs_random = 0.0
            if has_expert:
                n_exp_diag = min(200, n_expert)
                with torch.no_grad():
                    eidx = torch.randint(0, n_expert, (n_exp_diag,), device=device)
                    e_obs = expert_obs_dev[eidx]
                    e_act = expert_act_dev[eidx]
                    exp_q_expert = self.q_critic(
                        torch.cat([e_obs, e_act], dim=-1)
                    ).mean().item()
                    exp_q_noisy = self.q_critic(
                        torch.cat([e_obs, e_act + torch.randn(n_exp_diag, act_dim, device=device) * 0.3], dim=-1)
                    ).mean().item()
                    exp_q_rand = self.q_critic(
                        torch.cat([e_obs, torch.rand(n_exp_diag, act_dim, device=device) * 2 - 1], dim=-1)
                    ).mean().item()

                exp_gap_vs_noisy = exp_q_expert - exp_q_noisy
                exp_gap_vs_random = exp_q_expert - exp_q_rand

                self.logger.record("q_critic/expert_q_expert_action", exp_q_expert)
                self.logger.record("q_critic/expert_q_noisy_action", exp_q_noisy)
                self.logger.record("q_critic/expert_q_random_action", exp_q_rand)
                self.logger.record("q_critic/expert_gap_vs_noisy", exp_gap_vs_noisy)
                self.logger.record("q_critic/expert_gap_vs_random", exp_gap_vs_random)

            # Save best critic based on EXPERT-state gap (what ACGD actually uses)
            save_gap = exp_gap_vs_noisy if has_expert else q_gap_vs_noisy
            if self.save_dir is not None and save_gap > self._best_q_gap_vs_noisy:
                self._best_q_gap_vs_noisy = save_gap
                best_critic_path = self.save_dir / "best_critic_quality_q.pt"
                torch.save(self.q_critic.state_dict(), str(best_critic_path))
                self.logger.record("q_critic/best_q_gap_vs_noisy", save_gap)

            if self._rollout_count % 10 == 0:
                expert_str = ""
                if has_expert:
                    expert_str = (
                        f" | EXP: Q(exp)={exp_q_expert:.1f} Q(noisy)={exp_q_noisy:.1f} "
                        f"Q(rand)={exp_q_rand:.1f} gap={exp_gap_vs_noisy:.3f}"
                    )
                print(
                    f"[Q-Critic TD] rollout={self._rollout_count} "
                    f"td={avg_td:.4f} rank={avg_rank:.4f} exp_rank={avg_expert_rank:.4f} "
                    f"PPO: Q(on)={q_onpol:.1f} Q(rand)={q_rand:.1f} gap={q_gap_vs_noisy:.3f}"
                    f"{expert_str} "
                    f"replay={min(self._replay_filled, self.replay_size)}"
                )

    q_cb = None
    q_critic = None
    if args.train_q_critic:
        state_dim, action_dim = _acgd_qcritic_state_action_dims(vec_env.observation_space, vec_env.action_space)
        q_critic = _build_acgd_qcritic(state_dim, action_dim, int(args.q_critic_hidden_dim))
        q_optimizer = torch.optim.Adam(q_critic.parameters(), lr=float(args.q_critic_lr))
        # Resolve expert data path relative to repo root
        expert_data_path = str(repo_root / args.expert_data_path)
        q_cb = TrainQCriticCallback(
            q_critic=q_critic,
            q_optimizer=q_optimizer,
            q_epochs=int(args.q_critic_epochs),
            neg_samples=int(args.q_critic_neg_samples),
            neg_noise_std=float(args.q_critic_neg_noise_std),
            random_neg_ratio=float(args.q_critic_random_neg_ratio),
            replay_size=int(args.q_critic_replay_size),
            mini_batch_size=int(args.q_critic_mini_batch_size),
            margin=float(args.q_critic_margin),
            margin_weight=float(args.q_critic_margin_weight),
            gamma=0.99,
            target_update_freq=5,
            expert_data_path=expert_data_path,
            task_name=args.task,
            expert_ratio=float(args.q_critic_expert_ratio),
            save_dir=save_dir,
            verbose=0,
        )


    eval_cb = PeriodicEvalBestCallback(
        eval_env=eval_env,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.eval_episodes,
        save_dir=save_dir,
        best_metric=args.best_metric,
        q_critic=q_critic, 
        q_critic_name="best_expert_critic_q.pt",
        verbose=1,
    )
    callbacks = [eval_cb]
    if success_cb is not None:
        callbacks.append(success_cb)
    if q_cb is not None:
        callbacks.append(q_cb)
    callback = CallbackList(callbacks) if len(callbacks) > 1 else callbacks[0]

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        verbose=1,
        tensorboard_log=tensorboard_log,
        learning_rate=3e-4,
        batch_size=64,
        n_epochs=16,
        ent_coef=0.01,
        clip_range=0.2,              
        # use_sde=True,
        # sde_sample_freq=16,
        policy_kwargs=dict(
            net_arch=[dict(pi=[128, 128], vf=[128, 128])],
            activation_fn=nn.Tanh,
        ),  
        device=args.device,
    )

    # Now that the SB3 model exists, align q_critic device with model.device (if enabled).
    # Now that the SB3 model exists, align q_critic device with model.device (if enabled).
    if q_critic is not None:
        q_critic.to(model.device)
        # CRITICAL: Also move target network to the same device
        if q_cb is not None:
            q_cb.q_target.to(model.device)

    model.learn(
        total_timesteps=args.total_timesteps,
        progress_bar=True,
        callback=callback,
    )

    model_path = save_dir / "ppo_model.zip"
    model.save(str(model_path))
    if args.save_critic:
        save_critic(model, save_dir / "critic.pt")
    if args.train_q_critic and q_critic is not None:
        q_path = save_dir / args.q_critic_save_name
        torch.save(q_critic.state_dict(), str(q_path))
        print(f"[Saved Q-Critic] {q_path}")

    if success_cb is not None:
        n_eps = len(success_cb.episode_lengths)
        n_succ = len(success_cb.success_steps)
        succ_rate = (n_succ / n_eps) if n_eps > 0 else 0.0
        if n_succ > 0:
            s = np.array(success_cb.success_steps, dtype=np.float32)
            mean_s = float(np.mean(s))
            med_s = float(np.median(s))
            p90_s = float(np.percentile(s, 90))
            print(
                f"[Train Success Horizon] episodes={n_eps} success_episodes={n_succ} "
                f"success_rate={succ_rate:.3f} mean_step={mean_s:.1f} median_step={med_s:.1f} p90_step={p90_s:.1f}"
            )
        else:
            print(f"[Train Success Horizon] episodes={n_eps} success_episodes=0 success_rate={succ_rate:.3f}")

    # Final evaluation: reward + success rate (success is from MetaWorld info["success"])
    episode_rewards = []
    episode_success = []
    episode_success_steps = []  # first-success step per episode (-1 if failed)
    success_key_missing_steps = 0
    success_key_total_steps = 0
    for ep_i in range(args.eval_episodes):
        obs = eval_env.reset()
        done = np.array([False])
        ep_rew = 0.0
        ep_succ = False
        first_success_step = -1
        step_count = 0
        while not bool(done[0]):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, infos = eval_env.step(action)
            ep_rew += float(reward[0])
            step_count += 1
            info0 = infos[0] if isinstance(infos, (list, tuple)) and len(infos) > 0 else {}
            if isinstance(info0, dict):
                success_key_total_steps += 1
                if "success" not in info0:
                    success_key_missing_steps += 1
                elif bool(info0.get("success", False)):
                    if not ep_succ:
                        first_success_step = step_count
                    ep_succ = True
        episode_rewards.append(ep_rew)
        episode_success.append(1.0 if ep_succ else 0.0)
        episode_success_steps.append(first_success_step)
        status = f"SUCCESS at step {first_success_step}/{step_count}" if ep_succ else f"FAILED ({step_count} steps)"
        print(f"  [Eval ep {ep_i:02d}] reward={ep_rew:.1f} {status}")

    mean_reward = float(np.mean(episode_rewards)) if len(episode_rewards) else 0.0
    std_reward = float(np.std(episode_rewards)) if len(episode_rewards) else 0.0
    success_rate = float(np.mean(episode_success)) if len(episode_success) else 0.0

    # Success step statistics
    succ_steps = [s for s in episode_success_steps if s > 0]
    if succ_steps:
        print(
            f"[Eval Success Steps] mean={np.mean(succ_steps):.1f} "
            f"median={np.median(succ_steps):.1f} min={min(succ_steps)} max={max(succ_steps)}"
        )

    if success_key_total_steps > 0 and success_key_missing_steps == success_key_total_steps:
        print(
            "[Eval Warn] `info['success']` was missing for all eval steps. "
            "This usually means you're using an env variant that does not populate success in info."
        )
    print(
        f"[Eval] task={args.task} episodes={args.eval_episodes} "
        f"mean_reward={mean_reward:.3f} std={std_reward:.3f} success_rate={success_rate:.3f}"
    )
    print(f"[Saved] {model_path}")


if __name__ == "__main__":
    main()
