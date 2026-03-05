#!/usr/bin/env python3
"""
Single-task SAC training on MetaWorld using vendored Stable-Baselines3.

SAC (Soft Actor-Critic) benefits over PPO for critic/data collection:
- Off-policy: learns from diverse replay buffer data
- Entropy-maximizing: explores more of the action space
- Built-in twin Q-critics: better value estimation
- Better data diversity for downstream IQL/ACGD critic training

Includes optional auxiliary Q(s,a) critic training (same as PPO script)
with contrastive ranking, expert data mixing, and TD learning.

Example:
  # Basic SAC training
  python scripts/train_sac_metaworld.py --task reach --total-timesteps 1000000

  # With auxiliary Q-critic and tensorboard
  python scripts/train_sac_metaworld.py --task pick-place --total-timesteps 2000000 \
      --train-q-critic --tensorboard

  # With expert data mixing for Q-critic
  python scripts/train_sac_metaworld.py --task pick-place --total-timesteps 2000000 \
      --train-q-critic --expert-data-path ManiFlow/data --tensorboard
"""

import argparse
import os
import sys
from pathlib import Path
import importlib.util
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


def _acgd_qcritic_state_action_dims(obs_space, act_space) -> tuple:
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


def _save_sac_twin_critic(model, save_path: Path) -> None:
    """
    Extract SAC's twin Q-networks (qf0, qf1) and save them as a TwinQCritic
    state_dict compatible with the workspace loading code.
    Keys are prefixed with 'q0.' and 'q1.' to match TwinQCritic(q0=..., q1=...).
    """
    twin_state = {}
    for k, v in model.critic.qf0.state_dict().items():
        twin_state[f"q0.{k}"] = v.cpu()
    for k, v in model.critic.qf1.state_dict().items():
        twin_state[f"q1.{k}"] = v.cpu()
    torch.save(twin_state, str(save_path))
    print(f"  [Twin-Q] Saved SAC twin critic -> {save_path}")


def _ensure_sb3_importable(repo_root: Path) -> None:
    """
    Prefer system-installed stable_baselines3. If missing, fall back to vendored copy.
    """
    try:
        import stable_baselines3  # noqa: F401
        return
    except Exception:
        pass
    vendored_parent = repo_root / "third_party" / "dexart-release"
    sys.path.insert(0, str(vendored_parent))


def _import_from_path(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module {module_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _ensure_ld_library_path_on_process_start(mujoco_bin: str) -> None:
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


def parse_args():
    p = argparse.ArgumentParser(description="Single-task SAC training on MetaWorld")

    # ── Environment / general ──
    p.add_argument("--task", type=str, default="reach",
                   help="MetaWorld task name (e.g., reach, pick-place, coffee-pull, ...)")
    p.add_argument("--total-timesteps", type=int, default=10_000_000)
    p.add_argument("--n-envs", type=int, default=16,
                   help="Number of parallel envs (SAC supports multi-env with step-based training)")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", type=str, default="auto", help="SB3 device: auto|cpu|cuda")
    p.add_argument("--log-dir", type=str, default="runs/sb3_metaworld_sac")
    p.add_argument("--save-dir", type=str, default="runs/sb3_metaworld_sac/models_1")
    p.add_argument("--monitor-dir", type=str, default="",
                   help="If set, write Monitor CSVs for reward curves.")
    p.add_argument("--tensorboard", action="store_true", default=True, help="Enable tensorboard logging")
    p.add_argument("--eval-episodes", type=int, default=20)
    p.add_argument("--eval-freq", type=int, default=200_000,
                   help="Evaluate every N timesteps during training (0 disables).")
    p.add_argument("--best-metric", type=str, default="success_rate",
                   choices=["success_rate", "mean_reward"])
    p.add_argument("--mujoco-bin", type=str, default="~/.mujoco/mujoco210/bin")

    # ── SAC hyperparameters ──
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--buffer-size", type=int, default=1_000_000,
                   help="Replay buffer size for SAC.")
    p.add_argument("--learning-starts", type=int, default=5000,
                   help="Number of random exploration steps before SAC starts training.")
    p.add_argument("--batch-size", type=int, default=256,
                   help="Mini-batch size for SAC gradient updates.")
    p.add_argument("--tau", type=float, default=0.005,
                   help="Polyak averaging coefficient for target network updates.")
    p.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
    p.add_argument("--train-freq", type=int, default=1,
                   help="Update the model every N environment steps.")
    p.add_argument("--gradient-steps", type=int, default=1,
                   help="Gradient steps per rollout (-1 = as many as steps collected).")
    p.add_argument("--ent-coef", type=str, default="auto",
                   help="Entropy coefficient. 'auto' for automatic tuning, or a float value.")
    p.add_argument("--target-entropy", type=str, default="auto",
                   help="Target entropy for automatic ent_coef tuning. 'auto' or a float.")
    p.add_argument("--net-arch", type=int, nargs="+", default=[256, 256],
                   help="Hidden layer sizes for actor and critic (e.g., 256 256).")

    # ── Auxiliary Q-critic training ──
    p.add_argument("--train-q-critic", action="store_true",
                   help="Train an auxiliary Q(s,a) critic alongside SAC.")
    p.add_argument("--q-critic-lr", type=float, default=3e-4)
    p.add_argument("--q-critic-hidden-dim", type=int, default=256)
    p.add_argument("--q-critic-save-name", type=str, default="expert_critic_q.pt")
    p.add_argument("--q-critic-train-freq", type=int, default=2000,
                   help="Train auxiliary Q-critic every N env steps.")
    p.add_argument("--q-critic-epochs", type=int, default=20,
                   help="Gradient epochs per Q-critic training round.")
    p.add_argument("--q-critic-neg-samples", type=int, default=6)
    p.add_argument("--q-critic-neg-noise-std", type=float, default=0.3)
    p.add_argument("--q-critic-random-neg-ratio", type=float, default=0.5)
    p.add_argument("--q-critic-mini-batch-size", type=int, default=2048)
    p.add_argument("--q-critic-margin", type=float, default=5.0)
    p.add_argument("--q-critic-margin-weight", type=float, default=5.0)

    # ── Expert data mixing ──
    p.add_argument("--expert-data-path", type=str, default="ManiFlow/data",
                   help="Path to directory containing metaworld_<task>_expert.zarr files.")
    p.add_argument("--q-critic-expert-ratio", type=float, default=0.3,
                   help="Fraction of Q-critic mini-batch from expert data (0.3 = 30%%).")

    return p.parse_args()


def main():
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[1]

    # MuJoCo runtime
    mujoco_bin = os.path.expanduser(args.mujoco_bin)
    _ensure_ld_library_path_on_process_start(mujoco_bin)

    # Ensure MetaWorld imports correct gym
    sys.path.insert(0, str(repo_root / "third_party" / "gym-0.21.0"))
    sys.path.insert(0, str(repo_root / "third_party" / "Metaworld"))

    _ensure_sb3_importable(repo_root)

    # Import env
    env_mod = _import_from_path(
        "sb3_metaworld_state_env",
        repo_root / "ManiFlow" / "maniflow" / "env" / "metaworld" / "sb3_metaworld_state_env.py",
    )
    SB3MetaWorldStateEnv = env_mod.SB3MetaWorldStateEnv

    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import BaseCallback, CallbackList
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

    log_dir = repo_root / args.log_dir / args.task
    save_dir = repo_root / args.save_dir / args.task
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    tensorboard_log = None
    if args.tensorboard:
        try:
            import tensorboard  # noqa: F401
            tensorboard_log = str(log_dir)
        except Exception:
            print("[Warn] --tensorboard set but tensorboard is not installed; disabling.")

    # ── Environment setup ──
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

    # Separate eval env
    eval_env = DummyVecEnv([make_env(10_000)])

    # ══════════════════════════════════════════════════════════════════
    # Periodic Evaluation Callback
    # ══════════════════════════════════════════════════════════════════
    class PeriodicEvalBestCallback(BaseCallback):
        """
        Every `eval_freq` timesteps:
        - runs evaluation for N episodes
        - computes mean_reward + success_rate (MetaWorld info['success'])
        - saves best model to best_model.zip
        """

        def __init__(
            self,
            eval_env,
            eval_freq: int,
            n_eval_episodes: int,
            save_dir: Path,
            best_metric: str = "success_rate",
            q_critic: Optional[nn.Module] = None,
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
            self.q_critic = q_critic
            self.q_critic_name = q_critic_name
            self._saved_85_count = 0
            self._max_85_saves = 5

        def _init_callback(self) -> None:
            os.makedirs(self.save_dir, exist_ok=True)

        def _evaluate(self):
            episode_rewards = []
            episode_success = []

            for _ in range(self.n_eval_episodes):
                obs = self.eval_env.reset()
                done = np.array([False])
                ep_rew = 0.0
                ep_succ = False
                while not bool(done[0]):
                    action, _ = self.model.predict(obs, deterministic=False)
                    obs, reward, done, infos = self.eval_env.step(action)
                    ep_rew += float(reward[0])
                    info0 = infos[0] if isinstance(infos, (list, tuple)) and len(infos) > 0 else {}
                    if isinstance(info0, dict) and bool(info0.get("success", False)):
                        ep_succ = True
                episode_rewards.append(ep_rew)
                episode_success.append(1.0 if ep_succ else 0.0)

            mean_reward = float(np.mean(episode_rewards)) if episode_rewards else 0.0
            success_rate = float(np.mean(episode_success)) if episode_success else 0.0
            return mean_reward, success_rate

        def _on_step(self) -> bool:
            if self.eval_freq <= 0:
                return True
            if (self.num_timesteps - self.last_eval_t) < self.eval_freq:
                return True

            self.last_eval_t = int(self.num_timesteps)
            mean_reward, success_rate = self._evaluate()

            if success_rate >= 0.75 and self._saved_85_count < self._max_85_saves:
                self._saved_85_count += 1
                ckpt_path = self.save_dir / f"diverse_85_{self._saved_85_count}_t{self.num_timesteps}.zip"
                self.model.save(str(ckpt_path))
                if self.verbose:
                    print(f"  [Diverse 85%] #{self._saved_85_count}/{self._max_85_saves} "
                          f"success={success_rate:.2f} → {ckpt_path.name}")

            # Share success rate with Q-critic callback
            if q_cb is not None:
                q_cb._last_eval_success_rate = success_rate

            if self.verbose:
                print(
                    f"[Eval] t={self.num_timesteps} episodes={self.n_eval_episodes} "
                    f"mean_reward={mean_reward:.3f} success_rate={success_rate:.3f}"
                )

            self.logger.record("eval/mean_reward", float(mean_reward))
            self.logger.record("eval/success_rate", float(success_rate))
            self.logger.dump(self.num_timesteps)

            value = success_rate if self.best_metric == "success_rate" else mean_reward
            if value >= self.best_value:
                self.best_value = float(value)
                best_path = self.save_dir / "best_model.zip"
                self.model.save(str(best_path))
                # Save SAC's twin Q-critics as min(Q1,Q2)
                _save_sac_twin_critic(self.model, self.save_dir / "sac_twin_critic_q.pt")
                if self.q_critic is not None:
                    q_path = self.save_dir / self.q_critic_name
                    torch.save(self.q_critic.state_dict(), str(q_path))
                    if self.verbose:
                        print(f"  [Best] Q-Critic → {q_path}")
                if self.verbose:
                    print(f"  [Best] metric={self.best_metric} value={value:.3f} → {best_path}")

            # ── Save top-3 diverse models (success >= 0.9, highest entropy) ──
            if success_rate >= 0.9:
                # Get current entropy coefficient
                ent_coef = float(self.model.log_ent_coef.exp().item())
                # self._diverse_models: list of (ent_coef, timestep) sorted by ent_coef desc
                if not hasattr(self, "_diverse_models"):
                    self._diverse_models = []

                # Save candidate
                tag = f"diverse_ent{ent_coef:.5f}_t{self.num_timesteps}"
                cand_path = self.save_dir / f"{tag}.zip"
                self.model.save(str(cand_path))
                self._diverse_models.append((ent_coef, self.num_timesteps, cand_path))

                # Keep only top 3 by entropy (descending)
                self._diverse_models.sort(key=lambda x: x[0], reverse=True)
                while len(self._diverse_models) > 3:
                    _, _, old_path = self._diverse_models.pop()
                    if old_path.exists():
                        old_path.unlink()  # delete evicted model

                if self.verbose:
                    print(f"  [Diverse] success={success_rate:.2f} ent_coef={ent_coef:.5f} "
                          f"saved={cand_path.name} (top-3: {[f'{e:.5f}' for e,_,_ in self._diverse_models]})")

            return True

    # ══════════════════════════════════════════════════════════════════
    # Auxiliary Q(s,a) Critic Callback (adapted for SAC replay buffer)
    #
    # SAC already has twin Q-critics internally. This trains a SEPARATE
    # simple MLP Q-critic in the format expected by ACGD/IQL pipeline.
    # It samples transitions from SAC's replay buffer for training.
    # ══════════════════════════════════════════════════════════════════
    class TrainQCriticSACCallback(BaseCallback):
        """
        Auxiliary Q(s,a) critic trained alongside SAC.
        Samples from SAC's replay buffer (off-policy data) for training.
        """

        def __init__(
            self,
            q_critic: nn.Module,
            q_optimizer: torch.optim.Optimizer,
            train_freq: int = 2000,
            q_epochs: int = 20,
            neg_samples: int = 6,
            neg_noise_std: float = 0.3,
            random_neg_ratio: float = 0.5,
            mini_batch_size: int = 2048,
            margin: float = 5.0,
            margin_weight: float = 5.0,
            gamma: float = 0.99,
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
            self.train_freq = int(train_freq)
            self.q_epochs = int(q_epochs)
            self.neg_samples = int(neg_samples)
            self.neg_noise_std = float(neg_noise_std)
            self.random_neg_ratio = float(random_neg_ratio)
            self.mini_batch_size = int(mini_batch_size)
            self.margin = float(margin)
            self.margin_weight = float(margin_weight)
            self.gamma = float(gamma)
            self.tau = float(tau)
            self.expert_ratio = float(expert_ratio)
            self.mse_fn = nn.MSELoss()
            self.rank_fn = nn.MarginRankingLoss(margin=self.margin)

            # Target network
            import copy
            self.q_target = copy.deepcopy(q_critic)
            self.q_target.requires_grad_(False)

            # Best-critic tracking
            self.save_dir = save_dir
            self._best_q_gap_vs_noisy = -float("inf")
            self._last_eval_success_rate = 0.0
            self._last_train_t = 0
            self._train_count = 0

            # Expert data
            self._expert_obs = None
            self._expert_actions = None
            if expert_data_path and task_name:
                self._load_expert_data(expert_data_path, task_name)

        def _load_expert_data(self, data_path: str, task_name: str):
            """Load full_state and action from the expert zarr dataset."""
            import zarr
            zarr_path = Path(data_path) / f"metaworld_{task_name}_expert.zarr"
            if not zarr_path.exists():
                print(f"[Q-Critic] WARNING: Expert zarr not found at {zarr_path}. "
                      f"Critic trains on SAC replay data only.")
                return
            root = zarr.open(str(zarr_path), mode="r")
            full_state = np.array(root["data"]["full_state"])
            action = np.array(root["data"]["action"])
            self._expert_obs = torch.tensor(full_state, dtype=torch.float32)
            self._expert_actions = torch.tensor(action, dtype=torch.float32)
            print(
                f"[Q-Critic] Loaded expert data: {full_state.shape[0]} steps from {zarr_path}\n"
                f"           state_dim={full_state.shape[1]}, action_dim={action.shape[1]}"
            )

        def _soft_update_target(self):
            for p_target, p in zip(self.q_target.parameters(), self.q_critic.parameters()):
                p_target.data.mul_(1.0 - self.tau).add_(p.data * self.tau)

        def _on_step(self) -> bool:
            # Only train periodically and after SAC has started learning
            if (self.num_timesteps - self._last_train_t) < self.train_freq:
                return True

            replay_buffer = self.model.replay_buffer
            if replay_buffer.size() < self.mini_batch_size:
                return True

            self._last_train_t = int(self.num_timesteps)
            self._train_count += 1
            device = self.model.device

            # Move expert data to device
            has_expert = self._expert_obs is not None
            if has_expert:
                expert_obs_dev = self._expert_obs.to(device)
                expert_act_dev = self._expert_actions.to(device)
                n_expert = expert_obs_dev.shape[0]

            # ── Sample from SAC replay buffer for training ──
            self.q_critic.train()
            epoch_td_sum = 0.0
            epoch_rank_sum = 0.0
            epoch_expert_rank_sum = 0.0
            total_batches = 0

            # How many transitions to use per training round
            n_train_samples = min(replay_buffer.size(), 50_000)

            for _ in range(max(1, self.q_epochs)):
                # Sample a large batch from SAC's replay buffer
                replay_data = replay_buffer.sample(
                    min(self.mini_batch_size, n_train_samples)
                )

                b_obs = replay_data.observations
                b_act = replay_data.actions
                b_next_obs = replay_data.next_observations
                b_rewards = replay_data.rewards
                b_dones = replay_data.dones
                bs = b_obs.shape[0]
                act_dim = b_act.shape[1]

                # ── TD target using target network ──
                with torch.no_grad():
                    # Use SAC's actor to get next actions for TD target
                    next_actions, _ = self.model.actor.action_log_prob(b_next_obs)
                    next_sa = torch.cat([b_next_obs, next_actions], dim=-1)
                    q_next = self.q_target(next_sa)
                    td_target = b_rewards + self.gamma * (1.0 - b_dones) * q_next

                # Q(s, a)
                sa = torch.cat([b_obs, b_act], dim=-1)
                q_pred = self.q_critic(sa)
                loss_td = self.mse_fn(q_pred, td_target)

                # ── Margin ranking loss (contrastive) ──
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

                # ── Expert ranking loss ──
                loss_expert_rank = torch.tensor(0.0, device=device)
                if has_expert and self.neg_samples > 0:
                    n_exp_batch = max(int(bs * self.expert_ratio), 32)
                    exp_idx = torch.randint(0, n_expert, (n_exp_batch,), device=device)
                    exp_obs = expert_obs_dev[exp_idx]
                    exp_act = expert_act_dev[exp_idx]

                    exp_sa = torch.cat([exp_obs, exp_act], dim=-1)
                    q_exp_pos = self.q_critic(exp_sa)

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

            # Soft-update target network
            self._soft_update_target()

            avg_td = epoch_td_sum / max(total_batches, 1)
            avg_rank = epoch_rank_sum / max(total_batches, 1)
            avg_expert_rank = epoch_expert_rank_sum / max(total_batches, 1)

            # ── Diagnostics ──
            n_diag = min(200, replay_buffer.size())
            diag_data = replay_buffer.sample(n_diag)
            diag_obs = diag_data.observations
            diag_act = diag_data.actions
            act_dim = diag_act.shape[1]

            with torch.no_grad():
                q_onpol = self.q_critic(
                    torch.cat([diag_obs, diag_act], dim=-1)
                ).mean().item()
                q_rand = self.q_critic(
                    torch.cat([diag_obs, torch.rand(n_diag, act_dim, device=device) * 2 - 1], dim=-1)
                ).mean().item()
                q_noisy = self.q_critic(
                    torch.cat([diag_obs, diag_act + torch.randn(n_diag, act_dim, device=device) * 0.3], dim=-1)
                ).mean().item()

            q_gap_vs_noisy = q_onpol - q_noisy
            q_gap_vs_random = q_onpol - q_rand

            self.logger.record("q_critic/td_loss", avg_td)
            self.logger.record("q_critic/rank_loss", avg_rank)
            self.logger.record("q_critic/expert_rank_loss", avg_expert_rank)
            self.logger.record("q_critic/q_onpolicy", q_onpol)
            self.logger.record("q_critic/q_random_action", q_rand)
            self.logger.record("q_critic/q_noisy_action", q_noisy)
            self.logger.record("q_critic/q_gap_vs_random", q_gap_vs_random)
            self.logger.record("q_critic/q_gap_vs_noisy", q_gap_vs_noisy)
            self.logger.record("q_critic/replay_size", replay_buffer.size())
            self.logger.record("q_critic/train_count", self._train_count)

            # Expert diagnostics
            exp_gap_vs_noisy = 0.0
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

            # Save best critic
            save_gap = exp_gap_vs_noisy if has_expert else q_gap_vs_noisy
            if self.save_dir is not None and save_gap > self._best_q_gap_vs_noisy:
                self._best_q_gap_vs_noisy = save_gap
                best_path = self.save_dir / "best_critic_quality_q.pt"
                torch.save(self.q_critic.state_dict(), str(best_path))
                self.logger.record("q_critic/best_q_gap_vs_noisy", save_gap)

            if self._train_count % 10 == 0:
                expert_str = ""
                if has_expert:
                    expert_str = (
                        f" | EXP: Q(exp)={exp_q_expert:.1f} Q(noisy)={exp_q_noisy:.1f} "
                        f"Q(rand)={exp_q_rand:.1f} gap={exp_gap_vs_noisy:.3f}"
                    )
                print(
                    f"[Q-Critic] t={self.num_timesteps} round={self._train_count} "
                    f"td={avg_td:.4f} rank={avg_rank:.4f} exp_rank={avg_expert_rank:.4f} "
                    f"SAC: Q(on)={q_onpol:.1f} Q(rand)={q_rand:.1f} gap={q_gap_vs_noisy:.3f}"
                    f"{expert_str} "
                    f"replay={replay_buffer.size()}"
                )

            return True

    # ── Build Q-critic (optional) ──
    q_cb = None
    q_critic = None
    if args.train_q_critic:
        state_dim, action_dim = _acgd_qcritic_state_action_dims(
            vec_env.observation_space, vec_env.action_space
        )
        q_critic = _build_acgd_qcritic(state_dim, action_dim, args.q_critic_hidden_dim)
        q_optimizer = torch.optim.Adam(q_critic.parameters(), lr=args.q_critic_lr)
        expert_data_path = str(repo_root / args.expert_data_path)
        q_cb = TrainQCriticSACCallback(
            q_critic=q_critic,
            q_optimizer=q_optimizer,
            train_freq=args.q_critic_train_freq,
            q_epochs=args.q_critic_epochs,
            neg_samples=args.q_critic_neg_samples,
            neg_noise_std=args.q_critic_neg_noise_std,
            random_neg_ratio=args.q_critic_random_neg_ratio,
            mini_batch_size=args.q_critic_mini_batch_size,
            margin=args.q_critic_margin,
            margin_weight=args.q_critic_margin_weight,
            gamma=args.gamma,
            tau=args.tau,
            expert_data_path=expert_data_path,
            task_name=args.task,
            expert_ratio=args.q_critic_expert_ratio,
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
    if q_cb is not None:
        callbacks.append(q_cb)
    callback = CallbackList(callbacks) if len(callbacks) > 1 else callbacks[0]

    # ── Parse ent_coef ──
    ent_coef = args.ent_coef
    try:
        ent_coef = float(ent_coef)
    except ValueError:
        pass  # Keep as string (e.g., "auto")

    target_entropy = args.target_entropy
    try:
        target_entropy = float(target_entropy)
    except ValueError:
        pass  # Keep as string (e.g., "auto")

    # ── Create SAC model ──
    model = SAC(
        policy="MlpPolicy",
        env=vec_env,
        verbose=1,
        tensorboard_log=tensorboard_log,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        tau=args.tau,
        gamma=args.gamma,
        train_freq=args.train_freq,
        gradient_steps=args.gradient_steps,
        ent_coef=ent_coef,
        target_entropy=target_entropy,
        policy_kwargs=dict(
            net_arch=args.net_arch,
            activation_fn=nn.ReLU,
        ),
        device=args.device,
        seed=args.seed,
    )

    print(f"\n[SAC] Task: {args.task}")
    print(f"[SAC] Total timesteps: {args.total_timesteps}")
    print(f"[SAC] Buffer size: {args.buffer_size}")
    print(f"[SAC] Learning starts: {args.learning_starts}")
    print(f"[SAC] Batch size: {args.batch_size}")
    print(f"[SAC] Net arch: {args.net_arch}")
    print(f"[SAC] Ent coef: {ent_coef}")
    print(f"[SAC] Device: {model.device}")
    if args.train_q_critic:
        print(f"[SAC] Auxiliary Q-critic: ON (hidden_dim={args.q_critic_hidden_dim}, "
              f"train_freq={args.q_critic_train_freq})")
    print()

    # Move auxiliary Q-critic to same device
    if q_critic is not None:
        q_critic.to(model.device)
        if q_cb is not None:
            q_cb.q_target.to(model.device)

    # ── Train ──
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callback,
        log_interval=4,
    )

    # ── Save final model ──
    model_path = save_dir / "sac_model.zip"
    model.save(str(model_path))
    _save_sac_twin_critic(model, save_dir / "sac_twin_critic_q.pt")
    if args.train_q_critic and q_critic is not None:
        q_path = save_dir / args.q_critic_save_name
        torch.save(q_critic.state_dict(), str(q_path))
        print(f"[Saved Q-Critic] {q_path}")

    # ── Final evaluation ──
    episode_rewards = []
    episode_success = []
    episode_success_steps = []
    for ep_i in range(args.eval_episodes):
        obs = eval_env.reset()
        done = np.array([False])
        ep_rew = 0.0
        ep_succ = False
        first_success_step = -1
        step_count = 0
        while not bool(done[0]):
            action, _ = model.predict(obs, deterministic=False)
            obs, reward, done, infos = eval_env.step(action)
            ep_rew += float(reward[0])
            step_count += 1
            info0 = infos[0] if isinstance(infos, (list, tuple)) and len(infos) > 0 else {}
            if isinstance(info0, dict) and bool(info0.get("success", False)):
                if not ep_succ:
                    first_success_step = step_count
                ep_succ = True
        episode_rewards.append(ep_rew)
        episode_success.append(1.0 if ep_succ else 0.0)
        episode_success_steps.append(first_success_step)
        status = f"SUCCESS at step {first_success_step}/{step_count}" if ep_succ else f"FAILED ({step_count} steps)"
        print(f"  [Eval ep {ep_i:02d}] reward={ep_rew:.1f} {status}")

    mean_reward = float(np.mean(episode_rewards)) if episode_rewards else 0.0
    std_reward = float(np.std(episode_rewards)) if episode_rewards else 0.0
    success_rate = float(np.mean(episode_success)) if episode_success else 0.0

    succ_steps = [s for s in episode_success_steps if s > 0]
    if succ_steps:
        print(
            f"[Eval Success Steps] mean={np.mean(succ_steps):.1f} "
            f"median={np.median(succ_steps):.1f} min={min(succ_steps)} max={max(succ_steps)}"
        )

    print(
        f"\n[Final Eval] task={args.task} episodes={args.eval_episodes} "
        f"mean_reward={mean_reward:.3f} std={std_reward:.3f} success_rate={success_rate:.3f}"
    )
    print(f"[Saved] {model_path}")


if __name__ == "__main__":
    main()
