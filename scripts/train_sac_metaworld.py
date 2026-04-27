#!/usr/bin/env python3
"""
Single-task SAC training on MetaWorld using vendored Stable-Baselines3.

Example:
  # Basic SAC training
  python scripts/train_sac_metaworld.py --task reach --total-timesteps 1000000

  # Save replay buffer (with images) once success >= 80%
  python scripts/train_sac_metaworld.py --task pick-place --total-timesteps 2000000 \
      --save-replay-buffer --save-rb-at-success 0.8
"""

import argparse
import os
import sys
from pathlib import Path
import importlib.util

import numpy as np
import torch
import torch.nn as nn


METAWORLD_CAMERAS = {
    "default": "corner2",
}


def _save_replay_buffer_to_zarr(
    model,
    task_key: str,
    out_path: Path,
    env=None,
    image_size: int = 128,
    camera_name: str = "",
    device_id: int = 0,
    save_last_n: int = 0,
) -> None:
    """
    Dump the SB3 SAC replay buffer to a ManiFlow-compatible zarr dataset.

    Saves: full_state, state, action, reward, done, q_value, v_value, advantage,
           episode_ends, episode_success.
    If *env* has stored sim states, also renders and saves img + next_img.
    """
    import zarr

    replay_buffer = model.replay_buffer
    buf_size = replay_buffer.size()
    if buf_size == 0:
        print("[ReplayBuffer] Buffer is empty, skipping save.")
        return

    if save_last_n > 0 and save_last_n < buf_size:
        offset = buf_size - save_last_n
        save_size = save_last_n
        print(f"[ReplayBuffer] Saving last {save_last_n} of {buf_size} transitions")
    else:
        offset = 0
        save_size = buf_size

    device = model.device

    buf_capacity = replay_buffer.buffer_size
    if replay_buffer.full:
        pos = replay_buffer.pos
        chronological = np.concatenate([np.arange(pos, buf_capacity), np.arange(0, pos)])
        idx = chronological[offset:]
    else:
        idx = np.arange(offset, buf_size)

    obs_buf      = replay_buffer.observations[idx].squeeze(1)
    next_obs_buf = replay_buffer.next_observations[idx].squeeze(1)
    act_buf      = replay_buffer.actions[idx].squeeze(1)
    rew_buf      = replay_buffer.rewards[idx].squeeze(1)
    done_buf     = replay_buffer.dones[idx].squeeze(1)

    full_state_arr      = obs_buf.astype(np.float32)
    next_full_state_arr = next_obs_buf.astype(np.float32)
    act_arr  = act_buf.astype(np.float32)
    rew_arr  = rew_buf.astype(np.float32).ravel()
    done_arr = done_buf.astype(np.float32).ravel()

    # Reconstruct episode boundaries from done flags
    done_indices = np.where(done_arr > 0.5)[0]
    if len(done_indices) == 0:
        episode_ends = np.array([save_size], dtype=np.int64)
    else:
        episode_ends = (done_indices + 1).astype(np.int64)
        if episode_ends[-1] < save_size:
            episode_ends = np.append(episode_ends, save_size)

    # Compute Q(s,a), V(s), advantage using SAC twin critics
    print(f"[ReplayBuffer] Computing Q/V/advantage for {save_size} transitions ...")
    qf0 = model.critic.qf0
    qf1 = model.critic.qf1
    qf0.eval(); qf1.eval()

    q_value_arr = np.zeros(save_size, dtype=np.float32)
    v_value_arr = np.zeros(save_size, dtype=np.float32)

    batch = 4096
    with torch.no_grad():
        for start in range(0, save_size, batch):
            end = min(start + batch, save_size)
            s_t = torch.FloatTensor(full_state_arr[start:end]).to(device)
            a_t = torch.FloatTensor(act_arr[start:end]).to(device)
            sa  = torch.cat([s_t, a_t], dim=-1)
            q_value_arr[start:end] = torch.min(qf0(sa), qf1(sa)).squeeze(-1).cpu().numpy()
            pi_a, _ = model.actor.action_log_prob(s_t)
            sa_pi = torch.cat([s_t, pi_a], dim=-1)
            v_value_arr[start:end] = torch.min(qf0(sa_pi), qf1(sa_pi)).squeeze(-1).cpu().numpy()

    adv_arr = q_value_arr - v_value_arr

    if out_path.exists():
        import shutil
        shutil.rmtree(out_path)
        print(f"[ReplayBuffer] Removed existing {out_path}")

    zarr_root = zarr.group(str(out_path))
    zarr_data = zarr_root.create_group("data")
    zarr_meta = zarr_root.create_group("meta")
    compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=1)

    def _save(name, data, chunks=None, dtype=None):
        dtype = dtype or data.dtype
        if chunks is None:
            chunks = (min(100, len(data)),) + data.shape[1:]
        zarr_data.create_dataset(
            name, data=data, chunks=chunks, dtype=str(dtype),
            overwrite=True, compressor=compressor,
        )

    # agent_pos = [eef_pos(3), finger_right(3), finger_left(3)] stored per step in env ring buffer
    if env is None or getattr(env, "_agent_pos_buf", None) is None:
        raise RuntimeError(
            "agent_pos ring buffer not available. "
            "Make sure --n-envs 1 and store_sim_states=True so SB3MetaWorldStateEnv "
            "captures robot state each step."
        )
    state_arr      = env._agent_pos_buf[idx].astype(np.float32)
    next_state_arr = env._next_agent_pos_buf[idx].astype(np.float32)

    _save("full_state", full_state_arr)
    _save("next_full_state", next_full_state_arr)
    _save("state", state_arr)
    _save("next_state", next_state_arr)
    _save("action", act_arr)
    _save("reward",    rew_arr,    chunks=(min(100, save_size),))
    _save("done",      done_arr,   chunks=(min(100, save_size),))
    _save("q_value",   q_value_arr, chunks=(min(100, save_size),))
    _save("v_value",   v_value_arr, chunks=(min(100, save_size),))
    _save("advantage", adv_arr,    chunks=(min(100, save_size),))

    # Per-episode success
    episode_success_arr = np.zeros(save_size, dtype=np.float32)
    if env is not None and getattr(env, "_success_flags", None) is not None:
        step_success = env._success_flags[idx].astype(np.float32)
        ep_starts = np.concatenate([[0], episode_ends[:-1]])
        for ep_i, (s, e) in enumerate(zip(ep_starts, episode_ends)):
            ep_succeeded = float(step_success[s:e].max() > 0.5)
            episode_success_arr[s:e] = ep_succeeded
        n_succ = int((episode_success_arr[ep_starts.astype(int)] > 0.5).sum())
        print(f"[ReplayBuffer] Episode success: {n_succ}/{len(episode_ends)} episodes succeeded")
    else:
        print("[ReplayBuffer] No per-step success data; episode_success will be all zeros")
    _save("episode_success", episode_success_arr, chunks=(min(100, save_size),))

    # Save sim qpos/qvel so images can be rendered later by a separate script
    if env is not None and getattr(env, "_store_sim_states", False):
        _save("sim_qpos",      env._sim_qpos[idx].astype(np.float32))
        _save("sim_qvel",      env._sim_qvel[idx].astype(np.float32))
        _save("next_sim_qpos", env._next_sim_qpos[idx].astype(np.float32))
        _save("next_sim_qvel", env._next_sim_qvel[idx].astype(np.float32))
        print(f"[ReplayBuffer] Saved sim_qpos/qvel ({env._sim_qpos.shape[1]}D pos, "
              f"{env._sim_qvel.shape[1]}D vel) for deferred image rendering")

    # episode_ends and buf_idx saved before any rendering so the zarr is valid even
    # if image rendering is interrupted
    zarr_meta.create_dataset(
        "episode_ends", data=episode_ends,
        dtype="int64", overwrite=True, compressor=compressor,
    )
    zarr_meta.create_dataset(
        "buf_idx", data=idx.astype(np.int64),
        dtype="int64", overwrite=True, compressor=compressor,
    )

    n_episodes = len(episode_ends)
    ep_starts = np.concatenate([[0], episode_ends[:-1]])
    ep_lens = episode_ends - ep_starts
    ep_rewards = np.array([rew_arr[s:e].sum() for s, e in zip(ep_starts, episode_ends)])

    print(f"\n[ReplayBuffer -> Zarr] Non-image data saved to {out_path}")
    print(f"  transitions : {save_size}" + (f" (last {save_last_n} of {buf_size})" if offset > 0 else ""))
    print(f"  episodes    : {n_episodes}")
    print(f"  ep lengths  : mean={ep_lens.mean():.1f} min={ep_lens.min()} max={ep_lens.max()}")
    print(f"  ep rewards  : mean={ep_rewards.mean():.1f} std={ep_rewards.std():.1f}")
    print(f"  Q(s,a)      : mean={q_value_arr.mean():.1f} std={q_value_arr.std():.1f}")
    print(f"  V(s)        : mean={v_value_arr.mean():.1f} std={v_value_arr.std():.1f}")
    print(f"  Advantage   : mean={adv_arr.mean():.3f} std={adv_arr.std():.3f} "
          f"positive={100*(adv_arr>0).mean():.1f}%")
    print(f"  Shapes: full_state={full_state_arr.shape} state={state_arr.shape} action={act_arr.shape}")
    if env is not None and getattr(env, "_store_sim_states", False):
        print(f"  sim_qpos/qvel saved — run render_images_to_zarr.py to add img + next_img")


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

    # ── Replay buffer saving ──
    p.add_argument("--save-replay-buffer", action="store_true",
                   help="Save the SAC replay buffer to zarr at checkpoints.")
    p.add_argument("--save-last-n", type=int, default=0,
                   help="Only save the last N transitions to zarr (0 = save all).")
    p.add_argument("--no-images", action="store_true",
                   help="Skip image rendering when saving replay buffer to zarr.")
    p.add_argument("--save-rb-at-success", type=float, default=0.8,
                   help="Save replay buffer (with images) once when success rate first "
                        "reaches this threshold. 0 = save at every best-model checkpoint.")
    p.add_argument("--image-size", type=int, default=128,
                   help="Image size for replay buffer rendering.")
    p.add_argument("--camera", type=str, default="",
                   help="MuJoCo camera name (empty = task default 'corner2').")
    p.add_argument("--device-id", type=int, default=0,
                   help="MuJoCo render device id (GPU index).")

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
    from stable_baselines3.common.callbacks import BaseCallback
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
    def make_env(rank: int, store_sim_states: bool = False, buf_size: int = 0):
        def _init():
            env = SB3MetaWorldStateEnv(
                task_name=args.task,
                seed=args.seed + rank,
                store_sim_states=store_sim_states,
                sim_state_buffer_size=buf_size,
            )
            if args.monitor_dir:
                monitor_root = repo_root / args.monitor_dir / args.task
                os.makedirs(monitor_root, exist_ok=True)
                env = Monitor(env, filename=str(monitor_root / f"monitor_rank{rank}.csv"))
            return env
        return _init

    if args.n_envs <= 1:
        store_states = args.save_replay_buffer and not args.no_images
        # sim_state_buffer_size > 0 activates agent_pos + sim-state buffering
        rb_buf_size = args.buffer_size if args.save_replay_buffer else 0
        vec_env = DummyVecEnv([make_env(0, store_sim_states=store_states, buf_size=rb_buf_size)])
    else:
        # SubprocVecEnv doesn't expose inner env objects; replay buffer saving requires n_envs=1.
        if args.save_replay_buffer:
            raise RuntimeError("--save-replay-buffer requires --n-envs 1")
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
            save_replay_buffer: bool = False,
            task_key: str = "",
            image_size: int = 128,
            camera_name: str = "",
            device_id: int = 0,
            save_last_n: int = 0,
            save_rb_at_success: float = 0.8,
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
            self.save_replay_buffer = save_replay_buffer
            self.task_key = task_key
            self.image_size = image_size
            self.camera_name = camera_name
            self.device_id = device_id
            self.save_last_n = save_last_n
            self.save_rb_at_success = save_rb_at_success
            self._rb_saved = False
            self._saved_85_count = 0
            self._max_85_saves = 5

        def _init_callback(self) -> None:
            os.makedirs(self.save_dir, exist_ok=True)

        def _get_training_env(self):
            """Unwrap VecMonitor -> DummyVecEnv -> [Monitor ->] SB3MetaWorldStateEnv."""
            raw = self.model.env.venv.envs[0]
            if hasattr(raw, "env") and hasattr(raw.env, "_store_sim_states"):
                return raw.env
            return raw

        def _save_rb(self, tag: str, success_rate: float = 0.0) -> None:
            if not self.save_replay_buffer:
                return
            train_env = self._get_training_env()
            has_images = getattr(train_env, "_store_sim_states", False)

            # With images: only save once when success threshold is first reached.
            # Without images: save unconditionally.
            if has_images and self.save_rb_at_success > 0:
                if self._rb_saved or success_rate < self.save_rb_at_success:
                    return
                self._rb_saved = True
                print(f"[ReplayBuffer] Success {success_rate:.2f} >= {self.save_rb_at_success:.2f}, "
                      f"saving replay buffer with images (one-time).")

            rb_path = self.save_dir / f"replay_buffer_{tag}.zarr"
            _save_replay_buffer_to_zarr(
                model=self.model,
                task_key=self.task_key,
                out_path=rb_path,
                env=train_env,
                image_size=self.image_size,
                camera_name=self.camera_name,
                device_id=self.device_id,
                save_last_n=self.save_last_n,
            )

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
                _save_sac_twin_critic(self.model, self.save_dir / "sac_twin_critic_q.pt")
                self._save_rb("best", success_rate)
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

    eval_cb = PeriodicEvalBestCallback(
        eval_env=eval_env,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.eval_episodes,
        save_dir=save_dir,
        best_metric=args.best_metric,
        save_replay_buffer=args.save_replay_buffer,
        task_key=args.task,
        image_size=args.image_size,
        camera_name=args.camera,
        device_id=args.device_id,
        save_last_n=args.save_last_n,
        save_rb_at_success=args.save_rb_at_success,
        verbose=1,
    )
    callback = eval_cb

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
    print()

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
