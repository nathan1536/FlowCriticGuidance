#!/usr/bin/env python3
"""
Roll out a trained SB3 SAC policy on Adroit and save *successful* trajectories to Zarr.

Saves RGB images + sensor state (agent_pos) + full gym state + actions + rewards.
Only keeps episodes where goal_achieved is True at least once.

Supported tasks: door, hammer, pen, relocate

Example:
  python scripts/rollout_sac_adroit_to_zarr.py \
    --task door \
    --model runs/sb3_adroit_sac/models_1/door/best_model.zip \
    --num-success 50 \
    --out ManiFlow/data/adroit_door_sac_rgb.zarr

  # Evaluation only (no zarr save)
  python scripts/rollout_sac_adroit_to_zarr.py \
    --task door \
    --model runs/sb3_adroit_sac/models_1/door/best_model.zip \
    --eval-episodes 20
"""

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import zarr


ADROIT_CAMERAS = {
    "door": "top",
    "hammer": "top",
    "pen": "vil_camera",
    "relocate": "cam1",
}

ADROIT_MAX_STEPS = {
    "door": 200,
    "hammer": 200,
    "pen": 100,
    "relocate": 200,
}

ADROIT_SENSOR_SLICES = {
    "pen": lambda qp: qp[:-6],
    "door": lambda qp: qp[4:-2],
    "hammer": lambda qp: qp[2:-7],
    "relocate": lambda qp: qp[6:-6],
}


def _ensure_sb3_importable(repo_root: Path) -> None:
    try:
        import stable_baselines3  # noqa: F401
        return
    except Exception:
        pass
    sys.path.insert(0, str(repo_root / "third_party" / "dexart-release"))


def _import_from_path(module_name: str, file_path: Path):
    import importlib.util
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


def _set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _get_sensor_state(env, task_key: str) -> np.ndarray:
    """
    Extract the 24-dim hand sensor state (joint positions) used by AdroitEnv
    as agent_pos / observation_sensor.
    """
    env_state = env.env.get_env_state()
    qp = env_state["qpos"]
    return ADROIT_SENSOR_SLICES[task_key](qp).astype(np.float32)


def _render_rgb(env, image_size: int, camera_name: str, device_id: int) -> np.ndarray:
    img = env.env.sim.render(
        width=image_size, height=image_size,
        mode="offscreen", camera_name=camera_name, device_id=device_id,
    )
    if img.ndim == 3 and img.shape[-1] == 3:
        img = img.transpose(2, 0, 1)  # channel-first
    return img


def parse_args():
    p = argparse.ArgumentParser(description="Roll out SAC on Adroit, save successful episodes to zarr")
    p.add_argument("--task", type=str, default="door",
                   help="Adroit task (door, hammer, pen, relocate)")
    p.add_argument("--model", type=str, required=True,
                   help="Path to SAC .zip saved by SB3")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--deterministic", action="store_true",
                   help="Use deterministic actions")
    p.add_argument("--max-steps", type=int, default=0,
                   help="Override max steps per episode (0 = use task default)")
    p.add_argument("--num-success", type=int, default=50,
                   help="How many successful episodes to save")
    p.add_argument("--num-episodes", type=int, default=0,
                   help="Save exactly N episodes regardless of success/failure (0 = use --num-success mode).")
    p.add_argument("--max-attempts", type=int, default=5000,
                   help="Max rollout episodes to attempt (failed ones discarded)")
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--camera", type=str, default="",
                   help="MuJoCo camera name (empty = use task default)")
    p.add_argument("--device-id", type=int, default=0,
                   help="MuJoCo render device id (GPU index)")
    p.add_argument("--mujoco-bin", type=str, default="~/.mujoco/mujoco210/bin")
    p.add_argument("--out", type=str, required=True,
                   help="Output .zarr directory path")

    p.add_argument("--eval-episodes", type=int, default=0,
                   help="If >0, run evaluation only (print success rate). No zarr saved.")
    p.add_argument("--eval-seed-offset", type=int, default=10_000)
    return p.parse_args()


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]

    _ensure_ld_library_path_on_process_start(os.path.expanduser(args.mujoco_bin))

    sys.path.insert(0, str(repo_root / "third_party" / "gym-0.21.0"))
    rrl_deps = repo_root / "third_party" / "rrl-dependencies"
    for subpkg in ("mj_envs", "mjrl"):
        pkg_path = rrl_deps / subpkg
        if pkg_path.exists() and str(pkg_path) not in sys.path:
            sys.path.insert(0, str(pkg_path))

    _ensure_sb3_importable(repo_root)

    import gym  # noqa: E402
    import mj_envs  # noqa: E402, F401
    from stable_baselines3 import SAC

    task_key = args.task.replace("-v0", "")
    if task_key not in ADROIT_CAMERAS:
        raise ValueError(f"Unknown task '{args.task}'. Valid: {list(ADROIT_CAMERAS.keys())}")

    max_steps = args.max_steps if args.max_steps > 0 else ADROIT_MAX_STEPS[task_key]
    camera_name = args.camera if args.camera else ADROIT_CAMERAS[task_key]

    _set_global_seeds(args.seed)

    model = SAC.load(args.model)
    print(f"[Rollout] task={task_key} model={args.model} deterministic={args.deterministic} seed={args.seed}")

    # Extract twin Q-sacs from SAC for V(s) computation
    _sac_device = model.device
    _qf0 = model.critic.qf0
    _qf1 = model.critic.qf1
    _qf0.eval()
    _qf1.eval()

    def _compute_v(state_np: np.ndarray) -> float:
        """V(s) = min(Q0, Q1)(s, pi_SAC(s)) using SAC's deterministic action."""
        with torch.no_grad():
            s = np.asarray(state_np, dtype=np.float32)
            a_det, _ = model.predict(s.reshape(1, -1), deterministic=False)
            s_t = torch.FloatTensor(s).unsqueeze(0).to(_sac_device)
            a_t = torch.FloatTensor(np.asarray(a_det, dtype=np.float32)).to(_sac_device)
            if a_t.dim() == 1:
                a_t = a_t.unsqueeze(0)
            sa = torch.cat([s_t, a_t], dim=-1)
            return torch.min(_qf0(sa), _qf1(sa)).squeeze().cpu().item()

    # ── Optional: evaluation mode ──
    if args.eval_episodes > 0:
        from stable_baselines3.common.vec_env import DummyVecEnv

        env_mod = _import_from_path(
            "sb3_adroit_state_env",
            repo_root / "ManiFlow" / "maniflow" / "env" / "adroit" / "sb3_adroit_state_env.py",
        )
        SB3AdroitStateEnv = env_mod.SB3AdroitStateEnv
        eval_seed = args.seed + args.eval_seed_offset
        eval_env = DummyVecEnv([lambda: SB3AdroitStateEnv(task_name=task_key, seed=eval_seed)])

        eval_rewards = []
        eval_success = []
        for ep in range(args.eval_episodes):
            obs = eval_env.reset()
            done = np.array([False])
            ep_rew = 0.0
            ep_succ = False
            steps = 0
            while not bool(done[0]) and steps < max_steps:
                action, _ = model.predict(np.asarray(obs, dtype=np.float32), deterministic=args.deterministic)
                obs, reward, done, infos = eval_env.step(action)
                ep_rew += float(reward[0])
                info0 = infos[0] if isinstance(infos, (list, tuple)) and len(infos) > 0 else {}
                if isinstance(info0, dict) and bool(info0.get("success", False)):
                    ep_succ = True
                steps += 1
            eval_rewards.append(ep_rew)
            eval_success.append(1.0 if ep_succ else 0.0)
            status = "SUCCESS" if ep_succ else "FAILED"
            print(f"  [Eval {ep:03d}] reward={ep_rew:.1f} steps={steps} {status}")

        try:
            eval_env.close()
        except Exception:
            pass

        mean_reward = float(np.mean(eval_rewards))
        success_rate = float(np.mean(eval_success))
        print(f"\n[Eval] episodes={args.eval_episodes} mean_reward={mean_reward:.3f} "
              f"success_rate={success_rate:.3f}")
        return

    # ── Rollout collection mode ──
    env_id = f"{task_key}-v0"
    env = gym.make(env_id)

    out_path = Path(args.out)
    if out_path.exists():
        raise FileExistsError(f"Output already exists: {out_path} (delete it or choose a new --out)")
    os.makedirs(out_path, exist_ok=True)

    img_arrays = []
    state_arrays = []             # sensor / agent_pos (24-dim qpos slice)
    full_state_arrays = []        # full gym obs used by SAC
    next_full_state_arrays = []   # next-step full_state (for TD learning)
    action_arrays = []
    reward_arrays = []
    done_arrays = []
    v_value_arrays = []           # V(s) = Q(s, pi_SAC(s))
    episode_ends = []

    total_count = 0
    saved_success = 0
    attempted_episodes = 0

    n_attempts = args.num_episodes if args.num_episodes > 0 else args.max_attempts
    for attempt in range(n_attempts):
        attempted_episodes += 1
        full_state = env.reset()
        done = False
        ep_success = False
        ep_success_times = 0

        img_ep = []
        state_ep = []
        full_state_ep = []
        next_full_state_ep = []
        action_ep = []
        reward_ep = []
        done_ep = []
        v_ep = []

        steps = 0
        while not done and steps < max_steps:
            action, _ = model.predict(np.asarray(full_state, dtype=np.float32),
                                      deterministic=args.deterministic)

            img_ep.append(_render_rgb(env, args.image_size, camera_name, args.device_id))
            state_ep.append(_get_sensor_state(env, task_key))
            full_state_ep.append(np.asarray(full_state, dtype=np.float32))
            action_ep.append(np.asarray(action, dtype=np.float32))
            v_ep.append(_compute_v(full_state))

            full_state, reward, done, info = env.step(action)
            next_full_state_ep.append(np.asarray(full_state, dtype=np.float32))
            reward_ep.append(float(reward))
            done_ep.append(1.0 if done else 0.0)
            steps += 1

            if isinstance(info, dict):
                succ = bool(info.get("goal_achieved", False))
                ep_success = ep_success or succ
                ep_success_times += int(succ)

        save_all = args.num_episodes > 0

        if not save_all and (not ep_success or ep_success_times < 1):
            print(f"[Attempt {attempt:04d}] failed steps={steps} "
                  f"success_times={ep_success_times} -> discard")
            continue

        total_count += len(action_ep)
        episode_ends.append(int(total_count))
        img_arrays.extend(img_ep)
        state_arrays.extend(state_ep)
        full_state_arrays.extend(full_state_ep)
        next_full_state_arrays.extend(next_full_state_ep)
        action_arrays.extend(action_ep)
        reward_arrays.extend(reward_ep)
        done_arrays.extend(done_ep)
        v_value_arrays.extend(v_ep)
        saved_success += 1
        ep_reward = sum(reward_ep)
        target = args.num_episodes if save_all else args.num_success
        status = "SUCCESS" if ep_success else "FAILED"
        print(f"[Attempt {attempt:04d}] {status} saved={saved_success}/{target} "
              f"steps={steps} reward={ep_reward:.1f} success_times={ep_success_times}")

        if saved_success >= target:
            break

    try:
        env.close()
    except Exception:
        pass

    if saved_success < args.num_success:
        print(f"[Warn] Only collected {saved_success} successful episodes "
              f"(target {args.num_success}). Saving anyway.")
    attempt_succ_rate = (saved_success / attempted_episodes) if attempted_episodes > 0 else 0.0
    print(f"[Attempt Summary] attempted={attempted_episodes} saved_success={saved_success} "
          f"attempt_success_rate={attempt_succ_rate:.3f}")

    # ── Save to zarr ──
    zarr_root = zarr.group(str(out_path))
    zarr_data = zarr_root.create_group("data")
    zarr_meta = zarr_root.create_group("meta")

    img_arr = np.stack(img_arrays, axis=0)
    if img_arr.shape[1] == 3:  # channel-first -> channel-last
        img_arr = np.transpose(img_arr, (0, 2, 3, 1))
    state_arr = np.stack(state_arrays, axis=0).astype(np.float32)
    full_state_arr = np.stack(full_state_arrays, axis=0).astype(np.float32)
    next_full_state_arr = np.stack(next_full_state_arrays, axis=0).astype(np.float32)
    act_arr = np.stack(action_arrays, axis=0).astype(np.float32)
    reward_arr = np.asarray(reward_arrays, dtype=np.float32)
    done_arr = np.asarray(done_arrays, dtype=np.float32)
    v_value_arr = np.asarray(v_value_arrays, dtype=np.float32)
    episode_ends_arr = np.asarray(episode_ends, dtype=np.int64)

    compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=1)
    zarr_data.create_dataset("img", data=img_arr, chunks=(100, *img_arr.shape[1:]),
                             dtype="uint8", overwrite=True, compressor=compressor)
    zarr_data.create_dataset("state", data=state_arr, chunks=(100, state_arr.shape[1]),
                             dtype="float32", overwrite=True, compressor=compressor)
    zarr_data.create_dataset("full_state", data=full_state_arr, chunks=(100, full_state_arr.shape[1]),
                             dtype="float32", overwrite=True, compressor=compressor)
    zarr_data.create_dataset("next_full_state", data=next_full_state_arr,
                             chunks=(100, next_full_state_arr.shape[1]),
                             dtype="float32", overwrite=True, compressor=compressor)
    zarr_data.create_dataset("action", data=act_arr, chunks=(100, act_arr.shape[1]),
                             dtype="float32", overwrite=True, compressor=compressor)
    zarr_data.create_dataset("reward", data=reward_arr, chunks=(100,),
                             dtype="float32", overwrite=True, compressor=compressor)
    zarr_data.create_dataset("done", data=done_arr, chunks=(100,),
                             dtype="float32", overwrite=True, compressor=compressor)
    zarr_data.create_dataset("v_value", data=v_value_arr, chunks=(100,),
                             dtype="float32", overwrite=True, compressor=compressor)
    zarr_meta.create_dataset("episode_ends", data=episode_ends_arr,
                             dtype="int64", overwrite=True, compressor=compressor)

    ep_starts = [0] + list(episode_ends_arr[:-1])
    ep_rewards = [float(reward_arr[s:e].sum()) for s, e in zip(ep_starts, episode_ends_arr)]
    print(f"\n[Reward Summary] mean={np.mean(ep_rewards):.1f} std={np.std(ep_rewards):.1f} "
          f"min={min(ep_rewards):.1f} max={max(ep_rewards):.1f}")

    q_from_data = np.zeros_like(v_value_arr)
    with torch.no_grad():
        for start in range(0, len(act_arr), 4096):
            end = min(start + 4096, len(act_arr))
            s_t = torch.FloatTensor(full_state_arr[start:end]).to(_sac_device)
            a_t = torch.FloatTensor(act_arr[start:end]).to(_sac_device)
            sa = torch.cat([s_t, a_t], dim=-1)
            q_from_data[start:end] = torch.min(
                _qf0(sa), _qf1(sa)).squeeze(-1).cpu().numpy()
    adv_arr = q_from_data - v_value_arr
    zarr_data.create_dataset("q_value", data=q_from_data, chunks=(100,),
                             dtype="float32", overwrite=True, compressor=compressor)
    zarr_data.create_dataset("advantage", data=adv_arr, chunks=(100,),
                             dtype="float32", overwrite=True, compressor=compressor)

    print(f"\n[V(s) Stats] mean={v_value_arr.mean():.1f} std={v_value_arr.std():.1f}")
    print(f"[Advantage]  mean={adv_arr.mean():.3f} std={adv_arr.std():.3f} "
          f"positive={100*(adv_arr>0).mean():.1f}%")

    print(f"\n[Saved] {out_path}")
    print(f"[Shapes] img={img_arr.shape} state={state_arr.shape} full_state={full_state_arr.shape} "
          f"action={act_arr.shape} reward={reward_arr.shape} v_value={v_value_arr.shape}")
    print(f"[Episodes] success={saved_success} total_steps={len(act_arr)}")


if __name__ == "__main__":
    main()
