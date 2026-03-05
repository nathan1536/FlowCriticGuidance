#!/usr/bin/env python3
"""
Roll out a trained SB3 PPO policy on MetaWorld and save *successful* RGB-only trajectories to Zarr.

This is similar to `scripts/rollout_sb3_ppo_metaworld_visual_to_zarr.py` but:
- does NOT compute point clouds
- does NOT compute depth
- only records 2D RGB images + (robot state, full_state, action)

Example:
  python scripts/rollout_sb3_ppo_metaworld_rgb_to_zarr.py \
    --task window-open \
    --model runs/sb3_metaworld_ppo/models/window-open/best_model.zip \
    --num-success 50 \
    --out runs/sb3_metaworld_ppo/rollouts/metaworld_window-open_ppo_rgb_success.zarr
"""

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import zarr


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
    """
    Native libs (mujoco-py deps) are resolved using LD_LIBRARY_PATH as of process start.
    Mutating os.environ after Python starts is often too late; re-exec once if needed.
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


def _set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _save_critic(model, path: Path) -> None:
    """
    Save critic/value-function related weights for later inspection.
    Note: value prediction still depends on the feature extractor and mlp_extractor.
    """
    import torch

    policy = model.policy
    payload = {
        "policy_class": policy.__class__.__name__,
        "device": str(getattr(model, "device", "")),
        "state_dict": {
            "features_extractor": getattr(policy, "features_extractor", None).state_dict()
            if getattr(policy, "features_extractor", None) is not None
            else {},
            "mlp_extractor": getattr(policy, "mlp_extractor", None).state_dict()
            if getattr(policy, "mlp_extractor", None) is not None
            else {},
            "value_net": getattr(policy, "value_net", None).state_dict()
            if getattr(policy, "value_net", None) is not None
            else {},
        },
    }
    torch.save(payload, str(path))


def _get_robot_state(env) -> np.ndarray:
    """
    Match `MetaWorldEnv.get_robot_state()` (eef + right/left fingertip site positions).
    """
    eef_pos = env.get_endeff_pos()
    finger_right = env._get_site_pos("rightEndEffector")
    finger_left = env._get_site_pos("leftEndEffector")
    return np.concatenate([eef_pos, finger_right, finger_left]).astype(np.float32)


def _render_rgb(env, image_size: int, camera_name: str, device_id: int) -> np.ndarray:
    img = env.sim.render(width=image_size, height=image_size, camera_name=camera_name, device_id=device_id)
    # Make channel-first like the visual script; we'll convert to channel-last on save.
    if img.ndim == 3 and img.shape[-1] == 3:
        img = img.transpose(2, 0, 1)
    return img


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task", type=str, default="reach")
    p.add_argument("--model", type=str, default="runs/sb3_metaworld_sac/models/pick-place/diverse_ent0.35018_t6200000.zip", help="Path to PPO .zip saved by SB3")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--deterministic", action="store_true", help="Use deterministic actions")
    # In parse_args(), add:
    p.add_argument("--algo", type=str, default="sac", choices=["ppo", "sac"],
                help="Which algorithm was used to train the model (ppo or sac).")    
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument(
        "--eval-episodes",
        type=int,
        default=0,
        help="If >0, run an evaluation (like training) for N episodes and print success_rate/mean_reward. Does not save Zarr.",
    )
    p.add_argument(
        "--eval-seed-offset",
        type=int,
        default=10_000,
        help="Match training eval seeding: eval uses seed + eval_seed_offset.",
    )
    p.add_argument(
        "--eval-debug-success-steps",
        type=int,
        default=0,
        help="If >0, print first K step-level values of info['success'] during eval (episode 0).",
    )
    p.add_argument("--num-success", type=int, default=20, help="How many successful episodes to save")
    p.add_argument(
        "--max-attempts",
        type=int,
        default=1000,
        help="Max rollout episodes to attempt (failed episodes are discarded)",
    )
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--camera", type=str, default="corner2", help="MuJoCo camera name (e.g., corner2, corner3, topview, ...)")
    p.add_argument("--device-id", type=int, default=0, help="MuJoCo render device id (GPU index)")
    p.add_argument("--mujoco-bin", type=str, default="~/.mujoco/mujoco210/bin")
    p.add_argument(
        "--save-critic-out",
        type=str,
        default="",
        help="If set, export critic/value-function weights to this *.pt path (in addition to saving Zarr).",
    )
    p.add_argument("--out", type=str, default="/home/nathan/Project/ManiFlow_Policy/dataset/reach/ppo_rgb_success.zarr", help="Output .zarr directory path")
    return p.parse_args()


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]

    # Ensure mujoco libs are visible before importing mujoco_py/MetaWorld.
    _ensure_ld_library_path_on_process_start(os.path.expanduser(args.mujoco_bin))

    # Ensure MetaWorld uses vendored OpenAI gym (not gymnasium shim)
    sys.path.insert(0, str(repo_root / "third_party" / "gym-0.21.0"))
    sys.path.insert(0, str(repo_root / "third_party" / "Metaworld"))

    _ensure_sb3_importable(repo_root)


    # In main(), replace the PPO import/load block with:
    if args.algo == "sac":
        from stable_baselines3 import SAC
        model = SAC.load(args.model)
    else:
        from stable_baselines3 import PPO
        model = PPO.load(args.model)
    import metaworld

    _set_global_seeds(int(args.seed))

    if args.save_critic_out:
        critic_path = Path(args.save_critic_out)
        critic_path.parent.mkdir(parents=True, exist_ok=True)
        _save_critic(model, critic_path)
        print(f"[Saved Critic] {critic_path}")

    task_name = args.task
    if "-v2" not in task_name:
        task_name = task_name + "-v2-goal-observable"

    env = metaworld.envs.ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[task_name]()
    if hasattr(env, "_freeze_rand_vec"):
        env._freeze_rand_vec = False

    # Match the ManiFlow visual wrapper camera defaults (best-effort; safe if fields differ).
    try:
        env.sim.model.cam_pos[2] = [0.6, 0.295, 0.8]
        env.sim.model.vis.map.znear = 0.1
        env.sim.model.vis.map.zfar = 1.5
    except Exception:
        pass

    try:
        env.seed(int(args.seed))
    except Exception:
        pass

    print(f"[Rollout] task={args.task} model={args.model} deterministic={bool(args.deterministic)} seed={int(args.seed)}")

    # Optional: SB3-style evaluation (apples-to-apples with training eval).
    if int(args.eval_episodes) > 0:
        # IMPORTANT: Training eval uses SB3MetaWorldStateEnv (not the raw metaworld env),
        # seeded with (seed + 10_000). Use the same here.
        from stable_baselines3.common.vec_env import DummyVecEnv

        env_mod = _import_from_path(
            "sb3_metaworld_state_env",
            repo_root / "ManiFlow" / "maniflow" / "env" / "metaworld" / "sb3_metaworld_state_env.py",
        )
        SB3MetaWorldStateEnv = env_mod.SB3MetaWorldStateEnv
        eval_seed = int(args.seed) + int(args.eval_seed_offset)
        eval_env = DummyVecEnv([lambda: SB3MetaWorldStateEnv(task_name=args.task, episode_length=int(args.max_steps), seed=eval_seed)])

        eval_rewards = []
        eval_success = []
        success_key_missing_steps = 0
        success_key_total_steps = 0
        success_true_steps = 0
        success_present_steps = 0
        for ep in range(int(args.eval_episodes)):
            obs = eval_env.reset()
            done = np.array([False])
            ep_rew = 0.0
            ep_succ = False
            steps = 0
            while not bool(done[0]) and steps < int(args.max_steps):
                # Match training eval: deterministic actions.
                action, _ = model.predict(np.asarray(obs, dtype=np.float32), deterministic=False)
                action = action + np.random.normal(0, 0.05, size=action.shape)
                action = np.clip(action, -1.0, 1.0)
                obs, reward, done, infos = eval_env.step(action)
                ep_rew += float(reward[0])
                info0 = infos[0] if isinstance(infos, (list, tuple)) and len(infos) > 0 else {}
                if isinstance(info0, dict):
                    success_key_total_steps += 1
                    if "success" not in info0:
                        success_key_missing_steps += 1
                    else:
                        success_present_steps += 1
                        succ_val = bool(info0.get("success", False))
                        if succ_val:
                            success_true_steps += 1
                            ep_succ = True
                        if ep == 0 and int(args.eval_debug_success_steps) > 0 and steps < int(args.eval_debug_success_steps):
                            print(f"[Eval Debug] step={steps} info['success']={info0.get('success', None)} -> {succ_val}")
                steps += 1
            eval_rewards.append(ep_rew)
            eval_success.append(1.0 if ep_succ else 0.0)
        try:
            eval_env.close()
        except Exception:
            pass
        mean_reward = float(np.mean(eval_rewards)) if len(eval_rewards) else 0.0
        success_rate = float(np.mean(eval_success)) if len(eval_success) else 0.0
        if success_key_total_steps > 0 and success_key_missing_steps == success_key_total_steps:
            print("[Eval Warn] `info['success']` missing for all steps (success_rate will be 0).")
        print(
            f"[Eval] episodes={int(args.eval_episodes)} mean_reward={mean_reward:.3f} "
            f"success_rate={success_rate:.3f} (deterministic=True, eval_seed={eval_seed})"
        )
        print(
            f"[Eval Success Steps] present_steps={success_present_steps} true_steps={success_true_steps} "
            f"missing_steps={success_key_missing_steps}/{success_key_total_steps}"
        )
        return

    out_path = Path(args.out)
    if out_path.exists():
        raise FileExistsError(f"Output already exists: {out_path} (delete it or choose a new --out)")
    os.makedirs(out_path, exist_ok=True)

    # Accumulators (episode-level buffering to discard failed episodes)
    img_arrays = []
    state_arrays = []            # robot state (agent_pos)
    full_state_arrays = []       # MetaWorld state obs used by PPO
    next_full_state_arrays = []  # next-step full_state (for IQL / TD learning)
    action_arrays = []
    reward_arrays = []           # per-step reward
    done_arrays = []             # per-step done flag (1.0 at episode end)
    episode_ends = []

    total_count = 0
    saved_success = 0
    attempted_episodes = 0

    for attempt in range(int(args.max_attempts)):
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

        steps = 0
        while not done and steps < int(args.max_steps):
            action, _ = model.predict(np.asarray(full_state, dtype=np.float32), deterministic=False)

            # record current obs (like expert generator style)
            img_ep.append(_render_rgb(env, int(args.image_size), str(args.camera), int(args.device_id)))
            state_ep.append(_get_robot_state(env))
            full_state_ep.append(np.asarray(full_state, dtype=np.float32))
            action_ep.append(np.asarray(action, dtype=np.float32))

            full_state, reward, done, info = env.step(action)
            next_full_state_ep.append(np.asarray(full_state, dtype=np.float32))
            reward_ep.append(float(reward))
            done_ep.append(1.0 if done else 0.0)
            steps += 1

            if isinstance(info, dict):
                succ = bool(info.get("success", False))
                ep_success = ep_success or succ
                ep_success_times += int(succ)

        if not ep_success or ep_success_times < 1:
            print(f"[Attempt {attempt:04d}] failed steps={steps} success_times={ep_success_times} -> discard")
            continue

        # commit episode
        total_count += len(action_ep)
        episode_ends.append(int(total_count))
        img_arrays.extend(img_ep)
        state_arrays.extend(state_ep)
        full_state_arrays.extend(full_state_ep)
        next_full_state_arrays.extend(next_full_state_ep)
        action_arrays.extend(action_ep)
        reward_arrays.extend(reward_ep)
        done_arrays.extend(done_ep)
        saved_success += 1
        ep_reward = sum(reward_ep)
        print(f"[Attempt {attempt:04d}] SUCCESS saved={saved_success}/{args.num_success} steps={steps} reward={ep_reward:.1f} success_times={ep_success_times}")

        if saved_success >= int(args.num_success):
            break

    try:
        env.close()
    except Exception:
        pass

    if saved_success < int(args.num_success):
        print(f"[Warn] Only collected {saved_success} successful episodes (target {args.num_success}). Saving anyway.")
    attempt_succ_rate = (saved_success / attempted_episodes) if attempted_episodes > 0 else 0.0
    print(f"[Attempt Summary] attempted={attempted_episodes} saved_success={saved_success} attempt_success_rate={attempt_succ_rate:.3f}")


    zarr_root = zarr.group(str(out_path))
    zarr_data = zarr_root.create_group("data")
    zarr_meta = zarr_root.create_group("meta")

    img_arr = np.stack(img_arrays, axis=0)
    if img_arr.shape[1] == 3:  # make channel-last like expert script
        img_arr = np.transpose(img_arr, (0, 2, 3, 1))
    state_arr = np.stack(state_arrays, axis=0).astype(np.float32)
    full_state_arr = np.stack(full_state_arrays, axis=0).astype(np.float32)
    next_full_state_arr = np.stack(next_full_state_arrays, axis=0).astype(np.float32)
    act_arr = np.stack(action_arrays, axis=0).astype(np.float32)
    reward_arr = np.asarray(reward_arrays, dtype=np.float32)
    done_arr = np.asarray(done_arrays, dtype=np.float32)
    episode_ends_arr = np.asarray(episode_ends, dtype=np.int64)

    compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=1)
    zarr_data.create_dataset("img", data=img_arr, chunks=(100, *img_arr.shape[1:]), dtype="uint8", overwrite=True, compressor=compressor)
    zarr_data.create_dataset("state", data=state_arr, chunks=(100, state_arr.shape[1]), dtype="float32", overwrite=True, compressor=compressor)
    zarr_data.create_dataset("full_state", data=full_state_arr, chunks=(100, full_state_arr.shape[1]), dtype="float32", overwrite=True, compressor=compressor)
    zarr_data.create_dataset("next_full_state", data=next_full_state_arr, chunks=(100, next_full_state_arr.shape[1]), dtype="float32", overwrite=True, compressor=compressor)
    zarr_data.create_dataset("action", data=act_arr, chunks=(100, act_arr.shape[1]), dtype="float32", overwrite=True, compressor=compressor)
    zarr_data.create_dataset("reward", data=reward_arr, chunks=(100,), dtype="float32", overwrite=True, compressor=compressor)
    zarr_data.create_dataset("done", data=done_arr, chunks=(100,), dtype="float32", overwrite=True, compressor=compressor)
    zarr_meta.create_dataset("episode_ends", data=episode_ends_arr, dtype="int64", overwrite=True, compressor=compressor)

    # Per-episode reward summary
    ep_starts = [0] + list(episode_ends_arr[:-1])
    ep_rewards = [float(reward_arr[s:e].sum()) for s, e in zip(ep_starts, episode_ends_arr)]
    print(f"\n[Reward Summary] mean={np.mean(ep_rewards):.1f} std={np.std(ep_rewards):.1f} "
          f"min={min(ep_rewards):.1f} max={max(ep_rewards):.1f}")

    print(f"[Saved] {out_path}")
    print(f"[Episodes] success={saved_success} steps={len(act_arr)}")


if __name__ == "__main__":
    main()


