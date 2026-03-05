#!/usr/bin/env python3
"""
Roll out a trained SB3 PPO policy on MetaWorld using the visual ManiFlow wrapper (renders RGB/depth/pointcloud),
and save *successful* trajectories to Zarr in the same format as:
  third_party/Metaworld/gen_demonstration_expert.py

This is useful when you want PPO-generated demonstrations with visual observations.

Example:
  conda activate maniflow
  python scripts/rollout_sb3_ppo_metaworld_visual_to_zarr.py \
    --task window-open \
    --model runs/sb3_metaworld_ppo/models/window-open/best_model.zip \
    --num-success 50 \
    --out runs/sb3_metaworld_ppo/rollouts/metaworld_window-open_ppo_visual_success.zarr
"""

import argparse
import importlib.util
import os
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task", type=str, default="window-open")
    p.add_argument("--model", type=str, required=True, help="Path to PPO .zip saved by SB3")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--deterministic", action="store_true", help="Use deterministic actions")
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--num-success", type=int, default=50, help="How many successful episodes to save")
    p.add_argument("--max-attempts", type=int, default=200, help="Max rollout episodes to attempt (failed episodes are discarded)")
    p.add_argument("--device", type=str, default="cuda:0", help="Rendering device string for MetaWorldEnv (e.g., cuda:0)")
    p.add_argument("--num-points", type=int, default=1024)
    p.add_argument("--use-point-crop", action="store_true", default=True)
    p.add_argument("--mujoco-bin", type=str, default="~/.mujoco/mujoco210/bin")
    p.add_argument(
        "--save-critic-out",
        type=str,
        default="",
        help="If set, export critic/value-function weights to this *.pt path (in addition to saving Zarr).",
    )
    p.add_argument("--out", type=str, required=True, help="Output .zarr directory path")
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
    from stable_baselines3 import PPO

    # Load visual ManiFlow wrapper without importing `maniflow.env` (optional deps)
    mw_wrap_mod = _import_from_path(
        "metaworld_wrapper",
        repo_root / "ManiFlow" / "maniflow" / "env" / "metaworld" / "metaworld_wrapper.py",
    )
    MetaWorldEnv = mw_wrap_mod.MetaWorldEnv

    out_path = Path(args.out)
    if out_path.exists():
        raise FileExistsError(f"Output already exists: {out_path} (delete it or choose a new --out)")
    os.makedirs(out_path, exist_ok=True)

    model = PPO.load(args.model)
    if args.save_critic_out:
        critic_path = Path(args.save_critic_out)
        critic_path.parent.mkdir(parents=True, exist_ok=True)
        _save_critic(model, critic_path)
        print(f"[Saved Critic] {critic_path}")

    env = MetaWorldEnv(
        task_name=args.task,
        device=args.device,
        use_point_crop=args.use_point_crop,
        num_points=args.num_points,
    )
    env.episode_length = args.max_steps

    # Accumulators (episode-level buffering to discard failed episodes)
    img_arrays = []
    point_cloud_arrays = []
    depth_arrays = []
    state_arrays = []       # agent_pos (robot state)
    full_state_arrays = []  # MetaWorld state obs used by PPO
    action_arrays = []
    episode_ends = []

    total_count = 0
    saved_success = 0

    for attempt in range(args.max_attempts):
        obs_dict = env.reset()
        full_state = obs_dict["full_state"]
        done = False
        ep_success = False
        ep_success_times = 0

        img_ep = []
        pc_ep = []
        depth_ep = []
        state_ep = []
        full_state_ep = []
        action_ep = []

        steps = 0
        while not done and steps < args.max_steps:
            # PPO was trained on MetaWorld's state observation; we use full_state from the wrapper.
            action, _ = model.predict(np.asarray(full_state, dtype=np.float32), deterministic=args.deterministic)

            # record current obs (like expert generator)
            img_ep.append(obs_dict["image"])
            pc_ep.append(obs_dict["point_cloud"])
            depth_ep.append(obs_dict["depth"])
            state_ep.append(obs_dict["agent_pos"])
            full_state_ep.append(full_state)
            action_ep.append(action)

            obs_dict, reward, done, info = env.step(action)
            full_state = obs_dict["full_state"]
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
        point_cloud_arrays.extend(pc_ep)
        depth_arrays.extend(depth_ep)
        state_arrays.extend(state_ep)
        full_state_arrays.extend(full_state_ep)
        action_arrays.extend(action_ep)
        saved_success += 1
        print(f"[Attempt {attempt:04d}] SUCCESS saved={saved_success}/{args.num_success} steps={steps} success_times={ep_success_times}")

        if saved_success >= args.num_success:
            break

    env.close()

    if saved_success < args.num_success:
        print(f"[Warn] Only collected {saved_success} successful episodes (target {args.num_success}). Saving anyway.")

    # Save to zarr (match expert keys)
    zarr_root = zarr.group(str(out_path))
    zarr_data = zarr_root.create_group("data")
    zarr_meta = zarr_root.create_group("meta")

    img_arr = np.stack(img_arrays, axis=0)
    if img_arr.shape[1] == 3:  # make channel-last like expert script
        img_arr = np.transpose(img_arr, (0, 2, 3, 1))
    state_arr = np.stack(state_arrays, axis=0).astype(np.float32)
    full_state_arr = np.stack(full_state_arrays, axis=0).astype(np.float32)
    pc_arr = np.stack(point_cloud_arrays, axis=0).astype(np.float32)
    depth_arr = np.stack(depth_arrays, axis=0).astype(np.float32)
    act_arr = np.stack(action_arrays, axis=0).astype(np.float32)
    episode_ends_arr = np.asarray(episode_ends, dtype=np.int64)

    compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=1)
    zarr_data.create_dataset("img", data=img_arr, chunks=(100, *img_arr.shape[1:]), dtype="uint8", overwrite=True, compressor=compressor)
    zarr_data.create_dataset("state", data=state_arr, chunks=(100, state_arr.shape[1]), dtype="float32", overwrite=True, compressor=compressor)
    zarr_data.create_dataset("full_state", data=full_state_arr, chunks=(100, full_state_arr.shape[1]), dtype="float32", overwrite=True, compressor=compressor)
    zarr_data.create_dataset("point_cloud", data=pc_arr, chunks=(100, pc_arr.shape[1], pc_arr.shape[2]), dtype="float32", overwrite=True, compressor=compressor)
    zarr_data.create_dataset("depth", data=depth_arr, chunks=(100, depth_arr.shape[1], depth_arr.shape[2]), dtype="float32", overwrite=True, compressor=compressor)
    zarr_data.create_dataset("action", data=act_arr, chunks=(100, act_arr.shape[1]), dtype="float32", overwrite=True, compressor=compressor)
    zarr_meta.create_dataset("episode_ends", data=episode_ends_arr, dtype="int64", overwrite=True, compressor=compressor)

    print(f"[Saved] {out_path}")
    print(f"[Episodes] success={saved_success} steps={len(act_arr)}")


if __name__ == "__main__":
    main()


