#!/usr/bin/env python3
"""
Roll out a trained SB3 PPO policy on a single MetaWorld task and save trajectories to Zarr.

This is the PPO analogue of `third_party/Metaworld/gen_demonstration_expert.py`, but it logs
what the PPO policy sees (state obs), plus actions/rewards/dones/episode_ends.

Example:
  python scripts/rollout_sb3_ppo_metaworld_to_zarr.py \
    --task assembly \
    --model runs/sb3_metaworld_ppo/models/assembly/ppo_model.zip \
    --num-episodes 50 \
    --out runs/sb3_metaworld_ppo/rollouts/metaworld_assembly_ppo.zarr
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
    p.add_argument("--task", type=str, default="assembly")
    p.add_argument("--model", type=str, required=True, help="Path to PPO .zip saved by SB3")
    p.add_argument("--num-episodes", type=int, default=50)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--deterministic", action="store_true", help="Use deterministic policy during rollout")
    p.add_argument("--mujoco-bin", type=str, default="~/.mujoco/mujoco210/bin", help="MuJoCo bin dir (for mujoco-py LD_LIBRARY_PATH)")
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

    # mujoco-py requires MuJoCo's bin directory on LD_LIBRARY_PATH.
    mujoco_bin = os.path.expanduser(args.mujoco_bin)
    _ensure_ld_library_path_on_process_start(mujoco_bin)

    # Ensure MetaWorld imports the correct OpenAI gym (vendored) instead of any gymnasium shim.
    sys.path.insert(0, str(repo_root / "third_party" / "gym-0.21.0"))
    sys.path.insert(0, str(repo_root / "third_party" / "Metaworld"))

    _ensure_sb3_importable(repo_root)
    from stable_baselines3 import PPO

    env_mod = _import_from_path(
        "sb3_metaworld_state_env",
        repo_root / "ManiFlow" / "maniflow" / "env" / "metaworld" / "sb3_metaworld_state_env.py",
    )
    SB3MetaWorldStateEnv = env_mod.SB3MetaWorldStateEnv

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

    env = SB3MetaWorldStateEnv(task_name=args.task, episode_length=args.max_steps, seed=args.seed)

    obs_list = []
    action_list = []
    reward_list = []
    done_list = []
    success_list = []
    episode_ends = []

    total_steps = 0
    for ep in range(args.num_episodes):
        obs = env.reset()
        done = False
        ep_steps = 0
        ep_return = 0.0
        ep_success = False

        while not done:
            action, _ = model.predict(obs, deterministic=args.deterministic)
            next_obs, reward, done, info = env.step(action)

            obs_list.append(np.asarray(obs, dtype=np.float32))
            action_list.append(np.asarray(action, dtype=np.float32))
            reward_list.append(np.float32(reward))
            done_list.append(np.bool_(done))

            success = bool(info.get("success", False)) if isinstance(info, dict) else False
            success_list.append(np.bool_(success))
            ep_success = ep_success or success

            obs = next_obs
            ep_steps += 1
            total_steps += 1
            ep_return += float(reward)

            if ep_steps >= args.max_steps:
                # Safety cap
                done = True

        episode_ends.append(total_steps)
        print(f"[Episode {ep:04d}] steps={ep_steps} return={ep_return:.3f} success={int(ep_success)}")

    env.close()

    # Save to zarr in a similar layout to your expert generator:
    # - data/state, data/action, data/reward, data/done, data/success
    # - meta/episode_ends
    zarr_root = zarr.group(str(out_path))
    zarr_data = zarr_root.create_group("data")
    zarr_meta = zarr_root.create_group("meta")

    obs_arr = np.stack(obs_list, axis=0)
    act_arr = np.stack(action_list, axis=0)
    rew_arr = np.asarray(reward_list, dtype=np.float32)
    done_arr = np.asarray(done_list, dtype=np.bool_)
    success_arr = np.asarray(success_list, dtype=np.bool_)
    episode_ends_arr = np.asarray(episode_ends, dtype=np.int64)

    compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=1)
    zarr_data.create_dataset("state", data=obs_arr, chunks=(100, obs_arr.shape[1]), dtype="float32", overwrite=True, compressor=compressor)
    zarr_data.create_dataset("action", data=act_arr, chunks=(100, act_arr.shape[1]), dtype="float32", overwrite=True, compressor=compressor)
    zarr_data.create_dataset("reward", data=rew_arr, chunks=(100,), dtype="float32", overwrite=True, compressor=compressor)
    zarr_data.create_dataset("done", data=done_arr, chunks=(100,), dtype="bool", overwrite=True, compressor=compressor)
    zarr_data.create_dataset("success", data=success_arr, chunks=(100,), dtype="bool", overwrite=True, compressor=compressor)
    zarr_meta.create_dataset("episode_ends", data=episode_ends_arr, dtype="int64", overwrite=True, compressor=compressor)

    print(f"[Saved] {out_path}")
    print(f"[Shapes] state={obs_arr.shape} action={act_arr.shape} reward={rew_arr.shape} episode_ends={episode_ends_arr.shape}")


if __name__ == "__main__":
    main()


