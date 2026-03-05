#!/usr/bin/env python3
"""
Check action min/max in Adroit zarr datasets and compare to environment bounds.
"""
import sys
import numpy as np

# Add paths for gym and mj_envs
sys.path.insert(0, 'third_party/gym-0.21.0')
sys.path.insert(0, 'third_party/rrl-dependencies/mj_envs')
sys.path.insert(0, 'third_party/rrl-dependencies/mjrl')

import zarr

# List your zarr paths here (adjust as needed)
zarr_paths = [
    "ManiFlow/data/adroit_door_expert.zarr",
    # Add more paths as needed
]

print("=" * 70)
print("CHECKING ZARR DATASET ACTION RANGES")
print("=" * 70)

for path in zarr_paths:
    try:
        root = zarr.open(path, mode='r')
        actions = np.array(root['data']['action'])
        
        print(f"\n{path}")
        print(f"  Shape: {actions.shape}")
        print(f"  Action dim: {actions.shape[-1]}")
        print(f"  Global min: {actions.min():.4f}")
        print(f"  Global max: {actions.max():.4f}")
        print(f"  Per-dim min: {actions.min(axis=0)}")
        print(f"  Per-dim max: {actions.max(axis=0)}")
        
        # Check how much of [-1, 1] is covered
        coverage = (actions.max() - actions.min()) / 2.0  # 2.0 is full range
        print(f"  Range coverage: {coverage*100:.1f}% of [-1, 1]")
        
        # Check if any actions are outside [-1, 1]
        outside = np.sum((actions < -1) | (actions > 1))
        if outside > 0:
            print(f"  WARNING: {outside} values outside [-1, 1]!")
        else:
            print(f"  All values within [-1, 1]: OK")
            
    except FileNotFoundError:
        print(f"\n{path}: NOT FOUND")
    except Exception as e:
        print(f"\n{path}: ERROR - {e}")

print("\n" + "=" * 70)
print("ENVIRONMENT ACTION SPACE BOUNDS (expected)")
print("=" * 70)

try:
    import gym
    import mj_envs
    
    for task in ['door', 'hammer', 'pen', 'relocate']:
        env = gym.make(f'{task}-v0')
        print(f"\n{task}-v0:")
        print(f"  Action dim: {env.action_space.shape[0]}")
        print(f"  Low:  {env.action_space.low.min():.1f} to {env.action_space.low.max():.1f}")
        print(f"  High: {env.action_space.high.min():.1f} to {env.action_space.high.max():.1f}")
        env.close()
except Exception as e:
    print(f"Could not check env bounds: {e}")

print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)
print("""
If dataset actions are well within [-1, 1] (e.g., [-0.7, 0.8]):
  -> The 'limits' normalizer will map this subset to [-1, 1]
  -> This could limit the student's action range
  -> Consider using identity normalizer for actions

If dataset actions span nearly [-1, 1] (e.g., [-0.99, 0.99]):
  -> Normalization should be fine
""")