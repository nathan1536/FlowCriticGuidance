#!/usr/bin/env python3
"""
Visualize Q-critic landscape: sweep each action dimension around the expert action
and plot Q(s, a) to check if the critic has a meaningful gradient signal.

If the landscape is flat (Q barely changes), ACGD has nothing to follow.

Usage:
  cd /home/nathan4074/Project/ManiFlow_Policy/ManiFlow
  python ../scripts/debug_critic_landscape.py

Output:
  - PNG plots saved to scripts/ directory (one per task)
  - Console summary of Q-value ranges per task per action dim
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import zarr
import matplotlib

matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt

# ─── Config ───────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "ManiFlow" / "data"
CRITIC_DIR = REPO_ROOT / "runs" / "sb3_metaworld_ppo" / "models_v2"
OUTPUT_DIR = REPO_ROOT / "scripts" / "critic_debug_output"

STATE_DIM = 39
ACTION_DIM = 4
HIDDEN_DIM = 256

N_POINTS = 200          # sweep resolution per action dimension
PERTURB_RANGE = 0.5     # sweep action_dim ± this amount around expert action
N_RANDOM_STATES = 5     # test multiple states per task for robustness

# Tasks that have BOTH zarr data and a Q-critic
TASK_CANDIDATES = [
    "reach",
    "reach-wall",
    "coffee-pull",
    "bin-picking",
    "window-open",
]
# ──────────────────────────────────────────────────────────────────────


def build_critic(state_dim: int, action_dim: int, hidden_dim: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(state_dim + action_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, 1),
    )


def load_zarr_data(task_name: str):
    """Load full_state and action arrays from the expert zarr dataset."""
    zarr_path = DATA_DIR / f"metaworld_{task_name}_expert.zarr"
    if not zarr_path.exists():
        return None, None, None
    root = zarr.open(str(zarr_path), mode="r")
    full_state = np.array(root["data"]["full_state"])   # (N, 39)
    action = np.array(root["data"]["action"])            # (N, 4)
    episode_ends = np.array(root["meta"]["episode_ends"])  # (E,)
    return full_state, action, episode_ends


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Output dir: {OUTPUT_DIR}")
    print("=" * 70)

    summary = {}

    for task_name in TASK_CANDIDATES:
        # Check critic exists
        critic_path = CRITIC_DIR / task_name / "best_expert_critic_q.pt"
        if not critic_path.exists():
            print(f"[SKIP] No critic for {task_name}")
            continue

        # Load data
        full_state, action, episode_ends = load_zarr_data(task_name)
        if full_state is None:
            print(f"[SKIP] No zarr data for {task_name}")
            continue

        # Load critic
        critic = build_critic(STATE_DIM, ACTION_DIM, HIDDEN_DIM)
        critic.load_state_dict(torch.load(str(critic_path), map_location=device))
        critic.eval().to(device)

        print(f"\n{'='*70}")
        print(f"TASK: {task_name}")
        print(f"  Data shape: state={full_state.shape}, action={action.shape}")
        print(f"  Critic: {critic_path}")

        # Sample random states from the dataset
        n_total = full_state.shape[0]
        rng = np.random.default_rng(42)
        sample_indices = rng.choice(n_total, size=min(N_RANDOM_STATES, n_total), replace=False)

        task_q_ranges = []

        fig, axes = plt.subplots(
            N_RANDOM_STATES, ACTION_DIM,
            figsize=(5 * ACTION_DIM, 4 * N_RANDOM_STATES),
            squeeze=False,
        )

        for si, idx in enumerate(sample_indices):
            state = torch.tensor(full_state[idx], dtype=torch.float32, device=device)
            expert_action = torch.tensor(action[idx], dtype=torch.float32, device=device)

            # Baseline Q
            with torch.no_grad():
                sa = torch.cat([state, expert_action]).unsqueeze(0)
                q_baseline = critic(sa).item()

            dim_ranges = []
            for dim in range(ACTION_DIM):
                center = expert_action[dim].item()
                sweep = torch.linspace(
                    center - PERTURB_RANGE,
                    center + PERTURB_RANGE,
                    N_POINTS,
                    device=device,
                )
                q_vals = []
                for val in sweep:
                    a = expert_action.clone()
                    a[dim] = val
                    with torch.no_grad():
                        sa = torch.cat([state, a]).unsqueeze(0)
                        q_vals.append(critic(sa).item())

                q_vals = np.array(q_vals)
                q_range = float(q_vals.max() - q_vals.min())
                dim_ranges.append(q_range)

                # Also compute gradient magnitude at expert action
                a_grad = expert_action.clone().detach().requires_grad_(True)
                sa_grad = torch.cat([state.detach(), a_grad]).unsqueeze(0)
                q_grad = critic(sa_grad)
                q_grad.backward()
                grad_at_expert = a_grad.grad[dim].item() if a_grad.grad is not None else 0.0

                # Plot
                ax = axes[si][dim]
                ax.plot(sweep.cpu().numpy(), q_vals, linewidth=1.5)
                ax.axvline(center, color="r", linestyle="--", alpha=0.7, label=f"expert={center:.3f}")
                ax.set_title(
                    f"sample {si} | dim {dim}\n"
                    f"Q range: {q_range:.4f} | ∇Q: {grad_at_expert:.4f}",
                    fontsize=9,
                )
                ax.set_xlabel(f"action[{dim}]")
                ax.set_ylabel("Q(s, a)")
                ax.legend(fontsize=7)

            task_q_ranges.append(dim_ranges)
            print(
                f"  Sample {si} (idx={idx}): Q_baseline={q_baseline:.4f} | "
                f"Q_range per dim: {['%.4f' % r for r in dim_ranges]}"
            )

        plt.suptitle(
            f"{task_name} — Q-Critic Landscape\n"
            f"(sweep ±{PERTURB_RANGE} around expert action, {N_POINTS} points)",
            fontsize=12,
        )
        plt.tight_layout()
        out_path = OUTPUT_DIR / f"critic_landscape_{task_name}.png"
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"  → Saved: {out_path}")

        # Summary stats
        ranges_arr = np.array(task_q_ranges)  # (N_RANDOM_STATES, ACTION_DIM)
        mean_range_per_dim = ranges_arr.mean(axis=0)
        max_range_per_dim = ranges_arr.max(axis=0)
        summary[task_name] = {
            "mean_range": mean_range_per_dim,
            "max_range": max_range_per_dim,
        }

    # ─── Print summary ───────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY: Q-value range across ±{:.1f} action perturbation".format(PERTURB_RANGE))
    print("=" * 70)
    print(f"{'Task':<20} {'Dim 0':>10} {'Dim 1':>10} {'Dim 2':>10} {'Dim 3':>10} {'Diagnosis':>15}")
    print("-" * 85)

    for task_name, stats in summary.items():
        mr = stats["mean_range"]
        max_r = max(mr)
        if max_r < 1.0:
            diagnosis = "FLAT ❌"
        elif max_r < 10.0:
            diagnosis = "MARGINAL ⚠️"
        else:
            diagnosis = "HEALTHY ✅"

        print(
            f"{task_name:<20} "
            f"{mr[0]:>10.4f} {mr[1]:>10.4f} {mr[2]:>10.4f} {mr[3]:>10.4f} "
            f"{diagnosis:>15}"
        )

    print("\nInterpretation:")
    print("  FLAT (< 1.0):     ACGD is useless — critic gradient ≈ 0")
    print("  MARGINAL (1-10):  ACGD might work with very large λ")
    print("  HEALTHY (> 10):   Critic has a meaningful landscape for ACGD")
    print(f"\nAll plots saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
