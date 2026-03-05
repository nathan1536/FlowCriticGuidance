#!/usr/bin/env python3
"""
Critic-Only Gradient Ascent Test for Adroit:
Start from random noise (or expert action + noise) and use gradient ascent
on Q(s, a) to see if the critic can guide actions toward high-value regions.

If gradient ascent can't even increase Q meaningfully, the critic is the problem.

Usage:
  python scripts/debug_critic_gradient_ascent_adroit.py
  python scripts/debug_critic_gradient_ascent_adroit.py --task door
  python scripts/debug_critic_gradient_ascent_adroit.py --task hammer --state-dim 46

Output:
  - Console log showing Q trajectory during optimization
  - PNG plots saved to scripts/critic_debug_output/ (one per task)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import zarr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─── Config ───────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[1]

ADROIT_STATE_DIMS = {
    "door": 39,
    "hammer": 46,
    "pen": 45,
    "relocate": 39,
}
ACTION_DIM = 28
HIDDEN_DIM = 256
# ──────────────────────────────────────────────────────────────────────


def build_critic(state_dim: int, action_dim: int, hidden_dim: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(state_dim + action_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, 1),
    )


class TwinQCritic(nn.Module):
    """Conservative twin-Q critic: returns min(Q1(s,a), Q2(s,a))."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.q0 = build_critic(state_dim, action_dim, hidden_dim)
        self.q1 = build_critic(state_dim, action_dim, hidden_dim)

    def forward(self, sa: torch.Tensor) -> torch.Tensor:
        return torch.min(self.q0(sa), self.q1(sa))


def load_critic(critic_dir: Path, task_name: str, state_dim: int, device):
    """Load twin-Q critic if available, otherwise fall back to single critic."""
    twin_path = critic_dir / task_name / "sac_twin_critic_q.pt"
    single_path = critic_dir / task_name / "best_expert_critic_q.pt"

    if twin_path.exists():
        critic = TwinQCritic(state_dim, ACTION_DIM, HIDDEN_DIM)
        critic.load_state_dict(torch.load(str(twin_path), map_location=device))
        print(f"  Loaded twin Q-critic from {twin_path}")
    elif single_path.exists():
        critic = build_critic(state_dim, ACTION_DIM, HIDDEN_DIM)
        critic.load_state_dict(torch.load(str(single_path), map_location=device))
        print(f"  Loaded single Q-critic from {single_path}")
    else:
        return None

    critic.eval().to(device)
    for p in critic.parameters():
        p.requires_grad = False
    return critic


def load_zarr_data(data_dir: Path, task_name: str):
    """Load full_state and action arrays from the Adroit zarr dataset."""
    zarr_path = data_dir / f"adroit_{task_name}_expert.zarr"
    if not zarr_path.exists():
        # Also try SAC rollout naming
        zarr_path = data_dir / f"adroit_{task_name}_sac_rgb.zarr"
    if not zarr_path.exists():
        return None, None, None
    print(f"  Loading data from {zarr_path}")
    root = zarr.open(str(zarr_path), mode="r")
    full_state = np.array(root["data"]["full_state"])
    action = np.array(root["data"]["action"])
    episode_ends = np.array(root["meta"]["episode_ends"])
    return full_state, action, episode_ends


def run_gradient_ascent(
    critic: nn.Module,
    state: torch.Tensor,
    init_action: torch.Tensor,
    lr: float = 0.01,
    n_steps: int = 500,
    action_clamp: float = 1.0,
):
    """
    Maximize Q(s, a) via gradient ascent on the action.

    Returns:
        q_trajectory: list of Q values at each step
        action_trajectory: list of action tensors at each step
        final_action: optimized action tensor
    """
    action = init_action.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([action], lr=lr)

    q_trajectory = []
    action_trajectory = []

    for step in range(n_steps):
        optimizer.zero_grad()
        sa = torch.cat([state.detach(), action], dim=-1).unsqueeze(0)
        q = critic(sa).squeeze()
        loss = -q  # maximize Q
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            action.clamp_(-action_clamp, action_clamp)

        q_trajectory.append(q.item())
        action_trajectory.append(action.detach().clone())

    return q_trajectory, action_trajectory, action.detach().clone()


def parse_args():
    p = argparse.ArgumentParser(description="Critic gradient ascent debug for Adroit")
    p.add_argument("--task", type=str, nargs="+", default=["door"],
                    help="Adroit task(s) to test (door, hammer, pen, relocate)")
    p.add_argument("--data-dir", type=str,
                    default=str(REPO_ROOT / "ManiFlow" / "data"),
                    help="Directory containing adroit zarr datasets")
    p.add_argument("--critic-dir", type=str,
                    default=str(REPO_ROOT / "runs" / "sb3_adroit_sac" / "models_1"),
                    help="Directory containing per-task critic checkpoints")
    p.add_argument("--state-dim", type=int, default=0,
                    help="Override state dim (0 = auto per task)")
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--n-steps", type=int, default=400)
    p.add_argument("--n-test-states", type=int, default=10)
    p.add_argument("--n-traj", type=int, default=20,
                    help="Use states from the first N trajectories")
    p.add_argument("--output-dir", type=str,
                    default=str(REPO_ROOT / "scripts" / "critic_debug_output"))
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)
    critic_dir = Path(args.critic_dir)

    print(f"Device: {device}")
    print(f"Gradient Ascent: lr={args.lr}, steps={args.n_steps}")
    print("=" * 70)

    all_results = {}

    for task_name in args.task:
        state_dim = args.state_dim if args.state_dim > 0 else ADROIT_STATE_DIMS.get(task_name, 39)

        print(f"\n{'='*70}")
        print(f"TASK: {task_name}  (state_dim={state_dim}, action_dim={ACTION_DIM})")

        critic = load_critic(critic_dir, task_name, state_dim, device)
        if critic is None:
            print(f"  [SKIP] No critic found at {critic_dir / task_name}")
            continue

        full_state, action_data, episode_ends = load_zarr_data(data_dir, task_name)
        if full_state is None:
            print(f"  [SKIP] No zarr data found for {task_name} in {data_dir}")
            continue

        n_total = full_state.shape[0]
        rng = np.random.default_rng(42)
        if episode_ends is not None and len(episode_ends) >= args.n_traj:
            max_idx = int(episode_ends[args.n_traj - 1])
        else:
            max_idx = n_total
        print(f"  Using first {args.n_traj} trajectories: {max_idx} steps (out of {n_total} total)")

        sample_indices = rng.choice(max_idx, size=min(args.n_test_states, max_idx), replace=False)

        task_results = []

        # 3 rows: Q trajectory, action dims from noise, action dims from expert
        # Only plot first 6 action dims to keep readable (Adroit has 28 dims)
        n_plot_dims = min(6, ACTION_DIM)

        fig, axes = plt.subplots(
            3, args.n_test_states,
            figsize=(5 * args.n_test_states, 12),
            squeeze=False,
        )

        for si, idx in enumerate(sample_indices):
            state = torch.tensor(full_state[idx], dtype=torch.float32, device=device)
            expert_action = torch.tensor(action_data[idx], dtype=torch.float32, device=device)

            with torch.no_grad():
                sa_expert = torch.cat([state, expert_action]).unsqueeze(0)
                q_expert = critic(sa_expert).item()

            random_action = torch.randn(ACTION_DIM, device=device) * 2 - 1
            with torch.no_grad():
                sa_random = torch.cat([state, random_action]).unsqueeze(0)
                q_random = critic(sa_random).item()

            print(f"\n  Sample {si} (idx={idx}):")
            print(f"    Q(expert)    = {q_expert:.4f}")
            print(f"    Q(random)    = {q_random:.4f}")

            # Test 1: from random noise
            q_traj_from_noise, action_traj_noise, final_action_noise = run_gradient_ascent(
                critic, state, random_action, lr=args.lr, n_steps=args.n_steps
            )

            q_final_noise = q_traj_from_noise[-1]
            q_increase_noise = q_final_noise - q_random
            print(f"    [From noise]  Q: {q_random:.4f} -> {q_final_noise:.4f} (d={q_increase_noise:.4f})")

            dist_to_expert = torch.norm(final_action_noise - expert_action).item()
            print(f"    L2 dist(optimized, expert) = {dist_to_expert:.4f}")

            # Test 2: from expert action
            q_traj_from_expert, action_traj_expert, final_action_expert = run_gradient_ascent(
                critic, state, expert_action, lr=args.lr, n_steps=args.n_steps
            )

            q_final_expert = q_traj_from_expert[-1]
            q_increase_expert = q_final_expert - q_expert
            dist_expert_drift = torch.norm(final_action_expert - expert_action).item()
            print(f"    [From expert] Q: {q_expert:.4f} -> {q_final_expert:.4f} (d={q_increase_expert:.4f})")
            print(f"    Expert drift = {dist_expert_drift:.4f}")

            task_results.append({
                "q_expert": q_expert,
                "q_random": q_random,
                "q_final_from_noise": q_final_noise,
                "q_final_from_expert": q_final_expert,
                "q_increase_noise": q_increase_noise,
                "q_increase_expert": q_increase_expert,
                "dist_to_expert": dist_to_expert,
                "expert_drift": dist_expert_drift,
            })

            # Row 1: Q trajectory
            ax = axes[0][si]
            ax.plot(q_traj_from_noise, label="from noise", color="blue", alpha=0.8)
            ax.plot(q_traj_from_expert, label="from expert", color="green", alpha=0.8)
            ax.axhline(q_expert, color="red", linestyle="--", alpha=0.5, label=f"Q(expert)={q_expert:.2f}")
            ax.set_title(f"Sample {si} | Q trajectory", fontsize=9)
            ax.set_xlabel("Grad ascent step")
            ax.set_ylabel("Q(s, a)")
            ax.legend(fontsize=7)

            # Row 2: Action dims from noise (first N dims)
            ax2 = axes[1][si]
            action_hist = torch.stack(action_traj_noise).cpu().numpy()
            for dim in range(n_plot_dims):
                ax2.plot(action_hist[:, dim], label=f"dim {dim}", alpha=0.8)
                ax2.axhline(expert_action[dim].item(), color=f"C{dim}", linestyle="--", alpha=0.3)
            ax2.set_title(f"Action dims 0-{n_plot_dims-1} (from noise)", fontsize=9)
            ax2.set_xlabel("Grad ascent step")
            ax2.set_ylabel("Action value")
            ax2.legend(fontsize=7)

            # Row 3: Action dims from expert (first N dims)
            ax3 = axes[2][si]
            action_hist_exp = torch.stack(action_traj_expert).cpu().numpy()
            for dim in range(n_plot_dims):
                ax3.plot(action_hist_exp[:, dim], label=f"dim {dim}", alpha=0.8)
                ax3.axhline(expert_action[dim].item(), color=f"C{dim}", linestyle="--", alpha=0.3)
            ax3.set_title(f"Action dims 0-{n_plot_dims-1} (from expert)", fontsize=9)
            ax3.set_xlabel("Grad ascent step")
            ax3.set_ylabel("Action value")
            ax3.legend(fontsize=7)

        plt.suptitle(
            f"{task_name} (Adroit) -- Critic Gradient Ascent Test\n"
            f"lr={args.lr}, {args.n_steps} steps, state_dim={state_dim}, action_dim={ACTION_DIM}",
            fontsize=12,
        )
        plt.tight_layout()
        out_path = output_dir / f"critic_grad_ascent_adroit_{task_name}.png"
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"  -> Saved: {out_path}")

        all_results[task_name] = task_results

    # ─── Print summary ───────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY: Adroit Critic Gradient Ascent Test")
    print("=" * 70)
    print(
        f"{'Task':<12} "
        f"{'Q(expert)':>10} {'Q(random)':>10} "
        f"{'Q->(noise)':>10} {'dQ(noise)':>10} "
        f"{'Q->(expert)':>11} {'dQ(expert)':>11} "
        f"{'Dist->exp':>9} {'Diagnosis':>12}"
    )
    print("-" * 105)

    for task_name, results in all_results.items():
        avg_q_expert = np.mean([r["q_expert"] for r in results])
        avg_q_random = np.mean([r["q_random"] for r in results])
        avg_q_final_noise = np.mean([r["q_final_from_noise"] for r in results])
        avg_dq_noise = np.mean([r["q_increase_noise"] for r in results])
        avg_q_final_expert = np.mean([r["q_final_from_expert"] for r in results])
        avg_dq_expert = np.mean([r["q_increase_expert"] for r in results])
        avg_dist = np.mean([r["dist_to_expert"] for r in results])

        if abs(avg_dq_noise) < 1.0 and abs(avg_dq_expert) < 1.0:
            diagnosis = "FLAT"
        elif avg_dist > 5.0:
            diagnosis = "DIVERGE"
        elif avg_q_final_noise >= avg_q_expert * 0.9:
            diagnosis = "GOOD"
        else:
            diagnosis = "WEAK"

        print(
            f"{task_name:<12} "
            f"{avg_q_expert:>10.2f} {avg_q_random:>10.2f} "
            f"{avg_q_final_noise:>10.2f} {avg_dq_noise:>+10.2f} "
            f"{avg_q_final_expert:>11.2f} {avg_dq_expert:>+11.2f} "
            f"{avg_dist:>9.3f} {diagnosis:>12}"
        )

    print("\nInterpretation:")
    print("  FLAT:    Q barely changes -> critic has no gradient signal for ACGD")
    print("  DIVERGE: Action explodes  -> critic is unreliable out-of-distribution")
    print("  WEAK:    Q increases but doesn't reach expert level -> marginal signal")
    print("  GOOD:    Gradient ascent finds expert-quality actions -> critic works!")
    print(f"\nAll plots saved to: {output_dir}/")


if __name__ == "__main__":
    main()
