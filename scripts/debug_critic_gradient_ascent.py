#!/usr/bin/env python3
"""
Critic-Only Gradient Ascent Test:
Start from random noise (or expert action + noise) and use gradient ascent
on Q(s, a) to see if the critic can guide actions toward high-value regions.

If gradient ascent can't even increase Q meaningfully, the critic is the problem.

Usage:
  cd /home/nathan4074/Project/ManiFlow_Policy/ManiFlow
  python ../scripts/debug_critic_gradient_ascent.py

Output:
  - Console log showing Q trajectory during optimization
  - PNG plots saved to scripts/critic_debug_output/ (one per task)
"""

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
DATA_DIR = REPO_ROOT / "ManiFlow" / "data" / "SAC"
CRITIC_DIR = REPO_ROOT / "runs" / "sb3_metaworld_sac" / "models"
OUTPUT_DIR = REPO_ROOT / "scripts" / "critic_debug_output"

STATE_DIM = 39
ACTION_DIM = 4
HIDDEN_DIM = 128

# Gradient ascent settings
LR = 0.01
N_STEPS = 400
N_TEST_STATES = 10       # test with multiple states per task

# Tasks that have BOTH zarr data and a Q-critic
TASK_CANDIDATES = [


    "coffee-pull",
    "peg-insert-side",
    "pick-place",
    "door-lock",
    "bin-picking",



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
    action_clamp: float = 1.0
):
    """
    Maximize Q(s, a) via gradient ascent on the action.
    
    Returns:
        q_trajectory: list of Q values at each step
        action_trajectory: list of action tensors at each step
        final_action: optimized action tensor
    """
    device = state.device
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

        # Clamp action to reasonable range
        with torch.no_grad():
            action.clamp_(-action_clamp, action_clamp)

        q_trajectory.append(q.item())
        action_trajectory.append(action.detach().clone())

    return q_trajectory, action_trajectory, action.detach().clone()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Gradient Ascent: lr={LR}, steps={N_STEPS}")
    print("=" * 70)

    all_results = {}

    for task_name in TASK_CANDIDATES:
        # Check critic exists
        critic_path = CRITIC_DIR / task_name / "best_expert_critic_q.pt"
        if not critic_path.exists():
            print(f"[SKIP] No critic for {task_name}")
            continue

        # Load data
        full_state, action_data, episode_ends = load_zarr_data(task_name)
        if full_state is None:
            print(f"[SKIP] No zarr data for {task_name}")
            continue

        # Load critic
        critic = build_critic(STATE_DIM, ACTION_DIM, HIDDEN_DIM)
        critic.load_state_dict(torch.load(str(critic_path), map_location=device))
        critic.eval().to(device)

        # Freeze critic parameters (we only optimize the action)
        for p in critic.parameters():
            p.requires_grad = False

        print(f"\n{'='*70}")
        print(f"TASK: {task_name}")

        n_total = full_state.shape[0]
        rng = np.random.default_rng(42)
        # Only use states from the first 10 trajectories (what the student sees)
        N_TRAJ = 30
        if episode_ends is not None and len(episode_ends) >= N_TRAJ:
            max_idx = int(episode_ends[N_TRAJ - 1])  # end of 10th episode
        else:
            max_idx = n_total
        print(f"  Using first {N_TRAJ} trajectories: {max_idx} steps (out of {n_total} total)")

        sample_indices = rng.choice(max_idx, size=min(N_TEST_STATES, max_idx), replace=False)

        task_results = []

        # Create figure: 2 rows × N_TEST_STATES cols
        # Row 1: Q trajectory during gradient ascent
        # Row 2: Action dimension trajectories
        fig, axes = plt.subplots(
            3, N_TEST_STATES,
            figsize=(5 * N_TEST_STATES, 12),
            squeeze=False,
        )

        for si, idx in enumerate(sample_indices):
            state = torch.tensor(full_state[idx], dtype=torch.float32, device=device)
            expert_action = torch.tensor(action_data[idx], dtype=torch.float32, device=device)

            # Q value for expert action
            with torch.no_grad():
                sa_expert = torch.cat([state, expert_action]).unsqueeze(0)
                q_expert = critic(sa_expert).item()

            # Q value for random action
            random_action = torch.randn(ACTION_DIM, device=device)* 2 - 1 
            with torch.no_grad():
                sa_random = torch.cat([state, random_action]).unsqueeze(0)
                q_random = critic(sa_random).item()

            # === Test 1: Gradient ascent from RANDOM noise ===
            print(f"\n  Sample {si} (idx={idx}):")
            print(f"    Q(expert)    = {q_expert:.4f}")
            print(f"    Q(random)    = {q_random:.4f}")

            q_traj_from_noise, action_traj_noise, final_action_noise = run_gradient_ascent(
                critic, state, random_action, lr=LR, n_steps=N_STEPS
            )

            q_final_noise = q_traj_from_noise[-1]
            q_increase_noise = q_final_noise - q_random
            print(f"    [From noise]  Q: {q_random:.4f} → {q_final_noise:.4f} (Δ={q_increase_noise:.4f})")
            print(f"    Final action: {final_action_noise.cpu().numpy()}")
            print(f"    Expert action: {expert_action.cpu().numpy()}")

            # Distance between optimized and expert action
            dist_to_expert = torch.norm(final_action_noise - expert_action).item()
            print(f"    L2 dist(optimized, expert) = {dist_to_expert:.4f}")

            # === Test 2: Gradient ascent from EXPERT action (should stay near expert) ===
            q_traj_from_expert, action_traj_expert, final_action_expert = run_gradient_ascent(
                critic, state, expert_action, lr=LR, n_steps=N_STEPS
            )

            q_final_expert = q_traj_from_expert[-1]
            q_increase_expert = q_final_expert - q_expert
            dist_expert_drift = torch.norm(final_action_expert - expert_action).item()
            print(f"    [From expert] Q: {q_expert:.4f} → {q_final_expert:.4f} (Δ={q_increase_expert:.4f})")
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

            # ─── Row 1: Q trajectory ─────────────────────────
            ax = axes[0][si]
            ax.plot(q_traj_from_noise, label="from noise", color="blue", alpha=0.8)
            ax.plot(q_traj_from_expert, label="from expert", color="green", alpha=0.8)
            ax.axhline(q_expert, color="red", linestyle="--", alpha=0.5, label=f"Q(expert)={q_expert:.2f}")
            ax.set_title(f"Sample {si} | Q trajectory", fontsize=9)
            ax.set_xlabel("Grad ascent step")
            ax.set_ylabel("Q(s, a)")
            ax.legend(fontsize=7)

            # ─── Row 2: Action dims from noise ──────────────
            ax2 = axes[1][si]
            action_hist = torch.stack(action_traj_noise).cpu().numpy()  # (N_STEPS, 4)
            for dim in range(ACTION_DIM):
                ax2.plot(action_hist[:, dim], label=f"dim {dim}", alpha=0.8)
                ax2.axhline(
                    expert_action[dim].item(),
                    color=f"C{dim}",
                    linestyle="--",
                    alpha=0.3,
                )
            ax2.set_title(f"Action dims (from noise)", fontsize=9)
            ax2.set_xlabel("Grad ascent step")
            ax2.set_ylabel("Action value")
            ax2.legend(fontsize=7)

            # ─── Row 3: Action dims from expert ──────────────
            ax3 = axes[2][si]
            action_hist_exp = torch.stack(action_traj_expert).cpu().numpy()
            for dim in range(ACTION_DIM):
                ax3.plot(action_hist_exp[:, dim], label=f"dim {dim}", alpha=0.8)
                ax3.axhline(
                    expert_action[dim].item(),
                    color=f"C{dim}",
                    linestyle="--",
                    alpha=0.3,
                )
            ax3.set_title(f"Action dims (from expert)", fontsize=9)
            ax3.set_xlabel("Grad ascent step")
            ax3.set_ylabel("Action value")
            ax3.legend(fontsize=7)

        plt.suptitle(
            f"{task_name} — Critic Gradient Ascent Test\n"
            f"lr={LR}, {N_STEPS} steps | Can gradient ascent find good actions?",
            fontsize=12,
        )
        plt.tight_layout()
        out_path = OUTPUT_DIR / f"critic_grad_ascent_{task_name}.png"
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"  → Saved: {out_path}")

        all_results[task_name] = task_results

    # ─── Print summary ───────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY: Critic Gradient Ascent Test")
    print("=" * 70)
    print(
        f"{'Task':<18} "
        f"{'Q(expert)':>10} {'Q(random)':>10} "
        f"{'Q→(noise)':>10} {'ΔQ(noise)':>10} "
        f"{'Q→(expert)':>11} {'ΔQ(expert)':>11} "
        f"{'Dist→exp':>9} {'Diagnosis':>12}"
    )
    print("-" * 115)

    for task_name, results in all_results.items():
        # Average across test states
        avg_q_expert = np.mean([r["q_expert"] for r in results])
        avg_q_random = np.mean([r["q_random"] for r in results])
        avg_q_final_noise = np.mean([r["q_final_from_noise"] for r in results])
        avg_dq_noise = np.mean([r["q_increase_noise"] for r in results])
        avg_q_final_expert = np.mean([r["q_final_from_expert"] for r in results])
        avg_dq_expert = np.mean([r["q_increase_expert"] for r in results])
        avg_dist = np.mean([r["dist_to_expert"] for r in results])

        # Diagnosis
        if abs(avg_dq_noise) < 1.0 and abs(avg_dq_expert) < 1.0:
            diagnosis = "FLAT ❌"
        elif avg_dist > 2.0:
            diagnosis = "DIVERGE ⚠️"
        elif avg_q_final_noise >= avg_q_expert * 0.9:
            diagnosis = "GOOD ✅"
        else:
            diagnosis = "WEAK ⚠️"

        print(
            f"{task_name:<18} "
            f"{avg_q_expert:>10.2f} {avg_q_random:>10.2f} "
            f"{avg_q_final_noise:>10.2f} {avg_dq_noise:>+10.2f} "
            f"{avg_q_final_expert:>11.2f} {avg_dq_expert:>+11.2f} "
            f"{avg_dist:>9.3f} {diagnosis:>12}"
        )

    print("\nInterpretation:")
    print("  FLAT:    Q barely changes → critic has no gradient signal for ACGD")
    print("  DIVERGE: Action explodes → critic is unreliable out-of-distribution")
    print("  WEAK:    Q increases but doesn't reach expert level → marginal signal")
    print("  GOOD:    Gradient ascent finds expert-quality actions → critic works!")
    print(f"\nAll plots saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
