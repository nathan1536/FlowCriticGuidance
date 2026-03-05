#!/usr/bin/env python3
"""
Offline IQL (Implicit Q-Learning) critic training for ACGD.

Trains Q(s,a) and V(s) networks on a pre-collected zarr dataset (from
rollout_sb3_ppo_metaworld_rgb_to_zarr.py). The trained Q-critic is saved in
the same format as the PPO-trained critic, so it can be directly loaded by
the ManiFlow ACGD workspace.

Key advantages over PPO-coupled critic training:
  - No distribution shift: trains directly on the dataset states ACGD will see
  - No OOD overestimation: IQL never queries max_a Q(s', a)
  - Works purely offline: no PPO dependency at critic-training time

Reference:
  Kostrikov et al., "Offline Reinforcement Learning with Implicit Q-Learning",
  ICLR 2022. https://arxiv.org/abs/2110.06169

Usage:
  python scripts/train_iql_critic.py \\
      --task window-open \\
      --zarr-path ManiFlow/data/metaworld_window-open_expert.zarr \\
      --save-dir runs/sb3_metaworld_ppo/models_v3/window-open \\
      --epochs 500 --batch-size 2048

  # The output file best_expert_critic_q.pt is directly compatible with ACGD.
"""

import argparse
import copy
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import zarr


# ═══════════════════════════════════════════════════════════════════════
# Network definitions
# ═══════════════════════════════════════════════════════════════════════

def build_q_network(state_dim: int, action_dim: int, hidden_dim: int) -> nn.Module:
    """
    Q(s, a) -> scalar.
    Architecture MUST match build_qcritic() in workspace / train_sb3_ppo_metaworld.py
    so that saved weights are interchangeable.
    """
    return nn.Sequential(
        nn.Linear(state_dim + action_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, 1),
    )


def build_v_network(state_dim: int, hidden_dim: int) -> nn.Module:
    """V(s) -> scalar.  Only takes state (no action)."""
    return nn.Sequential(
        nn.Linear(state_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, 1),
    )


# ═══════════════════════════════════════════════════════════════════════
# IQL loss functions
# ═══════════════════════════════════════════════════════════════════════

def expectile_loss(diff: torch.Tensor, expectile: float = 0.7) -> torch.Tensor:
    """
    Asymmetric squared loss used for V-network training in IQL.

    L_τ(u) = |τ - 1(u < 0)| * u²

    When τ > 0.5, the loss penalises under-estimation more than over-estimation,
    causing V(s) to approximate an upper expectile of Q(s,a) under the dataset
    action distribution — effectively an implicit max without querying OOD actions.
    """
    weight = torch.where(diff > 0, expectile, 1.0 - expectile)
    return (weight * diff.pow(2)).mean()


# ═══════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════

def load_zarr_transitions(zarr_path: str, device: torch.device):
    """
    Load (s, a, r, s', done) transition tuples from a zarr dataset.

    Expects the zarr to contain:
      data/full_state       (N, state_dim)
      data/next_full_state  (N, state_dim)
      data/action           (N, action_dim)
      data/reward           (N,)
      data/done             (N,)
      meta/episode_ends     (E,)

    If next_full_state / done are missing, they are reconstructed from
    full_state + episode_ends (backward compatible with older zarr files).
    """
    root = zarr.open(str(zarr_path), mode="r")
    data = root["data"]
    meta = root["meta"]

    full_state = torch.tensor(np.array(data["full_state"]), dtype=torch.float32, device=device)
    action = torch.tensor(np.array(data["action"]), dtype=torch.float32, device=device)
    reward = torch.tensor(np.array(data["reward"]), dtype=torch.float32, device=device)

    # next_full_state: use explicit array if available, else reconstruct
    if "next_full_state" in data:
        next_full_state = torch.tensor(np.array(data["next_full_state"]), dtype=torch.float32, device=device)
    else:
        # Shift full_state by 1; last step of each episode wraps (not ideal, but usable)
        next_full_state = torch.roll(full_state, -1, dims=0)
        episode_ends = np.array(meta["episode_ends"])
        # Mark last step of each episode: next_state = current_state (will be masked by done)
        for end_idx in episode_ends:
            next_full_state[int(end_idx) - 1] = full_state[int(end_idx) - 1]

    # done: use explicit array if available, else reconstruct from episode_ends
    if "done" in data:
        done = torch.tensor(np.array(data["done"]), dtype=torch.float32, device=device)
    else:
        episode_ends = np.array(meta["episode_ends"])
        done = torch.zeros(full_state.shape[0], dtype=torch.float32, device=device)
        for end_idx in episode_ends:
            done[int(end_idx) - 1] = 1.0

    N = full_state.shape[0]
    state_dim = full_state.shape[1]
    action_dim = action.shape[1]
    print(f"[IQL Data] Loaded {N} transitions from {zarr_path}")
    print(f"           state_dim={state_dim}  action_dim={action_dim}")
    print(f"           reward: mean={reward.mean():.2f} std={reward.std():.2f} "
          f"min={reward.min():.2f} max={reward.max():.2f}")
    print(f"           done: {done.sum().int().item()} terminal steps")

    return full_state, action, reward, next_full_state, done


# ═══════════════════════════════════════════════════════════════════════
# Diagnostics (compatible with debug_critic_gradient_ascent.py)
# ═══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def compute_q_diagnostics(
    q_net: nn.Module,
    states: torch.Tensor,
    actions: torch.Tensor,
    n_diag: int = 200,
    noise_std: float = 0.3,
):
    """
    Compute Q-gap diagnostics: Q(expert) vs Q(noisy) vs Q(random).
    Same metrics as TrainQCriticCallback in train_sb3_ppo_metaworld.py.
    """
    device = states.device
    n = min(n_diag, states.shape[0])
    idx = torch.randperm(states.shape[0], device=device)[:n]
    s = states[idx]
    a = actions[idx]
    act_dim = a.shape[1]

    q_expert = q_net(torch.cat([s, a], dim=-1)).mean().item()
    q_noisy = q_net(
        torch.cat([s, a + torch.randn(n, act_dim, device=device) * noise_std], dim=-1)
    ).mean().item()
    q_random = q_net(
        torch.cat([s, torch.rand(n, act_dim, device=device) * 2 - 1], dim=-1)
    ).mean().item()

    gap_vs_noisy = q_expert - q_noisy
    gap_vs_random = q_expert - q_random

    return {
        "q_expert": q_expert,
        "q_noisy": q_noisy,
        "q_random": q_random,
        "gap_vs_noisy": gap_vs_noisy,
        "gap_vs_random": gap_vs_random,
    }


# ═══════════════════════════════════════════════════════════════════════
# Training loop
# ═══════════════════════════════════════════════════════════════════════

def train_iql(
    states: torch.Tensor,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    next_states: torch.Tensor,
    dones: torch.Tensor,
    *,
    state_dim: int,
    action_dim: int,
    hidden_dim: int = 256,
    lr: float = 3e-4,
    epochs: int = 500,
    batch_size: int = 2048,
    gamma: float = 0.99,
    expectile_tau: float = 0.7,
    polyak_tau: float = 0.005,
    save_dir: Path,
    diag_every: int = 10,
    device: torch.device = torch.device("cpu"),
    # ── OOD suppression parameters ──
    cql_alpha: float = 5.0,
    gp_weight: float = 2.0,
    margin_weight: float = 10.0,
    margin: float = 5.0,
    neg_noise_std: float = 0.3,
    random_neg_ratio: float = 0.5,
    cql_k: int = 10,
    q_clip_min: float = -10.0,
    q_clip_max: float = 50.0,
):
    """Full IQL training loop with OOD suppression (CQL + contrastive + GP)."""
    N = states.shape[0]

    # ── Build networks ────────────────────────────────────────────────
    q_net = build_q_network(state_dim, action_dim, hidden_dim).to(device)
    q_target = copy.deepcopy(q_net)
    q_target.requires_grad_(False)
    v_net = build_v_network(state_dim, hidden_dim).to(device)

    q_optimizer = torch.optim.Adam(q_net.parameters(), lr=lr)
    v_optimizer = torch.optim.Adam(v_net.parameters(), lr=lr)

    rank_fn = nn.MarginRankingLoss(margin=margin)

    print(f"\n[IQL] Networks initialised on {device}")
    print(f"      Q params: {sum(p.numel() for p in q_net.parameters()):,}")
    print(f"      V params: {sum(p.numel() for p in v_net.parameters()):,}")
    print(f"      OOD suppression: cql_alpha={cql_alpha}  gp_weight={gp_weight}  "
          f"margin_weight={margin_weight}  margin={margin}")
    print(f"      Q-clip: [{q_clip_min}, {q_clip_max}]  CQL K={cql_k}")

    best_gap_vs_noisy = -float("inf")
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(1, epochs + 1):
        perm = torch.randperm(N, device=device)
        epoch_v_loss = 0.0
        epoch_q_loss = 0.0
        epoch_rank_loss = 0.0
        epoch_cql_loss = 0.0
        epoch_gp_loss = 0.0
        epoch_total_loss = 0.0
        n_batches = 0

        for i in range(0, N, batch_size):
            idx = perm[i : i + batch_size]
            s = states[idx]
            a = actions[idx]
            r = rewards[idx]
            s_next = next_states[idx]
            d = dones[idx]
            bs = s.shape[0]

            # ── 1. Update V via expectile regression ──────────────
            with torch.no_grad():
                sa = torch.cat([s, a], dim=-1)
                q_target_val = q_target(sa).squeeze(-1)  # (bs,)

            v_pred = v_net(s).squeeze(-1)  # (bs,)
            v_loss = expectile_loss(q_target_val - v_pred, expectile=expectile_tau)

            v_optimizer.zero_grad(set_to_none=True)
            v_loss.backward()
            v_optimizer.step()

            # ── 2. Update Q via Bellman + contrastive + CQL + GP ──

            # 2a. TD loss with Q-value clipping
            with torch.no_grad():
                v_next = v_net(s_next).squeeze(-1)  # (bs,)
                q_bellman_target = r + gamma * v_next * (1.0 - d)  # (bs,)
                # Clip TD targets to prevent unbounded Q growth
                q_bellman_target = torch.clamp(q_bellman_target, min=q_clip_min, max=q_clip_max)

            sa = torch.cat([s, a], dim=-1)
            q_pred = q_net(sa).squeeze(-1)  # (bs,)
            q_loss = nn.functional.mse_loss(q_pred, q_bellman_target)

            # 2b. Contrastive ranking loss: Q(s, a_expert) > Q(s, a_neg) + margin
            n_neg = min(bs, 256)
            n_noisy_neg = int(n_neg * (1.0 - random_neg_ratio))
            n_rand_neg = n_neg - n_noisy_neg

            neg_actions = torch.empty(n_neg, action_dim, device=device)
            if n_noisy_neg > 0:
                neg_actions[:n_noisy_neg] = a[:n_noisy_neg] + \
                    torch.randn(n_noisy_neg, action_dim, device=device) * neg_noise_std
            if n_rand_neg > 0:
                neg_actions[n_noisy_neg:] = torch.rand(n_rand_neg, action_dim, device=device) * 2 - 1

            neg_sa = torch.cat([s[:n_neg], neg_actions], dim=-1)
            q_neg = q_net(neg_sa).squeeze(-1)
            q_pos_for_rank = q_net(torch.cat([s[:n_neg], a[:n_neg]], dim=-1)).squeeze(-1)

            rank_target = torch.ones(n_neg, device=device)
            loss_rank = rank_fn(q_pos_for_rank, q_neg, rank_target)

            # 2c. CQL conservative penalty: push down Q for random OOD actions
            loss_cql = torch.tensor(0.0, device=device)
            if cql_alpha > 0:
                n_cql = min(bs, 128)
                cql_states = s[:n_cql].unsqueeze(1).expand(-1, cql_k, -1).reshape(-1, state_dim)
                cql_rand_acts = torch.rand(n_cql * cql_k, action_dim, device=device) * 2 - 1
                cql_sa = torch.cat([cql_states, cql_rand_acts], dim=-1)
                q_ood = q_net(cql_sa).reshape(n_cql, cql_k)

                q_data = q_net(torch.cat([s[:n_cql], a[:n_cql]], dim=-1)).squeeze(-1)

                # CQL: push down logsumexp(Q_ood) while keeping Q_data high
                loss_cql = torch.logsumexp(q_ood, dim=1).mean() - q_data.mean()

            # 2d. Gradient penalty: penalize large ||grad_a Q(s,a)||
            #     Smooths Q landscape to prevent sharp spurious peaks
            loss_gp = torch.tensor(0.0, device=device)
            if gp_weight > 0:
                gp_n = min(bs, 128)
                gp_actions = a[:gp_n].detach().requires_grad_(True)
                gp_sa = torch.cat([s[:gp_n].detach(), gp_actions], dim=-1)
                gp_q = q_net(gp_sa)
                grad_a = torch.autograd.grad(
                    gp_q.sum(), gp_actions, create_graph=True
                )[0]
                loss_gp = (grad_a ** 2).sum(dim=-1).mean()

            # 2e. Combined Q loss
            q_loss_total = (q_loss
                            + margin_weight * loss_rank
                            + cql_alpha * loss_cql
                            + gp_weight * loss_gp)

            q_optimizer.zero_grad(set_to_none=True)
            q_loss_total.backward()
            q_optimizer.step()

            epoch_v_loss += v_loss.item()
            epoch_q_loss += q_loss.item()
            epoch_rank_loss += loss_rank.item()
            epoch_cql_loss += loss_cql.item()
            epoch_gp_loss += loss_gp.item()
            epoch_total_loss += q_loss_total.item()
            n_batches += 1

        # ── 3. Soft-update Q-target ───────────────────────────────
        for p_tgt, p in zip(q_target.parameters(), q_net.parameters()):
            p_tgt.data.mul_(1.0 - polyak_tau).add_(p.data * polyak_tau)

        avg_v = epoch_v_loss / max(n_batches, 1)
        avg_q = epoch_q_loss / max(n_batches, 1)
        avg_rank = epoch_rank_loss / max(n_batches, 1)
        avg_cql = epoch_cql_loss / max(n_batches, 1)
        avg_gp = epoch_gp_loss / max(n_batches, 1)
        avg_total = epoch_total_loss / max(n_batches, 1)

        # ── Diagnostics ──────────────────────────────────────────
        if epoch % diag_every == 0 or epoch == 1:
            diag = compute_q_diagnostics(q_net, states, actions)
            gap_noisy = diag["gap_vs_noisy"]
            gap_random = diag["gap_vs_random"]

            print(
                f"[IQL epoch {epoch:4d}/{epochs}] "
                f"V={avg_v:.4f}  Q_td={avg_q:.4f}  rank={avg_rank:.4f}  "
                f"cql={avg_cql:.4f}  gp={avg_gp:.4f}  total={avg_total:.4f}  |  "
                f"Q(exp)={diag['q_expert']:.2f}  Q(noisy)={diag['q_noisy']:.2f}  "
                f"Q(rand)={diag['q_random']:.2f}  "
                f"gap_noisy={gap_noisy:+.3f}  gap_rand={gap_random:+.3f}"
            )

            # Save best critic by gap_vs_noisy
            if gap_noisy > best_gap_vs_noisy:
                best_gap_vs_noisy = gap_noisy
                best_path = save_dir / "best_expert_critic_q.pt"
                torch.save(q_net.state_dict(), str(best_path))
                print(f"  -> Saved best critic (gap_noisy={gap_noisy:.4f}) to {best_path}")

    # ── Save final models ─────────────────────────────────────────
    final_q_path = save_dir / "iql_critic_q_final.pt"
    final_v_path = save_dir / "iql_value_v_final.pt"
    torch.save(q_net.state_dict(), str(final_q_path))
    torch.save(v_net.state_dict(), str(final_v_path))
    print(f"\n[IQL] Training complete.")
    print(f"  Final Q-critic:  {final_q_path}")
    print(f"  Final V-network: {final_v_path}")
    print(f"  Best Q-critic (gap_noisy={best_gap_vs_noisy:.4f}): {save_dir / 'best_expert_critic_q.pt'}")

    return q_net, v_net


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Train an IQL Q(s,a) critic offline from a zarr dataset."
    )
    p.add_argument("--task", type=str, required=True, help="MetaWorld task name (e.g. window-open)")
    p.add_argument(
        "--zarr-path", type=str, required=True,
        help="Path to zarr dataset with (full_state, next_full_state, action, reward, done)."
    )
    p.add_argument("--save-dir", type=str, default="", help="Output directory for critic weights. Default: runs/sb3_metaworld_ppo/models_v3/<task>")
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    p.add_argument("--expectile-tau", type=float, default=0.7, help="IQL expectile (0.5=mean, 0.9=aggressive)")
    p.add_argument("--polyak-tau", type=float, default=0.005, help="Target network Polyak averaging rate")
    p.add_argument("--diag-every", type=int, default=10, help="Print diagnostics every N epochs")
    p.add_argument("--device", type=str, default="auto", help="Device: auto|cpu|cuda|cuda:0")
    p.add_argument("--seed", type=int, default=42)

    # ── OOD suppression: Contrastive ranking, CQL, Gradient penalty ──
    p.add_argument("--cql-alpha", type=float, default=1.0,
                   help="CQL conservative penalty weight. Pushes down Q for random OOD actions. "
                        "Higher = more aggressive OOD suppression. (0 = disable)")
    p.add_argument("--gp-weight", type=float, default=0.5,
                   help="Gradient penalty weight. Smooths Q landscape to prevent sharp spurious peaks. "
                        "(0 = disable)")
    p.add_argument("--margin-weight", type=float, default=5.0,
                   help="Weight of contrastive margin ranking loss. "
                        "Enforces Q(expert) > Q(random/noisy) + margin.")
    p.add_argument("--margin", type=float, default=5.0,
                   help="Margin for ranking loss: Q(s, a_good) > Q(s, a_bad) + margin.")
    p.add_argument("--neg-noise-std", type=float, default=0.3,
                   help="Std of Gaussian noise for noisy negative action samples.")
    p.add_argument("--random-neg-ratio", type=float, default=0.5,
                   help="Fraction of negative samples that are purely random (vs noisy perturbations).")
    p.add_argument("--cql-k", type=int, default=10,
                   help="Number of random actions per state for CQL logsumexp penalty.")
    p.add_argument("--q-clip-min", type=float, default=-10.0,
                   help="Lower bound for TD target clipping (prevents Q collapse).")
    p.add_argument("--q-clip-max", type=float, default=50.0,
                   help="Upper bound for TD target clipping (prevents unbounded Q growth).")
    return p.parse_args()


def main():
    args = parse_args()

    # Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[IQL] Device: {device}")

    # Save directory
    repo_root = Path(__file__).resolve().parents[1]
    if args.save_dir:
        save_dir = Path(args.save_dir)
    else:
        save_dir = repo_root / "runs" / "sb3_metaworld_ppo" / "models_v3" / args.task
    os.makedirs(save_dir, exist_ok=True)

    # Load data
    zarr_path = args.zarr_path
    if not Path(zarr_path).is_absolute():
        zarr_path = str(repo_root / zarr_path)

    states, actions, rewards, next_states, dones = load_zarr_transitions(zarr_path, device)
    state_dim = states.shape[1]
    action_dim = actions.shape[1]

    # Train
    print(f"\n[IQL] Config:")
    print(f"  task={args.task}  epochs={args.epochs}  batch_size={args.batch_size}")
    print(f"  lr={args.lr}  gamma={args.gamma}  expectile_tau={args.expectile_tau}")
    print(f"  polyak_tau={args.polyak_tau}  hidden_dim={args.hidden_dim}")
    print(f"  state_dim={state_dim}  action_dim={action_dim}")
    print(f"  cql_alpha={args.cql_alpha}  gp_weight={args.gp_weight}")
    print(f"  margin_weight={args.margin_weight}  margin={args.margin}")
    print(f"  q_clip=[{args.q_clip_min}, {args.q_clip_max}]  cql_k={args.cql_k}")
    print(f"  save_dir={save_dir}")

    train_iql(
        states, actions, rewards, next_states, dones,
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        gamma=args.gamma,
        expectile_tau=args.expectile_tau,
        polyak_tau=args.polyak_tau,
        save_dir=save_dir,
        diag_every=args.diag_every,
        device=device,
        cql_alpha=args.cql_alpha,
        gp_weight=args.gp_weight,
        margin_weight=args.margin_weight,
        margin=args.margin,
        neg_noise_std=args.neg_noise_std,
        random_neg_ratio=args.random_neg_ratio,
        cql_k=args.cql_k,
        q_clip_min=args.q_clip_min,
        q_clip_max=args.q_clip_max,
    )


if __name__ == "__main__":
    main()
