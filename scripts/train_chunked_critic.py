#!/usr/bin/env python3
"""
Distill a single-step SAC critic Q(s,a) into a chunked critic Q(s, a_t:t+k).

The chunked critic evaluates a full action chunk (e.g. 3 consecutive actions)
and is trained to predict the multi-step return:

    y_t = r_t + γ r_{t+1} + γ² r_{t+2} + γ³ V(s_{t+3})



Usage:
  python scripts/train_chunked_critic.py 
      --task door 
      --zarr-path runs/sb3_adroit_sac/models/door/replay_buffer_best.zarr 
      --save-dir runs/sb3_adroit_sac/models/door 
      --chunk-size 3 --epochs 1000
"""

import argparse
import copy
import os
from pathlib import Path

import sys

import numpy as np
import torch
import torch.nn as nn


def _ensure_sb3_importable(repo_root: Path) -> None:
    try:
        import stable_baselines3  # noqa: F401
        return
    except Exception:
        pass
    vendored_parent = repo_root / "third_party" / "dexart-release"
    sys.path.insert(0, str(vendored_parent))


# ═══════════════════════════════════════════════════════════════════════
# Network definitions
# ═══════════════════════════════════════════════════════════════════════

def build_chunked_q_network(
    state_dim: int, action_dim: int, chunk_size: int, hidden_dim: int,
) -> nn.Module:
    """Q_chunk(s_t, a_t, a_{t+1}, ..., a_{t+k-1}) -> scalar."""
    input_dim = state_dim + chunk_size * action_dim
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, 1),
    )


class TwinChunkedQCritic(nn.Module):
    """Conservative twin-Q chunked critic: returns min(Q0, Q1).

    """

    def __init__(
        self, state_dim: int, action_dim: int, chunk_size: int, hidden_dim: int,
    ):
        super().__init__()
        self.chunk_size = chunk_size
        self.q0 = build_chunked_q_network(state_dim, action_dim, chunk_size, hidden_dim)
        self.q1 = build_chunked_q_network(state_dim, action_dim, chunk_size, hidden_dim)
        # Target normalisation stats (persistent buffers saved in state_dict)
        self.register_buffer("target_mean", torch.tensor(0.0))
        self.register_buffer("target_std", torch.tensor(1.0))

    def set_target_stats(self, mean: float, std: float):
        self.target_mean.fill_(mean)
        self.target_std.fill_(max(std, 1e-8))

    def forward_normalised(self, sa_chunk: torch.Tensor) -> torch.Tensor:
        """Return min(Q0, Q1) in normalised space (for training loss)."""
        return torch.min(self.q0(sa_chunk), self.q1(sa_chunk))

    def forward(self, sa_chunk: torch.Tensor) -> torch.Tensor:
        """Return min(Q0, Q1) denormalised to original Q-value scale."""
        q_norm = self.forward_normalised(sa_chunk)
        return q_norm * self.target_std + self.target_mean


# ═══════════════════════════════════════════════════════════════════════
# Compute V(s) from SAC model
# ═══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def compute_v_from_sac(
    sac_model,
    full_state: np.ndarray,
    batch_size: int = 4096,
    n_samples: int = 10,
) -> np.ndarray:
    """
    Compute V(s) = E_{a~π}[min(Q0,Q1)(s,a) - α log π(a|s)]

    Matches exactly how SAC computes its TD target (see sac.py line 243).
    Uses n_samples stochastic action samples per state for the expectation.
    """
    device = sac_model.device
    qf0 = sac_model.critic.qf0
    qf1 = sac_model.critic.qf1
    qf0.eval()
    qf1.eval()

    # Get entropy coefficient α
    if hasattr(sac_model, 'log_ent_coef'):
        ent_coef = torch.exp(sac_model.log_ent_coef.detach()).item()
    else:
        ent_coef = sac_model.ent_coef_tensor.item()

    print(f"  SAC entropy coef α = {ent_coef:.4f}")

    N = full_state.shape[0]
    v_arr = np.zeros(N, dtype=np.float32)

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        bs = end - start
        s_t = torch.FloatTensor(full_state[start:end]).to(device)

        # Average over n_samples stochastic action samples
        v_sum = torch.zeros(bs, device=device)
        for _ in range(n_samples):
            # Sample action stochastically from π(·|s)
            pi_a, log_prob = sac_model.actor.action_log_prob(s_t)
            sa_pi = torch.cat([s_t, pi_a], dim=-1)

            # V(s) = min(Q0, Q1)(s, a) - α log π(a|s)
            q_val = torch.min(qf0(sa_pi), qf1(sa_pi)).squeeze(-1)
            v_sum += q_val - ent_coef * log_prob

        v_arr[start:end] = (v_sum / n_samples).cpu().numpy()

    return v_arr


# ═══════════════════════════════════════════════════════════════════════
# Multi-step return target computation
# ═══════════════════════════════════════════════════════════════════════

def compute_multistep_targets(
    full_state: np.ndarray,   # (N, state_dim)
    action: np.ndarray,       # (N, action_dim)
    reward: np.ndarray,       # (N,)
    v_value: np.ndarray,      # (N,)
    episode_ends: np.ndarray, # (E,)  cumulative episode boundary indices
    gamma: float = 0.99,
    chunk_size: int = 3,
):
    """
    Build chunked training data from replay buffer.

    For each valid index t (where t..t+chunk_size all lie within one episode):
        y_t = Σ_{k=0}^{chunk_size-1} γ^k r_{t+k}  +  γ^{chunk_size} V(s_{t+chunk_size})

    Returns:
        states:       (M, state_dim)
        action_chunks: (M, chunk_size * action_dim)
        targets:       (M,)
    """
    N = full_state.shape[0]
    action_dim = action.shape[1]

    # Build episode ranges: [(start, end), ...]
    ep_starts = np.concatenate([[0], episode_ends[:-1]])
    ep_ranges = list(zip(ep_starts.astype(int), episode_ends.astype(int)))

    # Collect valid indices: need t, t+1, ..., t+chunk_size all in same episode
    valid = []
    for s, e in ep_ranges:
        ep_len = e - s
        if ep_len > chunk_size:
            valid.extend(range(s, e - chunk_size))
    valid = np.array(valid, dtype=np.int64)

    print(f"[ChunkedCritic] {len(valid)} valid chunks from {len(ep_ranges)} episodes "
          f"(chunk_size={chunk_size}, N={N})")

    # Discount factors [1, γ, γ², ...]
    discounts = np.array([gamma ** k for k in range(chunk_size)], dtype=np.float32)

    # Vectorised reward sum: r[t] + γ r[t+1] + γ² r[t+2]
    offsets = np.arange(chunk_size)[None, :]  # (1, chunk_size)
    reward_indices = valid[:, None] + offsets   # (M, chunk_size)
    r_sum = (reward[reward_indices] * discounts[None, :]).sum(axis=1)

    # Bootstrap: γ^{chunk_size} * V(s_{t+chunk_size})
    bootstrap_idx = valid + chunk_size
    bootstrap_v = v_value[bootstrap_idx]
    targets = r_sum + (gamma ** chunk_size) * bootstrap_v

    # Collect states and flattened action chunks
    states_out = full_state[valid]  # (M, state_dim)
    action_chunks_out = np.concatenate(
        [action[valid + k] for k in range(chunk_size)],
        axis=-1,
    )  # (M, chunk_size * action_dim)

    print(f"[ChunkedCritic] Target stats: mean={targets.mean():.3f} "
          f"std={targets.std():.3f} min={targets.min():.3f} max={targets.max():.3f}")

    return states_out, action_chunks_out, targets


# ═══════════════════════════════════════════════════════════════════════
# Diagnostics
# ═══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def compute_chunked_q_diagnostics(
    critic: nn.Module,
    states: torch.Tensor,
    action_chunks: torch.Tensor,
    n_diag: int = 500,
    noise_scales: tuple = (0.01, 0.05, 0.1),
):
    """Q-gap: Q(expert_chunk) vs Q(noisy_chunk @ multiple scales) vs Q(random_chunk)."""
    device = states.device
    n = min(n_diag, states.shape[0])
    idx = torch.randperm(states.shape[0], device=device)[:n]
    s = states[idx]
    a = action_chunks[idx]
    chunk_action_dim = a.shape[1]

    sa_expert = torch.cat([s, a], dim=-1)
    q_expert = critic(sa_expert).mean().item()

    result = {"q_expert": q_expert}

    for ns in noise_scales:
        noisy_a = a + torch.randn(n, chunk_action_dim, device=device) * ns
        q_noisy = critic(torch.cat([s, noisy_a], dim=-1)).mean().item()
        result[f"q_noisy_{ns}"] = q_noisy
        result[f"gap_{ns}"] = q_expert - q_noisy

    q_random = critic(
        torch.cat([s, torch.rand(n, chunk_action_dim, device=device) * 2 - 1], dim=-1)
    ).mean().item()
    result["q_random"] = q_random
    result["gap_random"] = q_expert - q_random

    return result


# ═══════════════════════════════════════════════════════════════════════
# Training loop
# ═══════════════════════════════════════════════════════════════════════

def train_chunked_critic(
    states: torch.Tensor,
    action_chunks: torch.Tensor,
    targets: torch.Tensor,
    *,
    state_dim: int,
    action_dim: int,
    chunk_size: int = 3,
    hidden_dim: int = 256,
    lr: float = 3e-4,
    epochs: int = 300,
    batch_size: int = 2048,
    save_dir: Path,
    diag_every: int = 10,
    cql_alpha: float = 0.0,
    device: torch.device = torch.device("cpu"),
) -> TwinChunkedQCritic:
    """Train a TwinChunkedQCritic via MSE regression on multi-step returns."""
    N = states.shape[0]
    chunk_action_dim = chunk_size * action_dim

    critic = TwinChunkedQCritic(state_dim, action_dim, chunk_size, hidden_dim).to(device)
    optimizer = torch.optim.Adam(critic.parameters(), lr=lr)

    # Target normalisation: train in normalised space for stable gradients
    t_mean = targets.mean().item()
    t_std = targets.std().item()
    critic.set_target_stats(t_mean, t_std)
    targets_norm = (targets - t_mean) / max(t_std, 1e-8)
    print(f"\n[ChunkedCritic] Target normalisation: mean={t_mean:.2f}  std={t_std:.2f}")

    print(f"[ChunkedCritic] Network initialised on {device}")
    print(f"  input_dim = {state_dim} + {chunk_size}*{action_dim} = {state_dim + chunk_size * action_dim}")
    print(f"  params: {sum(p.numel() for p in critic.parameters()):,}")
    if cql_alpha > 0:
        print(f"  CQL regularisation: α = {cql_alpha}")

    best_loss = float("inf")
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(1, epochs + 1):
        perm = torch.randperm(N, device=device)
        epoch_loss = 0.0
        epoch_cql = 0.0
        n_batches = 0

        for i in range(0, N, batch_size):
            idx = perm[i : i + batch_size]
            bs = idx.shape[0]
            s = states[idx]
            a = action_chunks[idx]
            y = targets_norm[idx]

            sa = torch.cat([s, a], dim=-1)
            q0 = critic.q0(sa).squeeze(-1)
            q1 = critic.q1(sa).squeeze(-1)

            loss_mse = nn.functional.mse_loss(q0, y) + nn.functional.mse_loss(q1, y)

            # CQL regulariser: push down Q on random actions, push up on data actions
            if cql_alpha > 0:
                a_rand = torch.rand(bs, chunk_action_dim, device=device) * 2 - 1
                sa_rand = torch.cat([s, a_rand], dim=-1)
                q0_rand = critic.q0(sa_rand).squeeze(-1)
                q1_rand = critic.q1(sa_rand).squeeze(-1)
                # CQL penalty: E[Q(s, a_rand)] - E[Q(s, a_data)]
                cql_loss = (q0_rand.mean() - q0.detach().mean()) + \
                           (q1_rand.mean() - q1.detach().mean())
                loss = loss_mse + cql_alpha * cql_loss
                epoch_cql += cql_loss.item()
            else:
                loss = loss_mse

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            epoch_loss += loss_mse.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        avg_cql = epoch_cql / max(n_batches, 1) if cql_alpha > 0 else 0.0

        if epoch % diag_every == 0 or epoch == 1:
            # Diagnostics use forward() which denormalises automatically
            diag = compute_chunked_q_diagnostics(critic, states, action_chunks)

            cql_str = f"  cql={avg_cql:.4f}" if cql_alpha > 0 else ""
            print(
                f"[Epoch {epoch:4d}/{epochs}] loss={avg_loss:.6f}{cql_str}  |  "
                f"Q(exp)={diag['q_expert']:.2f}  "
                f"Q(n0.01)={diag['q_noisy_0.01']:.2f}  "
                f"Q(n0.05)={diag['q_noisy_0.05']:.2f}  "
                f"Q(n0.1)={diag['q_noisy_0.1']:.2f}  "
                f"Q(rand)={diag['q_random']:.2f}  "
                f"gap0.01={diag['gap_0.01']:+.3f}  "
                f"gap0.05={diag['gap_0.05']:+.3f}  "
                f"gap0.1={diag['gap_0.1']:+.3f}  "
                f"gap_rand={diag['gap_random']:+.3f}"
            )

            if avg_loss < best_loss:
                best_loss = avg_loss
                best_path = save_dir / "chunked_twin_critic_q.pt"
                torch.save(critic.state_dict(), str(best_path))
                print(f"  -> Saved best (loss={avg_loss:.6f}) to {best_path}")

    # Save final
    final_path = save_dir / "chunked_twin_critic_q_final.pt"
    torch.save(critic.state_dict(), str(final_path))
    print(f"\n[ChunkedCritic] Training complete.")
    print(f"  Best (loss={best_loss:.6f}): {save_dir / 'chunked_twin_critic_q.pt'}")
    print(f"  Final: {final_path}")

    return critic


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Distill single-step SAC critic into a chunked critic Q(s, a_t:t+k)."
    )
    p.add_argument("--task", type=str, required=True, help="Adroit task name (door, hammer, pen, relocate)")
    p.add_argument("--zarr-path", type=str, required=True, help="Path to replay buffer zarr dataset")
    p.add_argument("--sac-model", type=str, default="",
                   help="Path to SAC .zip.")
    p.add_argument("--save-dir", type=str, default="", help="Output directory for critic weights")
    p.add_argument("--chunk-size", type=int, default=3, help="Number of actions in a chunk")
    p.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--diag-every", type=int, default=10)
    p.add_argument("--cql-alpha", type=float, default=0.0,
                   help="CQL conservative regularisation weight. Pushes down Q on random "
                        "actions to prevent OOD exploitation. 0 = disabled. Try 1.0-5.0.")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    import zarr

    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[ChunkedCritic] Device: {device}")

    repo_root = Path(__file__).resolve().parents[1]
    if args.save_dir:
        save_dir = Path(args.save_dir)
    else:
        save_dir = repo_root / "runs" / "sb3_adroit_sac" / "models" / args.task
    os.makedirs(save_dir, exist_ok=True)

    zarr_path = args.zarr_path
    if not Path(zarr_path).is_absolute():
        zarr_path = str(repo_root / zarr_path)

    # Load zarr data
    root = zarr.open(str(zarr_path), mode="r")
    data = root["data"]
    meta = root["meta"]

    full_state = np.array(data["full_state"], dtype=np.float32)
    action = np.array(data["action"], dtype=np.float32)
    reward = np.array(data["reward"], dtype=np.float32)
    episode_ends = np.array(meta["episode_ends"])

    N = full_state.shape[0]
    state_dim = full_state.shape[1]
    action_dim = action.shape[1]
    print(f"[ChunkedCritic] Loaded {N} transitions: state_dim={state_dim} action_dim={action_dim}")
    print(f"  Episodes: {len(episode_ends)}")
    print(f"  Reward: mean={reward.mean():.2f} std={reward.std():.2f}")

    # Compute V(s): either from SAC model or from precomputed zarr values
    if args.sac_model:
        _ensure_sb3_importable(repo_root)
        from stable_baselines3 import SAC
        sac_path = args.sac_model if Path(args.sac_model).is_absolute() else str(repo_root / args.sac_model)
        print(f"[ChunkedCritic] Loading SAC model from {sac_path} ...")
        sac_model = SAC.load(sac_path, device=device)
        print(f"[ChunkedCritic] Computing V(s) = Q(s, π(s)) from SAC critic ...")
        v_value = compute_v_from_sac(sac_model, full_state)
        del sac_model  # free memory
        print(f"  V-value (from SAC): mean={v_value.mean():.2f} std={v_value.std():.2f}")
    else:
        v_value = np.array(data["v_value"], dtype=np.float32)
        print(f"  V-value (from zarr): mean={v_value.mean():.2f} std={v_value.std():.2f}")

    # Compute multi-step return targets
    states_np, chunks_np, targets_np = compute_multistep_targets(
        full_state, action, reward, v_value, episode_ends,
        gamma=args.gamma, chunk_size=args.chunk_size,
    )

    states_t = torch.tensor(states_np, dtype=torch.float32, device=device)
    chunks_t = torch.tensor(chunks_np, dtype=torch.float32, device=device)
    targets_t = torch.tensor(targets_np, dtype=torch.float32, device=device)

    print(f"\n[ChunkedCritic] Config:")
    print(f"  task={args.task}  chunk_size={args.chunk_size}  gamma={args.gamma}")
    print(f"  epochs={args.epochs}  batch_size={args.batch_size}  hidden_dim={args.hidden_dim}")
    print(f"  lr={args.lr}  save_dir={save_dir}")

    train_chunked_critic(
        states_t, chunks_t, targets_t,
        state_dim=state_dim,
        action_dim=action_dim,
        chunk_size=args.chunk_size,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        save_dir=save_dir,
        diag_every=args.diag_every,
        device=device,
        cql_alpha=args.cql_alpha,
    )


if __name__ == "__main__":
    main()
