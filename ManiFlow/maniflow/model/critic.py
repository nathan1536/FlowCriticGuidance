"""
Twin Q-Critic for FlowQL (DiffusionQL-style offline RL).

Architecture follows Wang et al. (ICLR 2023) "Diffusion Policies as an
Expressive Policy Class for Offline Reinforcement Learning".
"""

import torch
import torch.nn as nn


class TwinQCritic(nn.Module):
    """Twin Q-networks: two independent Q(s, a) -> scalar estimators."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.q0 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, 1),
        )
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor):
        """Returns (q0, q1) each of shape (B, 1)."""
        x = torch.cat([state, action], dim=-1)
        return self.q0(x), self.q1(x)

    def q_min(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Conservative Q estimate: min(Q0, Q1)."""
        q0, q1 = self.forward(state, action)
        return torch.min(q0, q1)
