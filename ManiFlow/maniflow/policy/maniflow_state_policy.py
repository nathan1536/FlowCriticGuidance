"""
MLP-based flow matching policy for state observations ).

"""

from typing import Dict, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import reduce
from termcolor import cprint
import math

from maniflow.model.common.normalizer import LinearNormalizer
from maniflow.policy.base_policy import BasePolicy
from maniflow.common.pytorch_util import dict_apply


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class FlowMLP(nn.Module):
    """
    MLP 

    (action_t, t, target_t, state) 
    """

    def __init__(self, state_dim, action_dim, hidden_dim=256, t_dim=16, num_layers=3):
        super().__init__()

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim * 2),
            nn.Mish(),
            nn.Linear(t_dim * 2, t_dim),
        )

        input_dim = action_dim + state_dim + t_dim
        self.mid_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
        )
        self.final_layer = nn.Linear(hidden_dim, action_dim)

    def forward(self, sample, timestep, state_cond, **kwargs):
        """
        Args:
            sample: (B, action_dim) noisy action at time t
            timestep: (B,) flow time
            state_cond: (B, state_dim) state condition
        Returns:
            (B, action_dim)
        """
        t_emb = self.time_mlp(timestep)
        x = torch.cat([sample, t_emb, state_cond], dim=-1)
        x = self.mid_layer(x)
        return self.final_layer(x)


class ManiFlowStatePolicy(BasePolicy):
    """
    MLP flow matching policy conditioned on full_state 

    """

    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_dim: int = 256,
                 t_dim: int = 16,
                 num_layers: int = 3,
                 horizon: int = 1,
                 n_action_steps: int = 1,
                 n_obs_steps: int = 1,
                 num_inference_steps: int = 10,
                 sample_t_mode_flow: str = "beta",
                 **kwargs):
        super().__init__()

        self.model = FlowMLP(state_dim, action_dim, hidden_dim, t_dim, num_layers)

        self.normalizer = LinearNormalizer()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.horizon = horizon
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.num_inference_steps = num_inference_steps
        self.sample_t_mode_flow = sample_t_mode_flow

        # obs_feature_dim is used by the workspace for vis_cond reshape
        # For state policy, "vis_cond" is just the state itself
        self.obs_feature_dim = state_dim

        n_params = sum(p.numel() for p in self.parameters())
        cprint(f"[ManiFlowStatePolicy] state_dim={state_dim}, action_dim={action_dim}, "
               f"hidden_dim={hidden_dim}, params={n_params:,}", "yellow")

    # ========= inference ============

    @torch.no_grad()
    def sample_ode(self, x0, N=None, vis_cond=None, **kwargs):
        """Euler ODE sampler. vis_cond is (B, state_dim) for state policy."""
        if N is None:
            N = self.num_inference_steps
        dt = 1.0 / N
        traj = []
        x = x0.detach().clone()
        batch_size = x.shape[0]

        # vis_cond shape: (B, 1, state_dim) from workspace -> squeeze to (B, state_dim)
        state_cond = vis_cond.squeeze(1) if vis_cond.dim() == 3 else vis_cond

        traj.append(x.detach().clone())
        for i in range(N):
            t = torch.ones(batch_size, device=self.device) * (i / N)

            # x is (B, horizon, action_dim), squeeze for MLP if horizon=1
            x_flat = x.squeeze(1) if x.dim() == 3 else x
            pred = self.model(x_flat, t, state_cond=state_cond)
            # Expand back if needed
            if x.dim() == 3:
                pred = pred.unsqueeze(1)
            x = x.detach().clone() + pred * dt
            traj.append(x.detach().clone())

        return traj

    # def sample_action(self, state: np.ndarray, critic, n_candidates: int = 50) -> np.ndarray:
    #     """Q-guided action selection: sample n_candidates actions, pick best by softmax Q.

    #     Mirrors DiffusionQL's sample_action for drop-in compatibility.
    #     Args:
    #         state: (state_dim,) raw unnormalized numpy state
    #         critic: TwinQCritic (or any object with .q_min(state, action) -> (B,1))
    #         n_candidates: number of candidate actions to sample
    #     Returns:
    #         (action_dim,) numpy action
    #     """
    #     device = self.device
    #     state_t = torch.FloatTensor(state.reshape(1, -1)).to(device)
    #     state_rpt = state_t.repeat(n_candidates, 1)  # (N, state_dim)

    #     # Normalize state for flow conditioning
    #     obs_dict = {'full_state': state_rpt.unsqueeze(1)}  # (N, 1, state_dim)
    #     nobs = self.normalizer.normalize(obs_dict)
    #     state_cond = nobs['full_state'].squeeze(1)  # (N, state_dim)

    #     noise = torch.randn(n_candidates, self.horizon, self.action_dim, device=device)

    #     with torch.no_grad():
    #         traj = self.sample_ode(x0=noise, N=self.num_inference_steps, vis_cond=state_cond)
    #         actions_norm = traj[-1]  # (N, horizon, action_dim) or (N, action_dim)

    #         actions_raw = self.normalizer['action'].unnormalize(actions_norm)
    #         exec_start = self.n_obs_steps - 1
    #         a0 = actions_raw[:, exec_start] if actions_raw.dim() == 3 else actions_raw
    #         a0 = a0.clamp(-1.0, 1.0)

    #         q_value = critic.q_min(state_rpt, a0).flatten()  # (N,)
    #         idx = torch.multinomial(F.softmax(q_value, dim=0), 1)

    #     return a0[idx].cpu().numpy().flatten()

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Standard predict_action interface. Uses full_state as condition."""
        nobs = self.normalizer.normalize(obs_dict)
        device = self.device
        dtype = self.dtype

        state = nobs['full_state'][:, :self.n_obs_steps].to(device)  # (B, T, state_dim)
        B = state.shape[0]
        state_cond = state.reshape(B, -1)  # (B, n_obs_steps * state_dim)

        # Empty action noise
        noise = torch.randn(B, self.horizon, self.action_dim, device=device, dtype=dtype)

        # ODE sample
        traj = self.sample_ode(x0=noise, N=self.num_inference_steps, vis_cond=state_cond)
        nsample = traj[-1]

        # Unnormalize
        action_pred = self.normalizer['action'].unnormalize(nsample[..., :self.action_dim])

        start = self.n_obs_steps - 1
        end = start + self.n_action_steps
        action = action_pred[:, start:end]

        return {
            'action': action,
            'action_pred': action_pred,
        }

    # ========= training ============

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def obs_encoder(self, nobs):
        """
        'obs_encoder' for state policy: just returns the full_state.
        This lets the workspace code call self.model.obs_encoder(this_nobs)
        and get back a tensor of shape (B, n_obs_steps, state_dim).
        """
        return nobs['full_state']

    def sample_t(self, batch_size, mode="uniform"):
        if mode == "uniform":
            t = torch.rand((batch_size,), device=self.device)
        elif mode == "beta":
            # Beta(2, 5) distribution, biased toward small t
            dist = torch.distributions.Beta(2.0, 5.0)
            t = dist.sample((batch_size,)).to(self.device)
        else:
            raise ValueError(f"Unsupported sample_t_mode {mode}")
        return t

    def few_step_sample_for_training(self, noise, vis_cond, lang_cond=None, num_steps=4):
        """Few Euler steps with gradient flow for QL loss. """
        x = noise
        dt = 1.0 / num_steps
        batch_size = x.shape[0]

        state_cond = vis_cond.squeeze(1) if vis_cond.dim() == 3 else vis_cond

        for i in range(num_steps):
            t = torch.ones(batch_size, device=self.device) * (i / num_steps)

            x_flat = x.squeeze(1) if x.dim() == 3 else x
            v = self.model(x_flat, t, state_cond=state_cond)
            x_flat = x_flat + v * dt

            if x.dim() == 3:
                x = x_flat.unsqueeze(1)
            else:
                x = x_flat

        return x

    def compute_loss(self, batch, ema_model=None, vis_cond=None, **kwargs):
        """Flow matching loss (BC loss) for state policy."""
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action']).to(self.device)

        batch_size = nactions.shape[0]

        # Get state condition
        if vis_cond is None:
            state = dict_apply(nobs, lambda x: x[:, :self.n_obs_steps, ...].to(self.device))
            vis_cond = self.obs_encoder(state)
            vis_cond = vis_cond.reshape(batch_size, -1, self.obs_feature_dim)

        state_cond = vis_cond.squeeze(1) if vis_cond.dim() == 3 else vis_cond  # (B, state_dim)

        # Flow matching: sample t, interpolate, predict velocity
        actions_flat = nactions.squeeze(1) if nactions.dim() == 3 and self.horizon == 1 else nactions

        t = self.sample_t(batch_size, mode=self.sample_t_mode_flow).to(self.device)
        t_expand = t.view(-1, 1)

        x0 = torch.randn_like(actions_flat, device=self.device)
        x1 = actions_flat.to(self.device)

        # Linear interpolation: x_t = (1-t) * x0 + t * x1
        x_t = (1.0 - t_expand) * x0 + t_expand * x1
        # Target velocity: v = x1 - x0
        v_target = x1 - x0

        v_pred = self.model(x_t, t, state_cond=state_cond)

        loss = F.mse_loss(v_pred, v_target)

        loss_dict = {
            'loss_flow': loss.item(),
            'loss_ct': 0.0,
            'v_flow_pred_magnitude': torch.sqrt(torch.mean(v_pred ** 2)).item(),
            'v_ct_pred_magnitude': 0.0,
            'bc_loss': loss.item(),
        }

        return loss, loss_dict

    def compute_flowql_loss(
            self,
            batch,
            vis_cond,
            lang_cond,
            critic,
            num_steps: int = 4,
            critic_states=None,
    ):
        """DiffusionQL-style Q-loss: L_q = -Q(s, π(s)) / E|Q|"""
        batch_size = vis_cond.shape[0]
        device = self.device

        noise = torch.randn(
            batch_size, self.horizon, self.action_dim,
            device=device, dtype=vis_cond.dtype,
        )
        sampled_actions_norm = self.few_step_sample_for_training(
            noise, vis_cond, lang_cond, num_steps=num_steps,
        )

        action_raw = self.normalizer['action'].unnormalize(sampled_actions_norm)
        action_raw = action_raw.clamp(-1.0, 1.0)
        a0 = action_raw[:, 0]  # (B, action_dim) — horizon=1, always index 0

        if critic_states is not None:
            states = critic_states  # image embedding (B, obs_feature_dim)
        else:
            states = batch['obs']['full_state'][:, 0].to(device)  # (B, state_dim)

        q0, q1 = critic(states, a0)

        # if torch.rand(1).item() > 0.5:
        #     q_loss = -q0.mean() / q1.abs().mean().detach() 
        # else:
        #     q_loss = -q1.mean() / q0.abs().mean().detach()

        q_min = torch.min(q0, q1)
        q_loss = -q_min.mean() / q_min.abs().mean().detach()


        with torch.no_grad():
            q_mean = torch.min(q0, q1).mean().item()
            q_std = torch.min(q0, q1).std().item()

        return q_loss, {
            'flowql_q_mean': q_mean,
            'flowql_q_std': q_std,
            'flowql_q_loss': q_loss.item(),
        }
