"""
DRQ-v2 style asymmetric FlowQL:
  Actor  — DRQ-v2 CNN encoder + RandomShiftsAug + MLP flow model (image-based)
  Critic — DRQ-v2 trunk + Q1/Q2 MLP on full_state (state-based, no CNN)

Reference architecture: drqv2/drqv2.py (Yarats et al. 2021)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

from maniflow.common.pytorch_util import dict_apply
from maniflow.policy.maniflow_state_policy import ManiFlowStatePolicy


# ─────────────────────────────────────────────────────────────────────────────
# Augmentation  (exact DRQ-v2 implementation)
# ─────────────────────────────────────────────────────────────────────────────

class RandomShiftsAug(nn.Module):
    """Pad image by `pad` pixels (replicate boundary), then random bilinear crop."""
    def __init__(self, pad: int = 4):
        super().__init__()
        self.pad = pad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, C, H, W)
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps, 1.0 - eps,
                                h + 2 * self.pad, device=x.device, dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)   # (n, h, h, 2)
        shift = torch.randint(0, 2 * self.pad + 1, size=(n, 1, 1, 2),
                              device=x.device, dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)
        return F.grid_sample(x, base_grid + shift,
                             padding_mode='zeros', align_corners=False)


# ─────────────────────────────────────────────────────────────────────────────
# Actor encoder  (DRQ-v2 CNN + trunk)
# ─────────────────────────────────────────────────────────────────────────────

class DrQv2Encoder(nn.Module):
    """
    DRQ-v2 CNN encoder — Conv×4 only, matching drqv2.py Encoder exactly.
    Input : (B, C, H, W) float32, pixel values 0-255
    Output: (B, repr_dim)   e.g. repr_dim = 39200 for 84×84 input

    The trunk (Linear→LayerNorm→Tanh) lives in DrQv2FlowPolicy.obs_encoder,
    matching DRQ-v2 where trunk is part of Actor/Critic, not the Encoder.
    """
    def __init__(self, obs_shape: tuple):
        super().__init__()
        assert len(obs_shape) == 3, "obs_shape must be (C, H, W)"

        self.convnet = nn.Sequential(
            nn.Conv2d(obs_shape[0], 32, 3, stride=2), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1), nn.ReLU(),
        )
        with torch.no_grad():
            self.repr_dim = self.convnet(torch.zeros(1, *obs_shape)).view(1, -1).shape[1]

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        obs = obs / 255.0 - 0.5
        return self.convnet(obs).view(obs.shape[0], -1)  # (B, repr_dim)


# ─────────────────────────────────────────────────────────────────────────────
# Critic  (DRQ-v2 trunk + Q1/Q2 on full_state — no CNN)
# ─────────────────────────────────────────────────────────────────────────────

# class DrQv2StateCritic(nn.Module):
#     """
#     Asymmetric critic: Q(full_state, action) — no CNN, no trunk.
#     state_dim is already small (e.g. 39) and normalized, so the DRQ-v2
#     trunk (which exists to compress repr_dim=39200) is unnecessary here.

#       Q1/Q2 : Linear(state_dim + action_dim, hidden) → ReLU
#                → Linear(hidden, hidden) → ReLU → Linear(hidden, 1)
#     """
#     def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256, **kwargs):
#         super().__init__()
#         def _q_net():
#             return nn.Sequential(
#                 nn.Linear(state_dim + action_dim, hidden_dim), nn.ReLU(inplace=True),
#                 nn.Linear(hidden_dim, hidden_dim),             nn.ReLU(inplace=True),
#                 nn.Linear(hidden_dim, hidden_dim),             nn.ReLU(inplace=True),
#                 nn.Linear(hidden_dim, 1),
#             )
#         self.Q1 = _q_net()
#         self.Q2 = _q_net()
#         self._init_weights()

#     def _init_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.orthogonal_(m.weight)
#                 nn.init.zeros_(m.bias)

#     def forward(self, state: torch.Tensor, action: torch.Tensor):
#         x = torch.cat([state, action], dim=-1)
#         return self.Q1(x), self.Q2(x)

#     def q_min(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
#         q1, q2 = self.forward(state, action)
#         return torch.min(q1, q2)


# ─────────────────────────────────────────────────────────────────────────────
# Policy  (DRQ-v2 CNN actor + flow MLP, inherits ODE logic from state policy)
# ─────────────────────────────────────────────────────────────────────────────

class DrQv2FlowPolicy(ManiFlowStatePolicy):
    """
    Asymmetric FlowQL policy:
      - obs_encoder : aug + DRQ-v2 CNN → feature_dim  (image-based)
      - flow model  : FlowMLP conditioned on feature_dim (inherited)
      - critic      : DrQv2StateCritic on full_state (set up by workspace)

    All ODE / flow methods (compute_loss, compute_flowql_loss,
    sample_ode, few_step_sample_for_training) are inherited from
    ManiFlowStatePolicy unchanged.
    """

    def __init__(self,
                 obs_shape: tuple,        # (C, H, W), e.g. (3, 84, 84)
                 feature_dim: int = 256,  # CNN output = FlowMLP cond dim
                 aug_pad: int = 4,
                 **state_policy_kwargs):
        # FlowMLP is conditioned on `feature_dim`-dim image features
        super().__init__(state_dim=feature_dim, **state_policy_kwargs)
        self.cnn_encoder = DrQv2Encoder(obs_shape)
        # Trunk lives in the actor (matching drqv2.py Actor.trunk, not inside Encoder)
        self.trunk = nn.Sequential(
            nn.Linear(self.cnn_encoder.repr_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.Tanh(),
        )
        self.aug         = RandomShiftsAug(pad=aug_pad)
        self.obs_shape   = obs_shape
        self.is_image_policy = True   # signals workspace: use image obs path

    # ── encoder ──────────────────────────────────────────────────────────────

    def obs_encoder(self, nobs: dict) -> torch.Tensor:
        """
        nobs['image'] : (B, T, H, W, C) float32, 0-255
        returns       : (B, 1, feature_dim)  — T frames stacked along channel dim
        """
        img = nobs['image']                                       # (B, T, H, W, C) or (B, T, C, H, W)
        B, T = img.shape[:2]
        ### Single frame obs ###
        # img = img.reshape(B * T, *img.shape[2:])
        # # zarr dataset: (H, W, C) → permute to (C, H, W)
        # # env runner:   already (C, H, W) → no permute needed
        # if img.shape[-1] == self.obs_shape[0]:                    # channel-last (H, W, C)
        #     img = img.permute(0, 3, 1, 2).contiguous()
        # else:                                                      # already channel-first (C, H, W)
        #     img = img.contiguous()
        ### Single frame obs ###


        per_frame_C = self.obs_shape[0] // T                     # e.g. 9 // 3 = 3

        # Convert to channel-first per frame: (B, T, C, H, W)
        if img.shape[-1] == per_frame_C:                          # channel-last (B, T, H, W, C)
            img = img.permute(0, 1, 4, 2, 3).contiguous()
        else:                                                      # already channel-first (B, T, C, H, W)
            img = img.contiguous()

        H, W = img.shape[3], img.shape[4]
        img = img.reshape(B, T * per_frame_C, H, W)              # (B, T*C, H, W)

        if self.training:
            img = self.aug(img)                                   # same spatial shift for all frames
        h    = self.cnn_encoder(img)                              # (B, repr_dim)
        feat = self.trunk(h)                                      # (B, feature_dim)
        return feat.unsqueeze(1)                                  # (B, 1, feature_dim)

    # ── inference ─────────────────────────────────────────────────────────────

    # def sample_action(self, image: 'np.ndarray', state: 'np.ndarray',
    #                   critic, n_candidates: int = 50) -> 'np.ndarray':
    #     """Q-guided action selection for image-based policy.

    #     Args:
    #         image: (T, H, W, C) uint8 numpy stacked frames (T = n_obs_steps)
    #         state: (state_dim,) float numpy full state — passed to critic only
    #         critic: with .q_min(state, action) -> (B, 1)
    #     Returns:
    #         (action_dim,) numpy action
    #     """
    #     import numpy as np
    #     device = self.device

    #     state_t  = torch.FloatTensor(state.reshape(1, -1)).to(device)
    #     state_rpt = state_t.repeat(n_candidates, 1)                   # (N, state_dim)

    #     img_t = torch.FloatTensor(image).to(device)                   # (T, H, W, C)
    #     img_t = img_t.unsqueeze(0)                                    # (1, T, H, W, C)
    #     img_t = img_t.expand(n_candidates, -1, -1, -1, -1).contiguous()  # (N, T, H, W, C)

    #     nobs = self.normalizer.normalize({'image': img_t})
    #     this_nobs = dict_apply(nobs, lambda x: x[:, :self.n_obs_steps, ...].to(device))
    #     vis_cond = self.obs_encoder(this_nobs)                        # (N, T, feature_dim)
    #     vis_cond = vis_cond.reshape(n_candidates, -1, self.obs_feature_dim)

    #     noise = torch.randn(n_candidates, self.horizon, self.action_dim, device=device)

    #     with torch.no_grad():
    #         traj = self.sample_ode(x0=noise, N=self.num_inference_steps, vis_cond=vis_cond)
    #         actions_norm = traj[-1]

    #         actions_raw = self.normalizer['action'].unnormalize(actions_norm)
    #         exec_start = self.n_obs_steps - 1
    #         a0 = actions_raw[:, exec_start] if actions_raw.dim() == 3 else actions_raw
    #         a0 = a0.clamp(-1.0, 1.0)

    #         q_value = critic.q_min(state_rpt, a0).flatten()          # (N,)
    #         idx = torch.multinomial(F.softmax(q_value, dim=0), 1)

    #     return a0[idx].cpu().numpy().flatten()

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> dict:
        """obs_dict['image'] : (B, T, H, W, C)"""
        nobs = self.normalizer.normalize(obs_dict)
        B, To = nobs['image'].shape[0], self.n_obs_steps
        device, dtype = self.device, self.dtype

        this_nobs = dict_apply(nobs, lambda x: x[:, :To, ...].to(device))
        vis_cond = self.obs_encoder(this_nobs)                    # (B, T, feature_dim)
        vis_cond = vis_cond.reshape(B, -1, self.obs_feature_dim)

        noise = torch.randn(B, self.horizon, self.action_dim, device=device, dtype=dtype)
        traj  = self.sample_ode(x0=noise, N=self.num_inference_steps, vis_cond=vis_cond)
        nsample = traj[-1]

        action_pred = self.normalizer['action'].unnormalize(nsample[..., :self.action_dim])

        ### Single frame action ###
        # start  = To - 1
        # action = action_pred[:, start: start + self.n_action_steps]
        ### Single frame action ###


        # start  = min(To - 1, self.horizon - 1)  # clamp: horizon=1 → start=0
        # action = action_pred[:, start: start + self.n_action_steps]
        action = action_pred[:, 0:1]

        return {'action': action, 'action_pred': action_pred}
