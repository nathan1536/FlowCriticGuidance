"""
MCR (ResNet50) + FlowMLP policy.

Training  : obs_encoder() reads precomputed `img_embedding` (2048-dim) from the
            dataset — no CNN forward pass, maximally fast.
Inference : predict_action() accepts raw images + agent_pos, encodes images
            on-the-fly with the frozen MCR ResNet50, then runs the ODE sampler.

Critic uses full_state (39-dim) — asymmetric design preserved.
"""

import sys
import pathlib
import torch
import torch.nn as nn
import torchvision.transforms as T
import numpy as np
from PIL import Image
from typing import Dict

from maniflow.policy.maniflow_state_policy import ManiFlowStatePolicy

# MCR preprocessing — must match precompute_mcr_embeddings.py
MCR_TRANSFORMS = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),   # divides by 255
])

EMBEDDING_DIM = 2048   # MCR ResNet50 output dim


class MCRFlowPolicy(ManiFlowStatePolicy):
    """
    Flow matching policy conditioned on MCR image embeddings + agent_pos.

    - Training : fast — embeddings come from dataset, no encoder forward pass.
    - Inference: frozen MCR encodes raw images on-the-fly.
    - Critic   : still uses full_state (39-dim), not embeddings → asymmetric.
    """

    is_embedding_policy = True

    def __init__(self,
                 mcr_ckpt_path: str,
                 embedding_dim: int = EMBEDDING_DIM,
                #  agent_pos_dim: int = 9,
                 **state_policy_kwargs):
        # FlowMLP conditioned on embedding + agent_pos
        # super().__init__(state_dim=embedding_dim + agent_pos_dim, **state_policy_kwargs)
        super().__init__(state_dim=embedding_dim, **state_policy_kwargs)
        self.embedding_dim = embedding_dim
        # self.agent_pos_dim = agent_pos_dim
        # self.embedding_bn = nn.BatchNorm1d(embedding_dim)

        # Load frozen MCR
        MCR_DIR = str(pathlib.Path(__file__).parent.parent.parent.parent
                      / 'robots-pretrain-robots')
        if MCR_DIR not in sys.path:
            sys.path.insert(0, MCR_DIR)
        from mcr import load_mcr
        self.mcr = load_mcr(ckpt_path=mcr_ckpt_path)
        for p in self.mcr.parameters():
            p.requires_grad = False
        self.mcr.eval()

    # ── encoder ───────────────────────────────────────────────

    def obs_encoder(self, nobs: dict) -> torch.Tensor:
        """
        nobs['img_embedding'] : (B, T, 2048)
        nobs['agent_pos']     : (B, T, agent_pos_dim)
        returns               : (B, T, 2048 + agent_pos_dim)
        """
        emb = nobs['img_embedding']                       # (B, T, 2048)
        # B, T, D = emb.shape
        # emb = self.embedding_bn(emb.reshape(B * T, D)).reshape(B, T, D)
        # pos = nobs['agent_pos'].to(emb.device)            # (B, T, agent_pos_dim)
        return emb
        # return torch.cat([emb, pos], dim=-1)              # (B, T, 2048 + agent_pos_dim)

    # ── inference ─────────────────────────────────────────────────────────────

    def _encode_images(self, img: torch.Tensor) -> torch.Tensor:
        """
        img     : (B, T, H, W, C) or (B, T, C, H, W), uint8 or float32
        returns : (B, T, 2048)
        """
        B, T = img.shape[:2]
        img_np = img.cpu().numpy()

        # Ensure (B*T, H, W, C) uint8 for PIL
        flat = img_np.reshape(B * T, *img_np.shape[2:])
        if flat.dtype != np.uint8:
            if flat.max() <= 1.0:
                flat = flat * 255.0
            flat = flat.clip(0, 255).astype(np.uint8)
        # Channel-first → channel-last if needed
        if flat.shape[1] == 3:       # (B*T, C, H, W) → (B*T, H, W, C)
            flat = flat.transpose(0, 2, 3, 1)

        tensors = [MCR_TRANSFORMS(Image.fromarray(f)) for f in flat]
        x = torch.stack(tensors).to(self.device)   # (B*T, 3, 224, 224)

        self.mcr.to(self.device).eval()
        with torch.no_grad():
            emb = self.mcr(x)                      # (B*T, 2048)
        return emb.view(B, T, self.embedding_dim)  # (B, T, 2048)

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> dict:
        """
        obs_dict must contain ('image' OR 'img_embedding') AND 'agent_pos'.
        """
        device, dtype = self.device, self.dtype

        if 'img_embedding' in obs_dict:
            embedding = obs_dict['img_embedding'][:, :self.n_obs_steps].to(device)
            embedding = self.normalizer['img_embedding'].normalize(embedding)
        else:
            img = obs_dict['image'][:, :self.n_obs_steps]   # (B, T, H, W, C)
            embedding = self._encode_images(img)             # (B, T, 2048)
            embedding = self.normalizer['img_embedding'].normalize(embedding)

        B, T, D = embedding.shape
        # embedding = self.embedding_bn(embedding.reshape(B * T, D)).reshape(B, T, D)

        # agent_pos = obs_dict['agent_pos'][:, :self.n_obs_steps].to(device)
        # agent_pos = self.normalizer['agent_pos'].normalize(agent_pos)

        # combined = torch.cat([embedding, agent_pos], dim=-1)  # (B, T, 2048+agent_pos_dim)
        # vis_cond = combined.reshape(B, -1, self.obs_feature_dim)

        vis_cond = embedding.reshape(B, -1, self.embedding_dim)  # (B, T, 2048)

        noise = torch.randn(B, self.horizon, self.action_dim, device=device, dtype=dtype)
        traj  = self.sample_ode(x0=noise, N=self.num_inference_steps, vis_cond=vis_cond)
        nsample = traj[-1]

        action_pred = self.normalizer['action'].unnormalize(nsample[..., :self.action_dim])
        start  = self.n_obs_steps - 1
        action = action_pred[:, start: start + self.n_action_steps]
        return {'action': action, 'action_pred': action_pred}
