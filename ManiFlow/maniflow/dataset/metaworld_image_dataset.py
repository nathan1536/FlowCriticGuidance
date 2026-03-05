from typing import Dict
import torch
import numpy as np
import copy

from termcolor import cprint

from maniflow.common.pytorch_util import dict_apply
from maniflow.common.replay_buffer import ReplayBuffer
from maniflow.common.sampler import SequenceSampler, get_val_mask, downsample_mask
from maniflow.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from maniflow.dataset.base_dataset import BaseDataset


class MetaworldImageDataset(BaseDataset):
    """
    2D (RGB) version of MetaWorld dataset.

    Expected Zarr keys (under data/):
    - state: low-dim robot state (used as agent_pos)
    - action
    - img: RGB frames (uint8 or float32). Channel-last or channel-first is OK.

    Output sample structure matches other ManiFlow image datasets:
      {
        "obs": { "image": (T, ...), "agent_pos": (T, D) },
        "action": (T, Da)
      }
    """

    def __init__(
        self,
        zarr_path,
        horizon=1,
        pad_before=0,
        pad_after=0,
        seed=42,
        val_ratio=0.0,
        max_train_episodes=None,
        use_img: bool = True,
        use_depth: bool = False,
    ):
        super().__init__()
        cprint(f"Loading MetaworldImageDataset from {zarr_path}", "green")
        self.use_img = use_img
        self.use_depth = use_depth

        buffer_keys = ["state", "action"]
        if self.use_img:
            buffer_keys.append("img")
        if self.use_depth:
            buffer_keys.append("depth")

        self.replay_buffer = ReplayBuffer.copy_from_path(zarr_path, keys=buffer_keys)

        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed,
        )
        train_mask = ~val_mask
        train_mask = downsample_mask(mask=train_mask, max_n=max_train_episodes, seed=seed)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
        )
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.zarr_path = zarr_path
        self.train_episodes_num = np.sum(train_mask)
        self.val_episodes_num = np.sum(val_mask)

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask,
        )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode="limits", **kwargs):
        # Match AdroitImageDataset convention: normalize action; keep image/agent_pos identity.
        data = {"action": self.replay_buffer["action"]}
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)

        if self.use_img:
            normalizer["image"] = SingleFieldLinearNormalizer.create_identity()
        if self.use_depth:
            normalizer["depth"] = SingleFieldLinearNormalizer.create_identity()
        normalizer["agent_pos"] = SingleFieldLinearNormalizer.create_identity()
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        agent_pos = sample["state"][:,].astype(np.float32)
        data = {
            "obs": {
                "agent_pos": agent_pos,
            },
            "action": sample["action"].astype(np.float32),
        }
        if self.use_img:
            # Keep raw range; vision encoders normalize by /255 if needed.
            data["obs"]["image"] = sample["img"][:,].astype(np.float32)
        if self.use_depth:
            data["obs"]["depth"] = sample["depth"][:,].astype(np.float32)
        return data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        to_torch = lambda x: torch.from_numpy(x) if x.__class__.__name__ == "ndarray" else x
        return dict_apply(data, to_torch)


