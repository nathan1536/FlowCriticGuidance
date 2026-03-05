from typing import Dict, List
import numpy as np
import copy
from pathlib import Path

from termcolor import cprint

from maniflow.common.replay_buffer import ReplayBuffer
from maniflow.common.sampler import SequenceSampler, get_val_mask, downsample_mask
from maniflow.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from maniflow.dataset.base_dataset import BaseDataset


class MetaworldMultitaskImageDataset(BaseDataset):
    """
    2D (RGB) multitask dataset for MetaWorld.

    For each task, loads keys:
    - state (-> agent_pos)
    - action
    - img (-> image)

    Returns per-sample:
      {
        "obs": {
          "image": (T, ...),
          "agent_pos": (T, D),
          "task_id": (num_tasks,),
          "task_name": <str>
        },
        "action": (T, Da)
      }
    """

    def __init__(
        self,
        data_path: str,
        task_names: List[str],
        horizon=1,
        pad_before=0,
        pad_after=0,
        seed=42,
        val_ratio=0.0,
        max_train_episodes=None,
        zarr_name_template: str = "metaworld_{task_name}_expert.zarr",
        use_depth: bool = False,
    ):
        super().__init__()

        self.task_names = list(task_names)
        self.replay_buffers = {}
        self.samplers = {}
        self.train_masks = {}
        self.train_episodes_num = 0
        self.val_episodes_num = 0
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.data_path = data_path
        self.zarr_name_template = zarr_name_template
        self.use_depth = use_depth

        # Load data for each task
        for task_name in self.task_names:
            zarr_path = Path(data_path) / zarr_name_template.format(task_name=task_name)

            buffer_keys = ["state", "action", "img", "full_state"]
            if self.use_depth:
                buffer_keys.append("depth")

            rb = ReplayBuffer.copy_from_path(zarr_path, keys=buffer_keys)

            val_mask = get_val_mask(
                n_episodes=rb.n_episodes,
                val_ratio=val_ratio,
                seed=seed,
            )
            train_mask = ~val_mask
            train_mask = downsample_mask(
                mask=train_mask,
                max_n=max_train_episodes,
                seed=seed,
            )

            sampler = SequenceSampler(
                replay_buffer=rb,
                sequence_length=horizon,
                pad_before=pad_before,
                pad_after=pad_after,
                episode_mask=train_mask,
            )

            self.replay_buffers[task_name] = rb
            self.samplers[task_name] = sampler
            self.train_masks[task_name] = train_mask
            self.train_episodes_num += int(np.sum(train_mask))
            self.val_episodes_num += int(np.sum(val_mask))

            cprint(f"Task {task_name}: {int(np.sum(train_mask))} training episodes and {len(sampler)} rollout steps", "green")

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.samplers = {}
        for task_name in self.task_names:
            val_set.samplers[task_name] = SequenceSampler(
                replay_buffer=self.replay_buffers[task_name],
                sequence_length=self.horizon,
                pad_before=self.pad_before,
                pad_after=self.pad_after,
                episode_mask=~self.train_masks[task_name],
            )
        return val_set

    def get_normalizer(self, mode="limits", **kwargs):
        # Normalize action; keep image/agent_pos identity (like other image datasets).
        all_actions = []
        for task_name in self.task_names:
            rb = self.replay_buffers[task_name]
            all_actions.append(rb["action"])

        data = {"action": np.concatenate(all_actions)}
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)

        normalizer["image"] = SingleFieldLinearNormalizer.create_identity()
        if self.use_depth:
            normalizer["depth"] = SingleFieldLinearNormalizer.create_identity()
        normalizer["agent_pos"] = SingleFieldLinearNormalizer.create_identity()
        # task_id is already one-hot / bounded
        normalizer["task_id"] = SingleFieldLinearNormalizer.create_identity()
        return normalizer

    def __len__(self) -> int:
        return sum(len(s) for s in self.samplers.values())

    def _sample_to_data(self, sample, task_name: str):
        agent_pos = sample["state"][:,].astype(np.float32)
        image = sample["img"][:,].astype(np.float32)
        full_state = sample["full_state"][:,].astype(np.float32)

        task_idx = self.task_names.index(task_name)
        task_one_hot = np.zeros(len(self.task_names), dtype=np.float32)
        task_one_hot[task_idx] = 1.0

        data = {
            "obs": {
                "image": image,
                "agent_pos": agent_pos,
                "full_state": full_state,
                "task_id": task_one_hot,
                "task_name": task_name,
            },
            "action": sample["action"].astype(np.float32),
        }
        if self.use_depth:
            data["obs"]["depth"] = sample["depth"][:,].astype(np.float32)
        return data

    def __getitem__(self, idx):
        # Determine which task this index belongs to
        current_idx = idx
        for task_name, sampler in self.samplers.items():
            if current_idx < len(sampler):
                sample = sampler.sample_sequence(current_idx)
                return self._sample_to_data(sample, task_name)
            current_idx -= len(sampler)
        raise IndexError("Index out of range")


