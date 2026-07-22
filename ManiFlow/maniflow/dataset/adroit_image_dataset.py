from typing import Dict
import torch
import numpy as np
import copy
import zarr
import numcodecs
numcodecs.blosc.use_threads = False  # <--- ADD THIS to fix CPU gridlock
from maniflow.common.pytorch_util import dict_apply
from maniflow.common.replay_buffer import ReplayBuffer
from maniflow.common.sampler import (SequenceSampler, get_val_mask, downsample_mask)
from maniflow.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from maniflow.dataset.base_dataset import BaseDataset
from termcolor import cprint

class AdroitImageDataset(BaseDataset):
    def __init__(self,
            zarr_path,
            horizon=1,
            n_obs_steps=1,
            pad_before=0,
            pad_after=0,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
            task_name=None,
            use_img=True,
            use_depth=False,
            use_full_state=True,
            use_rl_signals=False,
            use_embedding=False,
            ):
        super().__init__()
        cprint(f'Loading AdroitImageDataset from {zarr_path}', 'green')
        self.task_name = task_name
        self.n_obs_steps = n_obs_steps
        self.use_img = use_img
        self.use_depth = use_depth
        self.use_full_state = use_full_state
        self.use_rl_signals = use_rl_signals
        self.use_embedding = use_embedding


        buffer_keys = [
            'state', 
            'action',]
        
        # Lazy load img from zarr (avoid ~46GB in RAM)
        self.lazy_img_zarr = None
        self.lazy_next_img_zarr = None
        if self.use_img:
            #self.lazy_img_zarr = zarr.open(zarr_path, mode='r')['data']['img']
            cprint(f'  Images: deferred lazy zarr loading setup', 'yellow')
            # Don't add 'img' to buffer_keys — load on demand in __getitem__
        if self.use_depth:
            buffer_keys.append('depth')
        if self.use_full_state:
            buffer_keys.append('full_state')

        self.has_v_value = False
        self.has_rl_signals = False
        self.has_next_img = False
        self.has_embedding = False
        try:
            _zr = zarr.open(zarr_path, mode='r')
            # Only load v_value for advantage weighting if NOT using RL signals (FlowQL trains its own critic)
            # if self.use_full_state and 'v_value' in _zr['data'] and not self.use_rl_signals:
            #     buffer_keys.append('v_value')
            #     self.has_v_value = True
            #     cprint('  Found v_value in zarr -> loading for advantage weighting', 'cyan')
            if self.use_rl_signals:
                rl_keys = ['reward', 'done', 'next_full_state', 'next_state']
                available = [k for k in rl_keys if k in _zr['data']]
                if len(available) == len(rl_keys):
                    buffer_keys.extend(rl_keys)
                    self.has_rl_signals = True
                    cprint('  Found RL signals (reward, done, next_full_state, next_state) -> loading for FlowQL', 'cyan')
                else:
                    missing = set(rl_keys) - set(available)
                    cprint(f'  WARNING: use_rl_signals=True but missing keys: {missing}', 'yellow')
                # next_img: derive from img[t+1] at runtime (index-based, no extra storage)
                self.has_next_img = self.use_img
                if self.has_next_img:
                    cprint(' Will load next_img', 'cyan')
            if self.use_embedding:
                emb_keys = ['img_embedding']
                if self.use_rl_signals and 'next_img_embedding' in _zr['data']:
                    emb_keys.append('next_img_embedding')
                available_emb = [k for k in emb_keys if k in _zr['data']]
                if 'img_embedding' in available_emb:
                    buffer_keys.extend(available_emb)
                    self.has_embedding = True
                    cprint(f'  Found embeddings {available_emb} -> loading for MCR policy', 'cyan')
                else:
                    cprint('  WARNING: use_embedding=True but img_embedding not found in zarr', 'yellow')
        except Exception:
            pass

        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=buffer_keys)
        
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)
        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=n_obs_steps + horizon - 1,
            pad_before=max(pad_before, n_obs_steps - 1),
            pad_after=pad_after,
            episode_mask=train_mask)
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

        self.zarr_path = zarr_path
        self.train_episodes_num = np.sum(train_mask)
        self.val_episodes_num = np.sum(val_mask)

        # # Precompute episode end indices for index-based next_img lookup
        # if self.has_next_img:
        #     ep_ends = self.replay_buffer.episode_ends[:]
        #     self._episode_end_set = set(ep_ends.tolist())
        #     self._buffer_size = int(ep_ends[-1]) if len(ep_ends) > 0 else 0

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
            )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='gaussian', **kwargs):
        data = {
            'action':    self.replay_buffer['action'],
            'agent_pos': self.replay_buffer['state'],
        }
        if self.use_full_state:
            data['full_state'] = self.replay_buffer['full_state']
        if self.has_embedding:
            data['img_embedding'] = self.replay_buffer['img_embedding']

        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)

        if self.use_full_state:
            normalizer['next_full_state'] = normalizer['full_state']
        if self.has_embedding:
            normalizer['next_img_embedding'] = normalizer['img_embedding']
        if self.has_rl_signals:
            normalizer['next_state'] = normalizer['agent_pos']

        return normalizer

    # def get_normalizer(self, mode='limits', **kwargs):
    #     data = {'action': self.replay_buffer['action']}
    #     normalizer = LinearNormalizer()
    #     normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
    #     if self.use_img:
    #         normalizer['image'] = SingleFieldLinearNormalizer.create_identity()
    #     if self.use_depth:
    #         normalizer['depth'] = SingleFieldLinearNormalizer.create_identity()
    #
    #     normalizer['agent_pos'] = SingleFieldLinearNormalizer.create_identity()
    #     if self.use_full_state:
    #         normalizer['full_state'] = SingleFieldLinearNormalizer.create_identity()
    #     if self.has_v_value:
    #         normalizer['v_value'] = SingleFieldLinearNormalizer.create_identity()
    #     if self.has_rl_signals:
    #         normalizer['reward'] = SingleFieldLinearNormalizer.create_identity()
    #         normalizer['done'] = SingleFieldLinearNormalizer.create_identity()
    #         normalizer['next_full_state'] = SingleFieldLinearNormalizer.create_identity()
    #         normalizer['next_state'] = SingleFieldLinearNormalizer.create_identity()
    #     if getattr(self, 'has_next_img', False):
    #         normalizer['next_img'] = SingleFieldLinearNormalizer.create_identity()
    #     if self.has_embedding:
    #         normalizer['img_embedding'] = SingleFieldLinearNormalizer.create_identity()
    #         normalizer['next_img_embedding'] = SingleFieldLinearNormalizer.create_identity()
    #
    #     return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        # With n_obs_steps > 1, the sequence has n_obs_steps+horizon-1 steps.
        # Images use all n_obs_steps frames; all other signals use only the
        # current step (index n_obs_steps-1) to stay shape (1, ...).
        n = self.n_obs_steps - 1

        agent_pos = sample['state'][n:n+1].astype(np.float32)

        if self.use_depth:
            depth = sample['depth'][n:n+1].astype(np.float32)

        data = {
            'obs': {
                'agent_pos': agent_pos,
                },
            'action': sample['action'][n:n+1].astype(np.float32)}
        if self.use_depth:
            data['obs']['depth'] = depth
        if self.use_full_state:
            data['obs']['full_state'] = sample['full_state'][n:n+1].astype(np.float32)
        if self.has_v_value:
            data['obs']['v_value'] = sample['v_value'][n:n+1].astype(np.float32)
        if self.has_rl_signals:
            data['obs']['reward'] = sample['reward'][n:n+1].astype(np.float32)
            data['obs']['done'] = sample['done'][n:n+1].astype(np.float32)
            data['obs']['next_full_state'] = sample['next_full_state'][n:n+1].astype(np.float32)
            data['obs']['next_state'] = sample['next_state'][n:n+1].astype(np.float32)
        if self.has_embedding:
            # data['obs']['img_embedding'] = sample['img_embedding'][n:n+1].astype(np.float32)
            data['obs']['img_embedding'] = sample['img_embedding'][:n+1].astype(np.float32)
            data['obs']['next_img_embedding'] = sample['next_img_embedding'][n:n+1].astype(np.float32)
        return data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)

        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx \
            = self.sampler.indices[idx]

        # Lazy load img from zarr
        if self.use_img:
            # Each worker opens its own connection on the first pull!
            if self.lazy_img_zarr is None:
                self.lazy_img_zarr = zarr.open(self.zarr_path, mode='r')['data']['img']
            img_sample = self.lazy_img_zarr[buffer_start_idx:buffer_end_idx]
            seq_len = self.sampler.sequence_length
            if (sample_start_idx > 0) or (sample_end_idx < seq_len):
                img_data = np.zeros(
                    (seq_len,) + img_sample.shape[1:], dtype=img_sample.dtype)
                if sample_start_idx > 0:
                    img_data[:sample_start_idx] = img_sample[0]
                if sample_end_idx < seq_len:
                    img_data[sample_end_idx:] = img_sample[-1]
                img_data[sample_start_idx:sample_end_idx] = img_sample
            else:
                img_data = img_sample
            data['obs']['image'] = img_data.astype(np.float32)

        if self.has_next_img:
            # Each worker opens its own connection on the first pull!
            if self.lazy_next_img_zarr is None:
                self.lazy_next_img_zarr = zarr.open(self.zarr_path, mode='r')['data']['next_img']
            next_img_sample = self.lazy_next_img_zarr[buffer_start_idx:buffer_end_idx]
            seq_len = self.sampler.sequence_length
            if (sample_start_idx > 0) or (sample_end_idx < seq_len):
                next_img_data = np.zeros(
                    (seq_len,) + next_img_sample.shape[1:], dtype=next_img_sample.dtype)
                if sample_start_idx > 0:
                    next_img_data[:sample_start_idx] = next_img_sample[0]
                if sample_end_idx < seq_len:
                    next_img_data[sample_end_idx:] = next_img_sample[-1]
                next_img_data[sample_start_idx:sample_end_idx] = next_img_sample
            else:
                next_img_data = next_img_sample
            data['obs']['next_img'] = next_img_data[-1:].astype(np.float32)  # (1, H, W, C): frame at t+1


        to_torch_function = lambda x: torch.from_numpy(x) if x.__class__.__name__ == 'ndarray' else x
        torch_data = dict_apply(data, to_torch_function)
        return torch_data
