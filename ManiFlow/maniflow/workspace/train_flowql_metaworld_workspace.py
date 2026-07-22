"""
FlowQL workspace for MetaWorld (multitask, image-based).

Supports three actor types via cfg.policy:
  - MCRFlowPolicy     : precomputed MCR embeddings (fastest training)
  - DrQv2FlowPolicy   : online CNN encoder (DRQ-v2 style)
  - ManiFlowStatePolicy: full_state directly (state-based)

Critic is always state-based (asymmetric): Q(full_state, action).
"""
if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)


def _copy_to_cpu(x):
    import copy
    if isinstance(x, torch.Tensor):
        return x.detach().to('cpu')
    elif isinstance(x, dict):
        return {k: _copy_to_cpu(v) for k, v in x.items()}
    elif isinstance(x, list):
        return [_copy_to_cpu(k) for k in x]
    else:
        return copy.deepcopy(x)


import os
import hydra
import torch
import torch.nn.functional as F
import dill
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader, RandomSampler
import copy
import random
import wandb
import tqdm
import numpy as np
from termcolor import cprint
import time
import threading
import traceback
from hydra.core.hydra_config import HydraConfig

from maniflow.dataset.base_dataset import BaseDataset
from maniflow.env_runner.base_runner import BaseRunner
from maniflow.common.checkpoint_util import TopKCheckpointManager
from maniflow.common.pytorch_util import dict_apply, optimizer_to
from maniflow.model.diffusion.ema_model import EMAModel
from maniflow.model.common.lr_scheduler import get_scheduler
from maniflow.model.critic import TwinQCritic
from maniflow.policy.drqv2_flow_policy import DrQv2FlowPolicy, DrQv2StateCritic
from maniflow.policy.mcr_flow_policy import MCRFlowPolicy

OmegaConf.register_new_resolver("eval", eval, replace=True)


class TrainFlowQLMetaWorldWorkspace:
    include_keys = ['global_step', 'epoch']
    exclude_keys = tuple()

    def __init__(self, cfg: OmegaConf, output_dir=None):
        self.cfg = cfg
        self._output_dir = output_dir
        self._saving_thread = None

        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.model = hydra.utils.instantiate(cfg.policy)

        self.ema_model = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.model.parameters())

        # FlowQL critic
        flowql_cfg = cfg.get("flowql", {})
        state_dim  = flowql_cfg.get("state_dim", 39)
        action_dim = flowql_cfg.get("action_dim", 4)
        hidden_dim = flowql_cfg.get("critic_hidden_dim", 256)

        if isinstance(self.model, DrQv2FlowPolicy):
            self.critic = DrQv2StateCritic(state_dim, action_dim, hidden_dim)
        else:
            self.critic = TwinQCritic(state_dim, action_dim, hidden_dim)
        self.critic_target = copy.deepcopy(self.critic)
        for p in self.critic_target.parameters():
            p.requires_grad = False

        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=flowql_cfg.get("critic_lr", 3e-4),
        )

        self.global_step = 0
        self.epoch = 0

    def run(self):
        cfg = copy.deepcopy(self.cfg)
        flowql_cfg = cfg.get("flowql", {})

        eta            = flowql_cfg.get("eta", 1.0)
        alpha          = flowql_cfg.get("alpha", 1.0)
        discount       = flowql_cfg.get("discount", 0.99)
        tau            = flowql_cfg.get("tau", 0.005)
        grad_norm      = flowql_cfg.get("grad_norm", 8.0)
        num_sample_steps = flowql_cfg.get("num_sample_steps", 5)
        max_q_backup   = flowql_cfg.get("max_q_backup", False)
        warmup_epochs  = flowql_cfg.get("warmup_epochs", 0)
        lr_decay       = flowql_cfg.get("lr_decay", True)

        num_batches_per_epoch = cfg.training.get("num_batches", 1000)
        lr_maxt = cfg.training.num_epochs * num_batches_per_epoch

        critic_lr_scheduler = None
        if lr_decay:
            from torch.optim.lr_scheduler import CosineAnnealingLR
            critic_lr_scheduler = CosineAnnealingLR(
                self.critic_optimizer, T_max=lr_maxt, eta_min=0.)
            cprint(f"Critic LR decay: CosineAnnealingLR, T_max={lr_maxt}", 'green')

        if cfg.training.debug:
            cfg.training.num_epochs = 10
            cfg.training.max_train_steps = 10
            cfg.training.rollout_every = 5
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1

        if cfg.training.resume:
            latest_ckpt = self.get_checkpoint_path()
            if latest_ckpt.is_file():
                cprint(f"Resuming from {latest_ckpt}", 'magenta')
                self.load_checkpoint(path=latest_ckpt)

        # ── Dataset ──────────────────────────────────────────────────────────
        dataset: BaseDataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseDataset)



        num_batches = cfg.training.get("num_batches", None)
        if num_batches is not None:
            total_samples = cfg.dataloader.batch_size * num_batches * cfg.training.num_epochs
            sampler = RandomSampler(dataset, replacement=True, num_samples=total_samples)
            dl_cfg = dict(cfg.dataloader)
            dl_cfg.pop('shuffle', None)
            train_dataloader = DataLoader(dataset, sampler=sampler, **dl_cfg)
            train_dataloader_iter = iter(train_dataloader)
            cprint(f"RandomSampler: {num_batches} batches/epoch, {total_samples} total", 'yellow')
        else:
            train_dataloader = DataLoader(dataset, **cfg.dataloader)
            train_dataloader_iter = None

        normalizer = dataset.get_normalizer()

        reward_tune = flowql_cfg.get("reward_tune", "normalize")
        if hasattr(dataset, 'has_rl_signals') and dataset.has_rl_signals and reward_tune != "no":
            reward_data = dataset.replay_buffer['reward']
            raw_mean, raw_std = reward_data.mean(), reward_data.std()
            raw_min,  raw_max  = reward_data.min(),  reward_data.max()
            if reward_tune == "normalize":
                reward_data[:] = reward_data / (raw_std + 1e-8)
            cprint(f"Reward {reward_tune}: raw [{raw_min:.2f}, {raw_max:.2f}] "
                   f"mean={raw_mean:.2f} std={raw_std:.2f} "
                   f"-> new [{reward_data.min():.2f}, {reward_data.max():.2f}]", 'green')

        cprint(f"Dataset: {dataset.__class__.__name__}", 'red')
        if hasattr(dataset, 'zarr_path'):
            cprint(f"Dataset Path: {dataset.zarr_path}", 'red')
        if hasattr(dataset, 'task_names'):
            cprint(f"Tasks: {dataset.task_names}", 'red')
        if hasattr(dataset, 'train_episodes_num'):
            cprint(f"Training episodes: {dataset.train_episodes_num}", 'red')
        cprint(f"RL signals: {getattr(dataset, 'has_rl_signals', False)}", 'red')

        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        self.model.set_normalizer(normalizer)
        if self.ema_model is not None:
            self.ema_model.set_normalizer(normalizer)

        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=lr_maxt // cfg.training.gradient_accumulate_every,
            last_epoch=self.global_step - 1,
        )

        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(cfg.ema, model=self.ema_model)

        env_runner = None
        try:
            env_runner = hydra.utils.instantiate(
                cfg.task.env_runner, output_dir=self.output_dir)
            assert isinstance(env_runner, BaseRunner)
        except Exception as e:
            cprint(f"WARNING: env runner unavailable: {e}", 'yellow')
            cprint(traceback.format_exc(), 'yellow')
            cprint("Rollout evaluation will be skipped.", 'yellow')

        cfg.logging.name = str(cfg.logging.name)
        cprint("-----------------------------", "yellow")
        cprint(f"[WandB] group: {cfg.logging.group}", "yellow")
        cprint(f"[WandB] name:  {cfg.logging.name}", "yellow")
        cprint("-----------------------------", "yellow")

        wandb_run = wandb.init(
            dir=str(self.output_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            **cfg.logging,
        )
        wandb.config.update({"output_dir": self.output_dir})

        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk,
        )

        device = torch.device(cfg.training.device)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        self.critic.to(device)
        self.critic_target.to(device)
        optimizer_to(self.optimizer, device)

        # ── Training loop ─────────────────────────────────────────────────────
        for _local_epoch_idx in range(cfg.training.num_epochs):
            t_epoch_start = time.time()
            step_log = dict()
            train_losses = []

            t_data_start = time.time()
            if train_dataloader_iter is not None:
                epoch_batches = (next(train_dataloader_iter) for _ in range(num_batches))
                epoch_iter = enumerate(epoch_batches)
            else:
                epoch_iter = enumerate(train_dataloader)
            t_data_end = time.time()

            total_batches = num_batches if train_dataloader_iter is not None else len(train_dataloader)
            ql_active = self.epoch >= warmup_epochs

            with tqdm.tqdm(epoch_iter, total=total_batches,
                           desc=f"Training epoch {self.epoch}",
                           leave=False,
                           mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                for batch_idx, batch in tepoch:
                    batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                    train_sampling_batch = batch

                    batch_size = batch['action'].shape[0]
                    task_name  = cfg.task.get("task_name", cfg.get("task_name", "metaworld"))

                    # ── 1. Critic update ──────────────────────────────────────
                    critic_loss_val = 0.0
                    target_q_mean   = 0.0
                    if ql_active:
                        if 'reward' not in batch['obs']:
                            raise RuntimeError("'reward' missing from batch — set use_rl_signals: true")
                        exec_start  = self.model.n_obs_steps - 1
                        states      = batch['obs']['full_state'][:, exec_start]
                        actions     = batch['action'][:, exec_start]
                        rewards     = batch['obs']['reward'][:, exec_start]
                        dones       = batch['obs']['done'][:, exec_start]
                        next_states = batch['obs']['next_full_state'][:, exec_start]

                        with torch.no_grad():
                            ema_policy = self.ema_model if self.ema_model is not None else self.model
                            ema_policy.eval()

                            next_obs_dict = {}
                            if getattr(self.model, 'is_embedding_policy', False):
                                next_obs_dict['img_embedding'] = batch['obs']['next_img_embedding'][:, :self.model.n_obs_steps]
                            elif getattr(self.model, 'is_image_policy', False):
                                next_obs_dict['image']     = batch['obs']['next_img'][:, :self.model.n_obs_steps]
                                next_obs_dict['agent_pos'] = batch['obs']['next_state'][:, :self.model.n_obs_steps]
                            else:
                                next_obs_dict['full_state'] = batch['obs']['next_full_state'][:, :self.model.n_obs_steps]

                            next_nobs     = self.model.normalizer.normalize(next_obs_dict)
                            next_nobs     = dict_apply(next_nobs, lambda x: x.to(device))
                            next_vis_cond = ema_policy.obs_encoder(next_nobs)
                            next_vis_cond = next_vis_cond.reshape(batch_size, -1, self.model.obs_feature_dim)

                            noise = torch.randn(
                                batch_size, self.model.horizon, self.model.action_dim,
                                device=device)

                            if max_q_backup:
                                n_repeat = 10
                                next_states_rpt   = next_states.repeat_interleave(n_repeat, dim=0)
                                next_vis_cond_rpt = next_vis_cond.repeat_interleave(n_repeat, dim=0)
                                noise_rpt = torch.randn(
                                    batch_size * n_repeat, self.model.horizon, self.model.action_dim,
                                    device=device)
                                next_actions_norm = ema_policy.sample_ode(
                                    x0=noise_rpt, N=num_sample_steps,
                                    vis_cond=next_vis_cond_rpt)[-1]
                                next_actions_raw = self.model.normalizer['action'].unnormalize(next_actions_norm)
                                next_a0  = next_actions_raw[:, exec_start].clamp(-1, 1)
                                target_q = self.critic_target.q_min(next_states_rpt, next_a0)
                                target_q = target_q.view(batch_size, n_repeat).max(dim=1, keepdim=True)[0]
                            else:
                                next_actions_norm = ema_policy.sample_ode(
                                    x0=noise, N=num_sample_steps,
                                    vis_cond=next_vis_cond)[-1]
                                next_actions_raw = self.model.normalizer['action'].unnormalize(next_actions_norm)
                                next_a0  = next_actions_raw[:, exec_start].clamp(-1, 1)
                                target_q = self.critic_target.q_min(next_states, next_a0)

                            target_q = (rewards.unsqueeze(-1) +
                                        (1.0 - dones.unsqueeze(-1)) * discount * target_q).detach()

                        if self.ema_model is None:
                            self.model.train()

                        q0, q1 = self.critic(states, actions)
                        critic_loss = F.mse_loss(q0, target_q) + F.mse_loss(q1, target_q)

                        self.critic_optimizer.zero_grad()
                        critic_loss.backward()
                        critic_grad_norm = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), grad_norm if grad_norm > 0 else float('inf'))
                        self.critic_optimizer.step()
                        if critic_lr_scheduler is not None:
                            critic_lr_scheduler.step()

                        critic_loss_val = critic_loss.item()
                        target_q_mean   = target_q.mean().item()

                    # ── 2. Policy update ──────────────────────────────────────
                    batch['obs']['task_name'] = [task_name] * batch_size
                    nobs      = self.model.normalizer.normalize(batch['obs'])
                    this_nobs = dict_apply(nobs,
                        lambda x: x[:, :self.model.n_obs_steps, ...].to(device))
                    vis_cond  = self.model.obs_encoder(this_nobs)
                    vis_cond  = vis_cond.reshape(batch_size, -1, self.model.obs_feature_dim)

                    raw_loss, loss_dict = self.model.compute_loss(
                        batch, self.ema_model, epoch=self.epoch, vis_cond=vis_cond)

                    ql_loss_val = 0.0
                    if ql_active:
                        # _has_encoder = isinstance(self.model, DrQv2FlowPolicy) or getattr(self.model, 'is_embedding_policy', False)
                        # ql_vis_cond = vis_cond.detach() if _has_encoder else vis_cond
                        ql_loss, ql_log = self.model.compute_flowql_loss(
                            batch, vis_cond, None, self.critic,
                            num_steps=num_sample_steps)
                        total_loss  = alpha * raw_loss + eta * ql_loss
                        loss_dict.update(ql_log)
                        ql_loss_val = ql_loss.item()
                    else:
                        total_loss = raw_loss

                    (total_loss / cfg.training.gradient_accumulate_every).backward()

                    if self.global_step % cfg.training.gradient_accumulate_every == 0:
                        actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            max_norm=cfg.training.get("max_grad_norm", 10.0))
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        lr_scheduler.step()

                    if cfg.training.use_ema:
                        ema.step(self.model)

                    if ql_active:
                        with torch.no_grad():
                            for p, tp in zip(self.critic.parameters(),
                                             self.critic_target.parameters()):
                                tp.data.copy_(tau * p.data + (1 - tau) * tp.data)

                    raw_loss_cpu = total_loss.item()
                    tepoch.set_postfix(loss=raw_loss_cpu, ql=ql_loss_val,
                                       critic=critic_loss_val, refresh=False)
                    train_losses.append(raw_loss_cpu)
                    step_log = {
                        'train_loss':      raw_loss_cpu,
                        'global_step':     self.global_step,
                        'epoch':           self.epoch,
                        'lr':              lr_scheduler.get_last_lr()[0],
                        'critic_lr':       critic_lr_scheduler.get_last_lr()[0] if critic_lr_scheduler
                                           else flowql_cfg.get("critic_lr", 3e-4),
                        'critic_loss':     critic_loss_val,
                        'target_q_mean':   target_q_mean,
                        'ql_loss':         ql_loss_val,
                        'ql_active':       1.0 if ql_active else 0.0,
                        'actor_grad_norm': actor_grad_norm.item() if ql_active else 0.0,
                        'critic_grad_norm': critic_grad_norm.item() if ql_active else 0.0,
                    }
                    step_log.update(loss_dict)

                    is_last_batch = (batch_idx == total_batches - 1)
                    if not is_last_batch:
                        wandb_run.log(step_log, step=self.global_step)
                        self.global_step += 1

                    if (cfg.training.max_train_steps is not None and
                            batch_idx >= cfg.training.max_train_steps - 1):
                        break

            # ── End of epoch ──────────────────────────────────────────────────
            t_train_end = time.time()
            step_log['train_loss'] = np.mean(train_losses)

            policy = self.ema_model if cfg.training.use_ema else self.model
            policy.eval()

            # Rollout
            t_rollout_start = time.time()
            if (self.epoch % cfg.training.rollout_every == 0 and env_runner is not None):
                all_steps = list(set([10, cfg.policy.num_inference_steps]))
                for n_steps in all_steps:
                    cprint(f"Rollout with inference_steps={n_steps}", 'green')
                    policy.num_inference_steps = n_steps
                    runner_log = env_runner.run(policy)
                    for k, v in runner_log.items():
                        step_log[f'{k}_infer{n_steps}'] = v
                    if n_steps == cfg.policy.num_inference_steps:
                        step_log.update(runner_log)
            t_rollout_end = time.time()

            # Sample sanity check
            t_sample_start = time.time()
            if self.epoch % cfg.training.sample_every == 0:
                with torch.no_grad():
                    obs_dict  = dict_apply(train_sampling_batch['obs'],
                                           lambda x: x.to(device, non_blocking=True))
                    gt_action = train_sampling_batch['action'].to(device)
                    result    = policy.predict_action(obs_dict)
                    mse = F.mse_loss(result['action_pred'], gt_action)
                    step_log['train_action_mse_error'] = mse.item()
            t_sample_end = time.time()

            if env_runner is None:
                step_log['test_mean_score'] = -np.mean(train_losses)

            # Checkpoint
            t_ckpt_start = time.time()
            if (self.epoch % cfg.training.checkpoint_every == 0 and cfg.checkpoint.save_ckpt):
                if cfg.checkpoint.save_last_ckpt:
                    self.save_checkpoint()
                if 'test_mean_score' not in step_log:
                    step_log['test_mean_score'] = -np.mean(train_losses)
                metric_dict = {k.replace('/', '_'): v for k, v in step_log.items()}
                topk_path = topk_manager.get_ckpt_path(metric_dict)
                if topk_path is not None:
                    self.save_checkpoint(path=topk_path)
            t_ckpt_end = time.time()

            policy.train()
            wandb_run.log(step_log, step=self.global_step)
            t_epoch_end = time.time()

            cprint(f"[Epoch {self.epoch}] "
                   f"train={t_train_end - t_data_end:.1f}s "
                   f"rollout={t_rollout_end - t_rollout_start:.1f}s "
                   f"sample={t_sample_end - t_sample_start:.1f}s "
                   f"ckpt={t_ckpt_end - t_ckpt_start:.1f}s "
                   f"total={t_epoch_end - t_epoch_start:.1f}s", 'yellow')

            self.global_step += 1
            self.epoch += 1
            del step_log

    # ── Checkpoint helpers ────────────────────────────────────────────────────

    @property
    def output_dir(self):
        output_dir = self._output_dir
        if output_dir is None:
            output_dir = HydraConfig.get().runtime.output_dir
        return output_dir

    def save_checkpoint(self, path=None, tag='latest',
                        exclude_keys=None, include_keys=None, use_thread=False):
        if path is None:
            path = pathlib.Path(self.output_dir).joinpath('checkpoints', f'{tag}.ckpt')
        else:
            path = pathlib.Path(path)
        if exclude_keys is None:
            exclude_keys = tuple(self.exclude_keys)
        if include_keys is None:
            include_keys = tuple(self.include_keys) + ('_output_dir',)

        path.parent.mkdir(parents=False, exist_ok=True)
        payload = {'cfg': self.cfg, 'state_dicts': {}, 'pickles': {}}
        for key, value in self.__dict__.items():
            if hasattr(value, 'state_dict') and hasattr(value, 'load_state_dict'):
                if key not in exclude_keys:
                    payload['state_dicts'][key] = (
                        _copy_to_cpu(value.state_dict()) if use_thread else value.state_dict())
            elif key in include_keys:
                payload['pickles'][key] = dill.dumps(value)

        if use_thread:
            self._saving_thread = threading.Thread(
                target=lambda: torch.save(payload, path.open('wb'), pickle_module=dill))
            self._saving_thread.start()
        else:
            torch.save(payload, path.open('wb'), pickle_module=dill)

        del payload
        torch.cuda.empty_cache()
        return str(path.absolute())

    def get_checkpoint_path(self, tag='latest'):
        if tag == 'latest':
            return pathlib.Path(self.output_dir).joinpath('checkpoints', f'{tag}.ckpt')
        elif tag == 'best':
            checkpoint_dir = pathlib.Path(self.output_dir).joinpath('checkpoints')
            best_ckpt, best_score = None, -1e10
            for ckpt in os.listdir(checkpoint_dir):
                if 'latest' in ckpt:
                    continue
                try:
                    score = float(ckpt.split('test_mean_score=')[1].split('.ckpt')[0])
                    if score > best_score:
                        best_ckpt, best_score = ckpt, score
                except (IndexError, ValueError):
                    continue
            return pathlib.Path(self.output_dir).joinpath('checkpoints', best_ckpt)
        else:
            raise NotImplementedError(f"tag {tag} not implemented")

    def load_payload(self, payload, exclude_keys=None, include_keys=None, **kwargs):
        if exclude_keys is None:
            exclude_keys = tuple()
        if include_keys is None:
            include_keys = payload['pickles'].keys()
        for key, value in payload['state_dicts'].items():
            if key not in exclude_keys:
                self.__dict__[key].load_state_dict(value, **kwargs)
        for key in include_keys:
            if key in payload['pickles']:
                self.__dict__[key] = dill.loads(payload['pickles'][key])

    def load_checkpoint(self, path=None, tag='latest',
                        exclude_keys=None, include_keys=None, **kwargs):
        if path is None:
            path = self.get_checkpoint_path(tag=tag)
        else:
            path = pathlib.Path(path)
        payload = torch.load(path.open('rb'), pickle_module=dill, map_location='cpu')
        self.load_payload(payload, exclude_keys=exclude_keys, include_keys=include_keys)
        return payload


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath('config')),
)
def main(cfg):
    workspace = TrainFlowQLMetaWorldWorkspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
