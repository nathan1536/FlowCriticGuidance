"""
FlowQL Workspace: DiffusionQL-style training with flow matching policy.

Trains twin Q-critics online alongside the flow policy using:
  Policy loss: L = α * L_flow + η * L_ql
  Critic loss: L = MSE(Q(s,a), r + γ min(Q'(s', π'(s'))))

Reference: Wang et al., "Diffusion Policies as an Expressive Policy Class
for Offline Reinforcement Learning", ICLR 2023.
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
import itertools
import hydra
import torch
import torch.nn.functional as F
import dill
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader, RandomSampler, Subset
import copy
import random
import wandb
import tqdm
import numpy as np
from termcolor import cprint
import time
import threading
from hydra.core.hydra_config import HydraConfig

from maniflow.dataset.base_dataset import BaseDataset
from maniflow.env_runner.base_runner import BaseRunner
from maniflow.common.checkpoint_util import TopKCheckpointManager
from maniflow.common.pytorch_util import dict_apply, optimizer_to
from maniflow.model.diffusion.ema_model import EMAModel
from maniflow.model.common.lr_scheduler import get_scheduler
from maniflow.model.critic import TwinQCritic
from maniflow.policy.drqv2_flow_policy import DrQv2FlowPolicy

OmegaConf.register_new_resolver("eval", eval, replace=True)


class TrainFlowQLAdroitWorkspace:
    include_keys = ['global_step', 'epoch']
    exclude_keys = tuple()

    def __init__(self, cfg: OmegaConf, output_dir=None):
        self.cfg = cfg
        self._output_dir = output_dir
        self._saving_thread = None

        # set seed
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

        # FlowQL: configure twin Q-critic (trained online)
        flowql_cfg = cfg.get("flowql", {})
        state_dim = flowql_cfg.get("state_dim", 39)
        action_dim = flowql_cfg.get("action_dim", 28)
        hidden_dim = flowql_cfg.get("critic_hidden_dim", 256)

        self.critic = TwinQCritic(state_dim, action_dim, hidden_dim)
        self.critic_target = copy.deepcopy(self.critic)
        # Freeze target
        for p in self.critic_target.parameters():
            p.requires_grad = False

        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=flowql_cfg.get("critic_lr", 3e-4),
        )

        # training state
        self.global_step = 0
        self.epoch = 0

    def run(self):
        cfg = copy.deepcopy(self.cfg)
        flowql_cfg = cfg.get("flowql", {})

        # FlowQL hyperparameters
        eta = flowql_cfg.get("eta", 1.0)
        alpha = flowql_cfg.get("alpha", 1.0)
        discount = flowql_cfg.get("discount", 0.99)
        tau = flowql_cfg.get("tau", 0.005)
        grad_norm = flowql_cfg.get("grad_norm", 9.0)
        num_sample_steps = flowql_cfg.get("num_sample_steps", 4)
        max_q_backup = flowql_cfg.get("max_q_backup", False)
        warmup_epochs = flowql_cfg.get("warmup_epochs", 0)
        lr_decay = flowql_cfg.get("lr_decay", True)

        # Critic LR scheduler (CosineAnnealingLR, same as DiffusionQL)
        num_batches_per_epoch = cfg.training.get("num_batches", 100)
        lr_maxt = cfg.training.num_epochs * num_batches_per_epoch
        critic_lr_scheduler = None
        if lr_decay:
            from torch.optim.lr_scheduler import CosineAnnealingLR
            critic_lr_scheduler = CosineAnnealingLR(
                self.critic_optimizer, T_max=lr_maxt, eta_min=0.)
            cprint(f"Critic LR decay: CosineAnnealingLR, T_max={lr_maxt}", 'green')

        if cfg.training.debug:
            cfg.training.num_epochs = 100
            cfg.training.max_train_steps = 10
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 20
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1
            RUN_ROLLOUT = True
            RUN_CKPT = False
            verbose = True
        else:
            RUN_ROLLOUT = True
            RUN_CKPT = True
            verbose = False

        RUN_VALIDATION = False

        # resume training
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)


        dataset: BaseDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseDataset)

        # Reward standardization 
        reward_tune = flowql_cfg.get("reward_tune", "normalize")
        if hasattr(dataset, 'has_rl_signals') and dataset.has_rl_signals and reward_tune != "no":
            reward_data = dataset.replay_buffer['reward']
            raw_mean, raw_std = reward_data.mean(), reward_data.std()
            raw_min, raw_max = reward_data.min(), reward_data.max()
            if reward_tune == "normalize":
                reward_data[:] = (reward_data - raw_mean) / (raw_std + 1e-8)
            cprint(f"Reward {reward_tune}: raw [{raw_min:.2f}, {raw_max:.2f}] mean={raw_mean:.2f} std={raw_std:.2f}"
                   f" -> new [{reward_data.min():.2f}, {reward_data.max():.2f}]", 'green')

        cprint(f"Dataset: {dataset.__class__.__name__}", 'red')
        cprint(f"Dataset Path: {dataset.zarr_path}", 'red')
        cprint(f"Number of training episodes: {dataset.train_episodes_num}", 'red')
        cprint(f"RL signals available: {getattr(dataset, 'has_rl_signals', False)}", 'red')

        num_batches = cfg.training.get("num_batches", None)
        if num_batches is not None:
            total_samples = cfg.dataloader.batch_size * num_batches * cfg.training.num_epochs
            sampler = RandomSampler(dataset, replacement=True, num_samples=total_samples)
            dataloader_cfg = dict(cfg.dataloader)
            dataloader_cfg.pop('shuffle', None)
            train_dataloader = DataLoader(dataset, sampler=sampler, **dataloader_cfg)
            train_dataloader_iter = iter(train_dataloader)
            cprint(f"Using RandomSampler: {num_batches} batches/epoch, {total_samples} total samples", 'yellow')
        else:
            train_dataloader = DataLoader(dataset, **cfg.dataloader)
            train_dataloader_iter = None
        normalizer = dataset.get_normalizer()


        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        # configure lr scheduler
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                len(train_dataloader) * cfg.training.num_epochs) \
                    // cfg.training.gradient_accumulate_every,
            last_epoch=self.global_step-1
        )

        # configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(cfg.ema, model=self.ema_model)

        env_runner = None
        try:
            env_runner = hydra.utils.instantiate(
                cfg.task.env_runner,
                output_dir=self.output_dir)
            assert isinstance(env_runner, BaseRunner)
        except Exception as e:
            cprint(f"WARNING: Could not create env runner: {e}", 'yellow')
            cprint("Rollout evaluation will be skipped.", 'yellow')
            env_runner = None

        cfg.logging.name = str(cfg.logging.name)
        cprint("-----------------------------", "yellow")
        cprint(f"[WandB] group: {cfg.logging.group}", "yellow")
        cprint(f"[WandB] name: {cfg.logging.name}", "yellow")
        cprint("-----------------------------", "yellow")

        wandb_run = wandb.init(
            dir=str(self.output_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            **cfg.logging
        )

        
        wandb.config.update({"output_dir": self.output_dir})

        # configure checkpoint
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )

        device = torch.device(cfg.training.device)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        self.critic.to(device)
        self.critic_target.to(device)
        optimizer_to(self.optimizer, device)
        optimizer_to(self.critic_optimizer, device)

        train_sampling_batch = None

        # ═══════════════════════════════════════════════════════════════
        # Training loop
        # ═══════════════════════════════════════════════════════════════
        for local_epoch_idx in range(cfg.training.num_epochs):
            t_epoch_start = time.time()
            step_log = dict()
            train_losses = list()

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
                    leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                for batch_idx, batch in tepoch:
                    batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                    if train_sampling_batch is None:
                        train_sampling_batch = batch

                    batch_size = batch['action'].shape[0]
                    task_name = cfg.task.get("task_name", cfg.get("task_name", "door"))

                    # ── 1. Critic update (TD learning) ──
                    critic_loss_val = 0.0
                    target_q_mean = 0.0
                    if ql_active and 'reward' in batch['obs']:
                        exec_start = self.model.n_obs_steps - 1
                        states = batch['obs']['full_state'][:, exec_start]       # (B, state_dim)
                        actions = batch['action'][:, exec_start]                  # (B, action_dim)
                        rewards = batch['obs']['reward'][:, exec_start]           # (B,)
                        dones = batch['obs']['done'][:, exec_start]               # (B,)
                        next_states = batch['obs']['next_full_state'][:, exec_start]  # (B, state_dim)

                        with torch.no_grad():
                            ema_policy = self.ema_model if self.ema_model is not None else self.model
                            ema_policy.eval()

                            next_obs_dict = {}
                            # _is_image_policy = getattr(self.model, 'is_image_policy', False)
                            # if _is_image_policy:
                            #     # Image-based policy (DrQv2): encode next obs through CNN
                            #     if 'next_img' in batch['obs']:
                            #         next_obs_dict['image'] = batch['obs']['next_img'][:, :self.model.n_obs_steps]
                            #     else:
                            #         next_obs_dict['image'] = batch['obs']['image'][:, :self.model.n_obs_steps]
                            #     if 'next_state' in batch['obs']:
                            #         next_obs_dict['agent_pos'] = batch['obs']['next_state'][:, :self.model.n_obs_steps]
                            #     else:
                            #         next_obs_dict['agent_pos'] = batch['obs']['agent_pos'][:, :self.model.n_obs_steps]
                            # else:
                            #     # State-based policy (D4RL, MetaWorld state, Adroit state)
                            #     next_obs_dict['full_state'] = batch['obs']['next_full_state'][:, :self.model.n_obs_steps]
                            if getattr(self.model, 'is_embedding_policy', False):
                                next_obs_dict['img_embedding'] = batch['obs']['next_img_embedding'][:, :self.model.n_obs_steps]
                                next_obs_dict['agent_pos'] = batch['obs']['next_state'][:, :self.model.n_obs_steps]
                            elif getattr(self.model, 'is_image_policy', False):
                                next_obs_dict['image']     = batch['obs']['next_img'][:, :self.model.n_obs_steps]
                                next_obs_dict['agent_pos'] = batch['obs']['next_state'][:, :self.model.n_obs_steps]
                            else:
                                next_obs_dict['full_state'] = batch['obs']['next_full_state'][:, :self.model.n_obs_steps]
                                
                            next_nobs = self.model.normalizer.normalize(next_obs_dict)
                            next_nobs = dict_apply(next_nobs, lambda x: x.to(device))
                            next_vis_cond = self.model.obs_encoder(next_nobs)
                            next_vis_cond = next_vis_cond.reshape(batch_size, -1, self.model.obs_feature_dim)

                            noise = torch.randn(
                                batch_size, self.model.horizon, self.model.action_dim,
                                device=device)

                            if max_q_backup:
                                # Sample 10 actions, take max Q
                                n_repeat = 10
                                next_states_rpt = next_states.repeat_interleave(n_repeat, dim=0)
                                next_vis_cond_rpt = next_vis_cond.repeat_interleave(n_repeat, dim=0)
                                noise_rpt = torch.randn(
                                    batch_size * n_repeat, self.model.horizon, self.model.action_dim,
                                    device=device)
                                next_actions_norm = ema_policy.sample_ode(
                                    x0=noise_rpt, N=num_sample_steps,
                                    vis_cond=next_vis_cond_rpt)[-1]
                                next_actions_raw = self.model.normalizer['action'].unnormalize(next_actions_norm)
                                next_a0 = next_actions_raw[:, exec_start].clamp(-1, 1)
                                target_q = self.critic_target.q_min(next_states_rpt, next_a0)
                                target_q = target_q.view(batch_size, n_repeat).max(dim=1, keepdim=True)[0]
                            else:
                                next_actions_norm = ema_policy.sample_ode(
                                    x0=noise, N=num_sample_steps,
                                    vis_cond=next_vis_cond)[-1]
                                next_actions_raw = self.model.normalizer['action'].unnormalize(next_actions_norm)
                                next_a0 = next_actions_raw[:, exec_start].clamp(-1, 1)
                                target_q = self.critic_target.q_min(next_states, next_a0)  # (B, 1)

                            target_q = (rewards.unsqueeze(-1) + (1.0 - dones.unsqueeze(-1)) * discount * target_q).detach()

                        # Current Q estimates
                        q0, q1 = self.critic(states, actions)
                        critic_loss = F.mse_loss(q0, target_q) + F.mse_loss(q1, target_q)

                        self.critic_optimizer.zero_grad()
                        critic_loss.backward()
                        if grad_norm > 0:
                            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=grad_norm)
                        self.critic_optimizer.step()
                        if critic_lr_scheduler is not None:
                            critic_lr_scheduler.step()

                        critic_loss_val = critic_loss.item()
                        target_q_mean = target_q.mean().item()

                    # 2. Policy update 
                    batch['obs']['task_name'] = [task_name] * batch_size

                    # Compute flow loss
                    raw_loss, loss_dict = self.model.compute_loss(
                        batch, self.ema_model, epoch=self.epoch)

                    # Compute QL loss (if active)
                    ql_loss_val = 0.0
                    if ql_active:
                        # Encode observations for flow sampling
                        nobs = self.model.normalizer.normalize(batch['obs'])
                        this_nobs = dict_apply(nobs,
                            lambda x: x[:, :self.model.n_obs_steps, ...].to(device))
                        vis_cond = self.model.obs_encoder(this_nobs)
                        vis_cond = vis_cond.reshape(batch_size, -1, self.model.obs_feature_dim)
                        lang_cond = None

                        _has_encoder = isinstance(self.model, DrQv2FlowPolicy) or getattr(self.model, 'is_embedding_policy', False)
                        ql_vis_cond = vis_cond.detach() if _has_encoder else vis_cond
                        ql_loss, ql_log = self.model.compute_flowql_loss(
                            batch, ql_vis_cond, lang_cond, self.critic,
                            num_steps=num_sample_steps)

                        total_loss = alpha * raw_loss + eta * ql_loss
                        loss_dict.update(ql_log)
                        ql_loss_val = ql_loss.item()
                    else:
                        total_loss = raw_loss

                    loss = total_loss / cfg.training.gradient_accumulate_every
                    loss.backward()

                    if self.global_step % cfg.training.gradient_accumulate_every == 0:
                        _policy_grad_norm = cfg.training.get("max_grad_norm", 1.0)
                        if _policy_grad_norm > 0:
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(),
                                max_norm=_policy_grad_norm)
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        lr_scheduler.step()

                    # update EMA
                    if cfg.training.use_ema:
                        ema.step(self.model)

                    # update critic target 
                    if ql_active:
                        with torch.no_grad():
                            for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
                                tp.data.copy_(tau * p.data + (1 - tau) * tp.data)

                    raw_loss_cpu = total_loss.item()
                    tepoch.set_postfix(loss=raw_loss_cpu, ql=ql_loss_val, critic=critic_loss_val, refresh=False)
                    train_losses.append(raw_loss_cpu)
                    step_log = {
                        'train_loss': raw_loss_cpu,
                        'global_step': self.global_step,
                        'epoch': self.epoch,
                        'lr': lr_scheduler.get_last_lr()[0],
                        'critic_lr': critic_lr_scheduler.get_last_lr()[0] if critic_lr_scheduler else flowql_cfg.get("critic_lr", 3e-4),
                        'critic_loss': critic_loss_val,
                        'target_q_mean': target_q_mean,
                        'ql_loss': ql_loss_val,
                        'ql_active': 1.0 if ql_active else 0.0,
                    }
                    step_log.update(loss_dict)

                    is_last_batch = (batch_idx == (total_batches - 1))
                    if not is_last_batch:
                        wandb_run.log(step_log, step=self.global_step)
                        self.global_step += 1

                    if (cfg.training.max_train_steps is not None) \
                        and batch_idx >= (cfg.training.max_train_steps - 1):
                        break

            # end of epoch
            t_train_end = time.time()
            train_loss = np.mean(train_losses)
            step_log['train_loss'] = train_loss

            # ════════ Eval ════════
            policy = self.model
            if cfg.training.use_ema:
                policy = self.ema_model
            policy.eval()

            # run rollout
            t_rollout_start = time.time()
            if (self.epoch % cfg.training.rollout_every) == 0 and RUN_ROLLOUT and env_runner is not None:
                all_rollout_steps = list(set([10, cfg.policy.num_inference_steps]))
                for inference_step in all_rollout_steps:
                    cprint(f"Running rollout with inference step {inference_step}", 'green')
                    policy.num_inference_steps = inference_step
                    runner_log = env_runner.run(policy)
                    for key in runner_log:
                        step_log[f"{key}_infer{inference_step}"] = runner_log[key]
                    if inference_step == cfg.policy.num_inference_steps:
                        step_log.update(runner_log)
            t_rollout_end = time.time()

            # run diffusion sampling on a training batch
            t_sample_start = time.time()
            if (self.epoch % cfg.training.sample_every) == 0:
                with torch.no_grad():
                    batch = dict_apply(train_sampling_batch, lambda x: x.to(device, non_blocking=True))
                    obs_dict = batch['obs']
                    gt_action = batch['action']
                    result = policy.predict_action(obs_dict)
                    pred_action = result['action_pred']
                    mse = F.mse_loss(pred_action, gt_action)
                    step_log['train_action_mse_error'] = mse.item()
            t_sample_end = time.time()

            if env_runner is None:
                step_log['test_mean_score'] = -train_loss

            # checkpoint
            t_ckpt_start = time.time()
            if (self.epoch % cfg.training.checkpoint_every) == 0 and cfg.checkpoint.save_ckpt:
                if cfg.checkpoint.save_last_ckpt:
                    self.save_checkpoint()
                if 'test_mean_score' not in step_log:
                    step_log['test_mean_score'] = -train_loss
                metric_dict = {k.replace('/', '_'): v for k, v in step_log.items()}
                topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)
                if topk_ckpt_path is not None:
                    self.save_checkpoint(path=topk_ckpt_path)
            t_ckpt_end = time.time()

            policy.train()

            wandb_run.log(step_log, step=self.global_step)
            t_epoch_end = time.time()

            cprint(f"[Epoch {self.epoch}] data={t_data_end-t_data_start:.1f}s "
                   f"train={t_train_end-t_data_end:.1f}s "
                   f"rollout={t_rollout_end-t_rollout_start:.1f}s "
                   f"sample={t_sample_end-t_sample_start:.1f}s "
                   f"ckpt={t_ckpt_end-t_ckpt_start:.1f}s "
                   f"total={t_epoch_end-t_epoch_start:.1f}s", 'yellow')

            self.global_step += 1
            self.epoch += 1
            del step_log

    def eval(self):
        cfg = copy.deepcopy(self.cfg)
        lastest_ckpt_path = self.get_checkpoint_path(tag="latest")
        if lastest_ckpt_path.is_file():
            cprint(f"Resuming from checkpoint {lastest_ckpt_path}", 'magenta')
            self.load_checkpoint(path=lastest_ckpt_path)

        env_runner: BaseRunner
        env_runner = hydra.utils.instantiate(
            cfg.task.env_runner, output_dir=self.output_dir)
        assert isinstance(env_runner, BaseRunner)
        policy = self.model
        if cfg.training.use_ema:
            policy = self.ema_model
        policy.eval()
        policy.cuda()

        runner_log = env_runner.run(policy)
        cprint(f"---------------- Eval Results --------------", 'magenta')
        for key, value in runner_log.items():
            if isinstance(value, float):
                cprint(f"{key}: {value:.4f}", 'magenta')

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
        payload = {
            'cfg': self.cfg,
            'state_dicts': dict(),
            'pickles': dict()
        }

        for key, value in self.__dict__.items():
            if hasattr(value, 'state_dict') and hasattr(value, 'load_state_dict'):
                if key not in exclude_keys:
                    if use_thread:
                        payload['state_dicts'][key] = _copy_to_cpu(value.state_dict())
                    else:
                        payload['state_dicts'][key] = value.state_dict()
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
            all_checkpoints = os.listdir(checkpoint_dir)
            best_ckpt = None
            best_score = -1e10
            for ckpt in all_checkpoints:
                if 'latest' in ckpt:
                    continue
                score = float(ckpt.split('test_mean_score=')[1].split('.ckpt')[0])
                if score > best_score:
                    best_ckpt = ckpt
                    best_score = score
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
        self.load_payload(payload,
            exclude_keys=exclude_keys, include_keys=include_keys)
        return payload

    @classmethod
    def create_from_checkpoint(cls, path,
            exclude_keys=None, include_keys=None, **kwargs):
        payload = torch.load(open(path, 'rb'), pickle_module=dill)
        instance = cls(payload['cfg'])
        instance.load_payload(
            payload=payload,
            exclude_keys=exclude_keys,
            include_keys=include_keys,
            **kwargs)
        return instance


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath('config'))
)
def main(cfg):
    workspace = TrainFlowQLAdroitWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
