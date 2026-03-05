if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)


def _copy_to_cpu(x):
    if isinstance(x, torch.Tensor):
        return x.detach().to("cpu")
    elif isinstance(x, dict):
        result = dict()
        for k, v in x.items():
            result[k] = _copy_to_cpu(v)
        return result
    elif isinstance(x, list):
        return [_copy_to_cpu(k) for k in x]
    else:
        return copy.deepcopy(x)


import os
import hydra
import torch
import dill
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import random
import wandb
import tqdm
import numpy as np
from termcolor import cprint
import time
import threading
import shutil
import traceback
from hydra.core.hydra_config import HydraConfig

# 2D workspace: default policy is image-based.
# Note: the concrete policy class is controlled by cfg.policy; this import is for convenience/type clarity.
from maniflow.policy.maniflow_image_policy import ManiFlowTransformerImagePolicy as ManiFlow

from maniflow.dataset.base_dataset import BaseDataset
from maniflow.env_runner.base_runner import BaseRunner
from maniflow.common.checkpoint_util import TopKCheckpointManager
from maniflow.common.pytorch_util import dict_apply, optimizer_to
from maniflow.model.diffusion.ema_model import EMAModel
from maniflow.model.common.lr_scheduler import get_scheduler


OmegaConf.register_new_resolver("eval", eval, replace=True)


def build_qcritic(state_dim: int, action_dim: int, hidden_dim: int):
    """Build a simple Q(s,a) critic network for advantage-weighted training."""
    import torch.nn as nn
    return nn.Sequential(
        nn.Linear(state_dim + action_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, 1),
    )


class TwinQCritic(torch.nn.Module):
    """Conservative twin-Q critic: returns min(Q1(s,a), Q2(s,a))."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.q0 = build_qcritic(state_dim, action_dim, hidden_dim)
        self.q1 = build_qcritic(state_dim, action_dim, hidden_dim)

    def forward(self, sa: torch.Tensor) -> torch.Tensor:
        return torch.min(self.q0(sa), self.q1(sa))


class TrainManiFlowMetaWorldMultiTaskWorkspace2D:
    """
    2D version of `train_maniflow_metaworld_multitask_workspace.py`.
    This is functionally identical, but intended to be used with:
    - image datasets (e.g., MetaworldMultitaskImageDataset)
    - 2D env runners (e.g., MetaworldMultitaskRunner2D)
    - image policies (e.g., ManiFlowTransformerImagePolicy)
    """

    include_keys = ["global_step", "epoch"]
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

        # configure model
        self.model: ManiFlow = hydra.utils.instantiate(cfg.policy)

        self.ema_model: ManiFlow = None
        if cfg.training.use_ema:
            try:
                self.ema_model = copy.deepcopy(self.model)
            except Exception:
                self.ema_model = hydra.utils.instantiate(cfg.policy)

        self.optimizer = hydra.utils.instantiate(cfg.optimizer, params=self.model.parameters())

        self.global_step = 0
        self.epoch = 0

        # Load per-task Q-critics for advantage-weighted training
        self.task_critics = {}
        critic_cfg = cfg.get("critic_guided", {})
        if critic_cfg.get("enabled", False):
            device = torch.device(cfg.training.device)
            critic_dir = pathlib.Path(critic_cfg.get("critic_dir", "runs/sb3_metaworld_ppo/models"))
            state_dim = critic_cfg.get("state_dim", 9)
            action_dim = critic_cfg.get("action_dim", 4)
            hidden_dim = critic_cfg.get("hidden_dim", 256)
            task_names = cfg.task.get("task_names", [])
            for task_name in task_names:
                twin_path = critic_dir / task_name / "sac_twin_critic_q.pt"
                single_path = critic_dir / task_name / "best_expert_critic_q.pt"
                if twin_path.exists():
                    critic = TwinQCritic(state_dim, action_dim, hidden_dim)
                    critic.load_state_dict(torch.load(str(twin_path), map_location=device))
                    critic.eval().to(device)
                    for param in critic.parameters():
                        param.requires_grad = False
                    self.task_critics[task_name] = critic
                    cprint(f"[Critic-Guided] Loaded twin Q-critic for task: {task_name}", "green")
                elif single_path.exists():
                    critic = build_qcritic(state_dim, action_dim, hidden_dim)
                    critic.load_state_dict(torch.load(str(single_path), map_location=device))
                    critic.eval().to(device)
                    for param in critic.parameters():
                        param.requires_grad = False
                    self.task_critics[task_name] = critic
                    cprint(f"[Critic-Guided] Loaded single Q-critic for task: {task_name}", "green")
                else:
                    cprint(f"[Critic-Guided] No Q-critic found for task: {task_name} at {critic_dir / task_name}", "yellow")
            cprint(f"[Critic-Guided] Loaded {len(self.task_critics)} critics", "green")

    def run(self):
        cfg = copy.deepcopy(self.cfg)

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

        RUN_VALIDATION = True

        # resume training
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)

        # dataset
        dataset: BaseDataset = hydra.utils.instantiate(cfg.task.dataset)
        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        normalizer = dataset.get_normalizer()

        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        # print dataset info (multitask datasets usually have these fields)
        cprint(f"Dataset: {dataset.__class__.__name__}", "red")
        if hasattr(dataset, "data_path"):
            cprint(f"Dataset Path: {dataset.data_path}", "red")
        if hasattr(dataset, "task_names"):
            cprint(f"Tasks: {dataset.task_names}", "red")
        if hasattr(dataset, "train_episodes_num"):
            cprint(f"Number of training episodes: {dataset.train_episodes_num}", "red")
        if hasattr(dataset, "val_episodes_num"):
            cprint(f"Number of validation episodes: {dataset.val_episodes_num}", "red")

        # set normalizer
        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(len(train_dataloader) * cfg.training.num_epochs) // cfg.training.gradient_accumulate_every,
            last_epoch=self.global_step - 1,
        )

        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(cfg.ema, model=self.ema_model)

        # env runner (2D/3D chosen by cfg.task.env_runner)
        try:
        env_runner: BaseRunner = hydra.utils.instantiate(cfg.task.env_runner, output_dir=self.output_dir)
        except Exception as exc:
            cprint(
                f"[WARN] Env runner unavailable; skipping rollouts/eval. Reason: {exc}",
                "yellow",
            )
            cprint(traceback.format_exc(), "yellow")
            env_runner = None

        # logging
        cfg.logging.name = str(cfg.logging.name)
        cprint("-----------------------------", "yellow")
        cprint(f"[WandB] group: {cfg.logging.group}", "yellow")
        cprint(f"[WandB] name: {cfg.logging.name}", "yellow")
        cprint("-----------------------------", "yellow")

        wandb_run = wandb.init(dir=str(self.output_dir), config=OmegaConf.to_container(cfg, resolve=True), **cfg.logging)
        if hasattr(cfg.task.dataset, "task_names"):
            wandb.config.update({"output_dir": self.output_dir, "task_names": cfg.task.dataset.task_names})

        # checkpoint
        topk_manager = TopKCheckpointManager(save_dir=os.path.join(self.output_dir, "checkpoints"), **cfg.checkpoint.topk)

        # device transfer
        device = torch.device(cfg.training.device)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)

        train_sampling_batch = None

        for _local_epoch_idx in range(cfg.training.num_epochs):
            step_log = dict()
            train_losses = list()
            with tqdm.tqdm(
                train_dataloader,
                desc=f"Training epoch {self.epoch}",
                leave=False,
                mininterval=cfg.training.tqdm_interval_sec,
            ) as tepoch:
                for batch_idx, batch in enumerate(tepoch):
                    t1 = time.time()
                    batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                    if train_sampling_batch is None:
                        train_sampling_batch = batch

                    critic_cfg = cfg.get("critic_guided", {}) if len(self.task_critics) > 0 else None
                    raw_loss, loss_dict = self.model.compute_loss(
                        batch, self.ema_model,
                        critics=self.task_critics if len(self.task_critics) > 0 else None,
                        critic_cfg=critic_cfg
                    )
                    loss = raw_loss / cfg.training.gradient_accumulate_every
                    loss.backward()

                    if self.global_step % cfg.training.gradient_accumulate_every == 0:
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        lr_scheduler.step()

                    if cfg.training.use_ema:
                        ema.step(self.model)

                    raw_loss_cpu = raw_loss.item()
                    tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                    train_losses.append(raw_loss_cpu)
                    step_log = {
                        "train_loss": raw_loss_cpu,
                        "global_step": self.global_step,
                        "epoch": self.epoch,
                        "lr": lr_scheduler.get_last_lr()[0],
                    }
                    step_log.update(loss_dict)

                    if verbose:
                        t2 = time.time()
                        print(f"total one step time: {t2 - t1:.3f}")

                    is_last_batch = batch_idx == (len(train_dataloader) - 1)
                    if not is_last_batch:
                        wandb_run.log(step_log, step=self.global_step)
                        self.global_step += 1

                    if (cfg.training.max_train_steps is not None) and batch_idx >= (cfg.training.max_train_steps - 1):
                        break

            train_loss = np.mean(train_losses)
            step_log["train_loss"] = train_loss

            # rollout
            policy = self.ema_model if cfg.training.use_ema else self.model
            policy.eval()

            if (self.epoch % cfg.training.rollout_every) == 0 and RUN_ROLLOUT and env_runner is not None and self.epoch > 1:
                runner_log = env_runner.run(policy)
                # Aggregate mean metrics across tasks if multitask dict
                if isinstance(runner_log, dict) and hasattr(cfg.task.dataset, "task_names"):
                    task_names = cfg.task.dataset.task_names
                    if len(task_names) > 0 and task_names[0] in runner_log:
                        mean_metrics = {}
                        for metric in runner_log[task_names[0]].keys():
                            if metric in ["sim_video_eval"]:
                                continue
                            values = [runner_log[t][metric] for t in task_names]
                            mean_metrics[f"mean/{metric}"] = np.mean(values)
                        runner_log.update(mean_metrics)
                step_log.update(runner_log)

            # validation
            if (self.epoch % cfg.training.val_every) == 0 and RUN_VALIDATION:
                with torch.no_grad():
                    val_losses = list()
                    with tqdm.tqdm(
                        val_dataloader,
                        desc=f"Validation epoch {self.epoch}",
                        leave=False,
                        mininterval=cfg.training.tqdm_interval_sec,
                    ) as tepoch:
                        for batch_idx, batch in enumerate(tepoch):
                            batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                            loss, loss_dict = self.model.compute_loss(batch, self.ema_model)
                            val_losses.append(loss)
                            if (cfg.training.max_val_steps is not None) and batch_idx >= (cfg.training.max_val_steps - 1):
                                break
                    if len(val_losses) > 0:
                        val_loss = torch.mean(torch.tensor(val_losses)).item()
                        step_log["val_loss"] = val_loss

            # diffusion sampling sanity metric on a training batch
            if (self.epoch % cfg.training.sample_every) == 0 and train_sampling_batch is not None:
                with torch.no_grad():
                    batch = dict_apply(train_sampling_batch, lambda x: x.to(device, non_blocking=True))
                    obs_dict = batch["obs"]
                    gt_action = batch["action"]
                    result = policy.predict_action(obs_dict)
                    pred_action = result["action_pred"]
                    mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                    step_log["train_action_mse_error"] = mse.item()

            if env_runner is None:
                step_log["test_mean_score"] = -train_loss

            # checkpoint
            if (self.epoch % cfg.training.checkpoint_every) == 0 and cfg.checkpoint.save_ckpt:
                if cfg.checkpoint.save_last_ckpt:
                    self.save_checkpoint()
                if cfg.checkpoint.save_last_snapshot:
                    self.save_snapshot()

                metric_dict = dict()
                for key, value in step_log.items():
                    if key.startswith("mean/"):
                        new_key = key.replace("mean/", "")
                    else:
                        new_key = key.replace("/", "_")
                    metric_dict[new_key] = value

                try:
                    topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)
                except Exception as e:
                    print(f"Error in getting topk ckpt path: {e}")
                    topk_ckpt_path = None

                if topk_ckpt_path is not None:
                    self.save_checkpoint(path=topk_ckpt_path)

            policy.train()

            wandb_run.log(step_log, step=self.global_step)
            self.global_step += 1
            self.epoch += 1
            del step_log

    def eval(self, mode="latest"):
        cfg = copy.deepcopy(self.cfg)
        lastest_ckpt_path = self.get_checkpoint_path(tag=mode, monitor_key=cfg.checkpoint.topk.monitor_key)
        if lastest_ckpt_path.is_file():
            cprint(f"Resuming from {mode} checkpoint {lastest_ckpt_path}", "magenta")
            self.load_checkpoint(path=lastest_ckpt_path)
            cprint(f"{self.epoch} epochs, {self.global_step} steps", "magenta")

        # Ensure normalizer is set for eval-only runs.
        dataset: BaseDataset = hydra.utils.instantiate(cfg.task.dataset)
        normalizer = dataset.get_normalizer()
        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        try:
        env_runner: BaseRunner = hydra.utils.instantiate(cfg.task.env_runner, output_dir=self.output_dir)
        except Exception as exc:
            cprint(
                f"[WARN] Evaluation skipped because env runner failed to initialize: {exc}",
                "yellow",
            )
            cprint(traceback.format_exc(), "yellow")
            return
        policy = self.ema_model if cfg.training.use_ema else self.model
        policy.eval()
        policy.cuda()

        runner_log = env_runner.run(policy)
        eval_dir = os.path.join(self.output_dir, f"eval_results/{self.epoch}")
        os.makedirs(eval_dir, exist_ok=True)

        cprint(f"---------------- Eval Results --------------", "magenta")
        metrics_dict = {}
        for key, value in runner_log.items():
            if isinstance(value, float):
                metrics_dict[key] = value
                cprint(f"{key}: {value:.4f}", "magenta")
            if isinstance(value, dict):
                for k, v in value.items():
                    if isinstance(v, float):
                        metrics_dict[f"{key}/{k}"] = v
                        cprint(f"{key}/{k}: {v:.4f}", "magenta")

        import json

        metrics_path = os.path.join(eval_dir, f"metrics_{mode}_{self.epoch}.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics_dict, f, indent=4)

        # Save videos if they exist in runner_log
        runner_log.pop("average_success_rate", None)
        for task_name, task_dict in runner_log.items():
            for k, v in task_dict.items():
                if isinstance(v, np.ndarray) and "video" in k:
                    video_dir = os.path.join(eval_dir, "videos", task_name)
                    os.makedirs(video_dir, exist_ok=True)
                    video_path = os.path.join(video_dir, f"{k}_{mode}_{self.epoch}.mp4")
                    v = np.transpose(v, (0, 2, 3, 1))
                    import imageio

                    imageio.mimsave(video_path, v, fps=10)
                elif "video" in k and hasattr(v, "_path"):
                    video_dir = os.path.join(eval_dir, "videos", task_name)
                    os.makedirs(video_dir, exist_ok=True)
                    video_path = os.path.join(video_dir, f"{k}_{mode}_{self.epoch}.mp4")
                    shutil.copy2(v._path, video_path)
        cprint(f"Evaluation results saved to {eval_dir}", "magenta")

    @property
    def output_dir(self):
        output_dir = self._output_dir
        if output_dir is None:
            output_dir = HydraConfig.get().runtime.output_dir
        return output_dir

    def save_checkpoint(self, path=None, tag="latest", exclude_keys=None, include_keys=None, use_thread=False):
        if path is None:
            path = pathlib.Path(self.output_dir).joinpath("checkpoints", f"{tag}.ckpt")
        else:
            path = pathlib.Path(path)
        if exclude_keys is None:
            exclude_keys = tuple(self.exclude_keys)
        if include_keys is None:
            include_keys = tuple(self.include_keys) + ("_output_dir",)

        path.parent.mkdir(parents=False, exist_ok=True)
        payload = {"cfg": self.cfg, "state_dicts": dict(), "pickles": dict()}

        for key, value in self.__dict__.items():
            if hasattr(value, "state_dict") and hasattr(value, "load_state_dict"):
                if key not in exclude_keys:
                    payload["state_dicts"][key] = _copy_to_cpu(value.state_dict()) if use_thread else value.state_dict()
            elif key in include_keys:
                payload["pickles"][key] = dill.dumps(value)

        if use_thread:
            self._saving_thread = threading.Thread(target=lambda: torch.save(payload, path.open("wb"), pickle_module=dill))
            self._saving_thread.start()
        else:
            torch.save(payload, path.open("wb"), pickle_module=dill)

        del payload
        torch.cuda.empty_cache()
        return str(path.absolute())

    def get_checkpoint_path(self, tag="latest", monitor_key="test_mean_score"):
        if tag == "latest":
            return pathlib.Path(self.output_dir).joinpath("checkpoints", f"{tag}.ckpt")
        elif tag == "best":
            checkpoint_dir = pathlib.Path(self.output_dir).joinpath("checkpoints")
            all_checkpoints = os.listdir(checkpoint_dir)
            best_ckpt = None
            best_score = -1e10 if "loss" not in monitor_key else float("inf")
            for ckpt in all_checkpoints:
                if "latest" in ckpt:
                    continue
                try:
                    score_str = ckpt.split(f"{monitor_key}=")[1].split(".ckpt")[0]
                    score = float(score_str)
                    if "loss" in monitor_key:
                        if score < best_score:
                            best_ckpt = ckpt
                            best_score = score
                    else:
                        if score > best_score:
                            best_ckpt = ckpt
                            best_score = score
                except (IndexError, ValueError):
                    continue

            if best_ckpt is None:
                raise ValueError(f"No checkpoints found with monitor key: {monitor_key}")
            return pathlib.Path(self.output_dir).joinpath("checkpoints", best_ckpt)
        else:
            raise NotImplementedError(f"tag {tag} not implemented")

    def load_payload(self, payload, exclude_keys=None, include_keys=None, **kwargs):
        if exclude_keys is None:
            exclude_keys = tuple()
        if include_keys is None:
            include_keys = payload["pickles"].keys()
        for key, value in payload["state_dicts"].items():
            if key not in exclude_keys:
                self.__dict__[key].load_state_dict(value, **kwargs)
        for key in include_keys:
            if key in payload["pickles"]:
                self.__dict__[key] = dill.loads(payload["pickles"][key])

    def load_checkpoint(self, path=None, tag="latest", exclude_keys=None, include_keys=None, **kwargs):
        if path is None:
            path = self.get_checkpoint_path(tag=tag)
        else:
            path = pathlib.Path(path)
        payload = torch.load(path.open("rb"), pickle_module=dill, map_location="cpu")
        self.load_payload(payload, exclude_keys=exclude_keys, include_keys=include_keys)
        return payload

    def save_snapshot(self, tag="latest"):
        path = pathlib.Path(self.output_dir).joinpath("snapshots", f"{tag}.pkl")
        path.parent.mkdir(parents=False, exist_ok=True)
        torch.save(self, path.open("wb"), pickle_module=dill)
        return str(path.absolute())

    @classmethod
    def create_from_snapshot(cls, path):
        return torch.load(open(path, "rb"), pickle_module=dill)


@hydra.main(version_base=None, config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")))
def main(cfg):
    workspace = TrainManiFlowMetaWorldMultiTaskWorkspace2D(cfg)
    workspace.run()


if __name__ == "__main__":
    main()


