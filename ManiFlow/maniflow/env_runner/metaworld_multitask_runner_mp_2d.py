from typing import Dict, List
import copy

import numpy as np
import tqdm
import torch
import torch.multiprocessing as mp
from termcolor import cprint

from maniflow.env_runner.base_runner import BaseRunner
from maniflow.env_runner.metaworld_runner_2d import MetaworldRunner2D
from maniflow.policy.base_policy import BasePolicy


def allocate_tasks_to_gpus(task_names, devices):
    """
    Distribute tasks across GPUs to ensure balanced load distribution.
    (Same helper as the 3D MP runner.)
    """
    num_devices = len(devices)
    num_tasks = len(task_names)

    base_tasks_per_gpu = num_tasks // num_devices
    remainder = num_tasks % num_devices

    tasks_per_gpu = [base_tasks_per_gpu + (1 if i < remainder else 0) for i in range(num_devices)]
    rank_to_device = {rank: device for rank, device in enumerate(devices)}

    allocation = {}
    current_task_idx = 0
    for gpu_idx, num_tasks_for_gpu in enumerate(tasks_per_gpu):
        for _ in range(num_tasks_for_gpu):
            if current_task_idx < len(task_names):
                allocation[task_names[current_task_idx]] = gpu_idx
                current_task_idx += 1

    return allocation, rank_to_device


class MetaworldMultitaskRunner2D(BaseRunner):
    """
    2D multiprocessing version of `MetaworldMultitaskRunner`:
    - uses `MetaworldRunner2D` per task
    - same GPU/process orchestration as `metaworld_multitask_runner_mp.py`
    """

    def __init__(
        self,
        output_dir,
        task_names: List[str],
        eval_episodes=20,
        max_steps=1000,
        n_obs_steps=8,
        n_action_steps=8,
        fps=10,
        crf=22,
        render_size=84,
        tqdm_interval_sec=5.0,
        n_envs=None,
        n_train=None,
        n_test=None,
        device="cuda:0",
        max_processes=12,  # Maximum processes, 12 for 4 GPUs
        image_size: int = 128,
        camera_name: str = "corner2",
    ):
        super().__init__(output_dir)

        self.task_names = task_names
        self.output_dir = output_dir
        self.eval_episodes = eval_episodes
        self.max_steps = max_steps
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.fps = fps
        self.crf = crf
        self.render_size = render_size
        self.tqdm_interval_sec = tqdm_interval_sec
        self.n_envs = n_envs
        self.n_train = n_train
        self.n_test = n_test
        self.max_processes = max_processes
        self.image_size = image_size
        self.camera_name = camera_name

        self.device = device
        self.device_list = [device if "cuda" in device else f"cuda:{device}" for device in self.device]
        cprint(f"Using devices: {self.device_list}", "blue")

        self.task_rank_allocation, self.rank_to_device = allocate_tasks_to_gpus(task_names, self.device_list)

        tasks_per_gpu = {}
        for task, gpu_rank in self.task_rank_allocation.items():
            tasks_per_gpu[gpu_rank] = tasks_per_gpu.get(gpu_rank, 0) + 1

        cprint("\nDevice Mapping and Load Distribution (2D):", "blue")
        for rank, dev in self.rank_to_device.items():
            task_count = tasks_per_gpu.get(rank, 0)
            cprint(f"GPU {dev} (Rank {rank}) -> {task_count} tasks", "blue")

        cprint("\nDetailed Task Allocation (2D):", "green")
        for task, rank in self.task_rank_allocation.items():
            actual_device = self.rank_to_device[rank]
            cprint(f"Task: {task} -> GPU {actual_device} (Rank {rank})", "green")

    def create_env_runner(self, task_name: str, rank: int) -> MetaworldRunner2D:
        try:
            device = f"cuda:{rank}"
            runner = MetaworldRunner2D(
                output_dir=self.output_dir,
                task_name=task_name,
                device=device,
                eval_episodes=self.eval_episodes,
                max_steps=self.max_steps,
                n_obs_steps=self.n_obs_steps,
                n_action_steps=self.n_action_steps,
                fps=self.fps,
                crf=self.crf,
                render_size=self.render_size,
                n_envs=self.n_envs,
                n_train=self.n_train,
                n_test=self.n_test,
                tqdm_interval_sec=self.tqdm_interval_sec,
                image_size=self.image_size,
                camera_name=self.camera_name,
            )
            return runner
        except Exception as e:
            print(f"Error creating 2D runner for task {task_name}: {str(e)}")
            return None

    def _run_single_task(self, task_name: str, policy: BasePolicy, rank: int, save_video: bool, results_queue: mp.Queue):
        try:
            device = f"cuda:{rank}"
            torch.cuda.set_device(device)

            runner = self.create_env_runner(task_name, rank)
            if runner is None:
                raise Exception("Failed to create runner")

            with torch.cuda.device(device):
                policy_cpu = copy.deepcopy(policy.to("cpu"))
                local_policy = policy_cpu.to(device)
                del policy_cpu
                torch.cuda.empty_cache()

            local_policy.eval()
            cprint(f"Task: {task_name} running on device {device} (2D)", "cyan")

            with torch.cuda.device(device):
                with torch.no_grad():
                    task_results = runner.run(local_policy, save_video)

            # Optional video persistence logic (same as 3D MP runner)
            if save_video and "sim_video_eval" in task_results:
                import os
                import shutil

                persistent_video_dir = os.path.join(self.output_dir, "eval_videos")
                os.makedirs(persistent_video_dir, exist_ok=True)

                tmp_video_path = task_results["sim_video_eval"]._path
                video_filename = f"{task_name}_eval.mp4"
                persistent_video_path = os.path.join(persistent_video_dir, video_filename)

                if os.path.exists(tmp_video_path):
                    shutil.copy2(tmp_video_path, persistent_video_path)
                    task_results["sim_video_eval"]._path = persistent_video_path
                    cprint(f"Video copied to persistent location: {persistent_video_path}", "cyan")
                else:
                    cprint(f"Warning: Temporary video not found at {tmp_video_path}", "yellow")

            del local_policy
            torch.cuda.empty_cache()

            success_rate = float(np.mean(task_results.get("mean_success_rates", 0.0)))
            results_queue.put((task_name, task_results, success_rate))

        except Exception as e:
            print(f"Error running task {task_name} (2D): {str(e)}")
            import traceback

            traceback.print_exc()
            results_queue.put((task_name, None, 0.0))

    def run(self, policy: BasePolicy, save_video=True) -> Dict:
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass

        results = {}
        valid_results = 0
        avg_success_rate = 0.0
        results_queue = mp.Queue()

        tasks_by_gpu = {}
        for task_name, rank in self.task_rank_allocation.items():
            tasks_by_gpu.setdefault(rank, []).append(task_name)

        num_gpus = len(self.device)
        processes_per_gpu = self.max_processes // max(1, num_gpus)
        cprint(f"\nRunning with exactly {processes_per_gpu} concurrent processes per GPU (2D)", "blue")

        total_tasks = len(self.task_names)
        tasks_completed = 0
        task_pointers = {gpu: 0 for gpu in tasks_by_gpu.keys()}

        while tasks_completed < total_tasks:
            active_processes = []
            current_batch_tasks = []

            for gpu_rank, gpu_tasks in tasks_by_gpu.items():
                start_idx = task_pointers[gpu_rank]
                remaining_tasks = len(gpu_tasks) - start_idx
                if remaining_tasks <= 0:
                    continue

                n_processes = min(processes_per_gpu, remaining_tasks)
                end_idx = start_idx + n_processes
                current_gpu_tasks = gpu_tasks[start_idx:end_idx]

                for task_name in current_gpu_tasks:
                    p = mp.Process(
                        target=self._run_single_task,
                        args=(task_name, policy, gpu_rank, save_video, results_queue),
                    )
                    p.start()
                    active_processes.append(p)
                    current_batch_tasks.append(task_name)
                    cprint(
                        f"Started task: {task_name} on GPU {self.rank_to_device[gpu_rank]} (Rank {gpu_rank}) (2D)",
                        "cyan",
                    )

                task_pointers[gpu_rank] = end_idx

            if not active_processes:
                break

            for _ in tqdm.tqdm(
                range(len(active_processes)),
                desc=f"Evaluating tasks {tasks_completed + 1}-{tasks_completed + len(active_processes)}/{total_tasks} (2D)",
            ):
                task_name, task_results, success_rate = results_queue.get()
                if task_results is not None:
                    results[task_name] = task_results
                    avg_success_rate += float(success_rate)
                    valid_results += 1
                    gpu_rank = self.task_rank_allocation[task_name]
                    cprint(
                        f"Completed task: {task_name} on GPU {self.rank_to_device[gpu_rank]} with success rate: {success_rate:.2%}",
                        "green",
                    )

            for p in active_processes:
                p.join()

            tasks_completed += len(active_processes)

            cprint(f"\nCompleted batch with {len(active_processes)} tasks (2D):", "blue")
            for gpu_rank in tasks_by_gpu.keys():
                completed_in_batch = sum(1 for t in current_batch_tasks if self.task_rank_allocation[t] == gpu_rank)
                cprint(f"GPU {self.rank_to_device[gpu_rank]}: {completed_in_batch} tasks", "blue")

        if valid_results > 0:
            avg_success_rate /= valid_results

        results["average_success_rate"] = avg_success_rate
        cprint(f"\nCompleted {valid_results}/{total_tasks} tasks with average success rate: {avg_success_rate:.2%} (2D)", "green")
        return results


