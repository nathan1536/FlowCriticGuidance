import wandb
import numpy as np
import torch
import tqdm
from maniflow.env import AdroitEnv
from maniflow.gym_util.multistep_wrapper import MultiStepWrapper
from maniflow.gym_util.video_recording_wrapper import SimpleVideoRecordingWrapper

from maniflow.policy.base_policy import BasePolicy
from maniflow.common.pytorch_util import dict_apply
from maniflow.env_runner.base_runner import BaseRunner
import maniflow.common.logger_util as logger_util
from termcolor import cprint


class AdroitRunner2D(BaseRunner):
    """
    2D image-based Adroit runner (no point cloud / no pytorch3d dependency).
    Uses AdroitEnv with use_point_cloud=False.
    """

    def __init__(self,
                 output_dir,
                 eval_episodes=20,
                 max_steps=200,
                 n_obs_steps=8,
                 n_action_steps=8,
                 fps=10,
                 crf=22,
                 render_size=84,
                 tqdm_interval_sec=5.0,
                 task_name=None,
                 ):
        super().__init__(output_dir)
        self.task_name = task_name

        def env_fn():
            return MultiStepWrapper(
                SimpleVideoRecordingWrapper(
                    AdroitEnv(env_name=task_name, use_point_cloud=False)
                ),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps,
                reward_agg_method='sum',
            )

        self.eval_episodes = eval_episodes
        self.env = env_fn()

        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec

        self.logger_util_test = logger_util.LargestKRecorder(K=3)
        self.logger_util_test10 = logger_util.LargestKRecorder(K=5)

    def run(self, policy: BasePolicy, save_video=True):
        device = policy.device
        env = self.env

        all_goal_achieved = []
        all_success_rates = []

        for episode_idx in tqdm.tqdm(range(self.eval_episodes),
                                     desc=f"Eval in Adroit {self.task_name} 2D Env",
                                     leave=False, mininterval=self.tqdm_interval_sec):
            obs = env.reset()
            policy.reset()

            done = False
            num_goal_achieved = 0
            while not done:
                np_obs_dict = dict(obs)
                obs_dict = dict_apply(np_obs_dict,
                                      lambda x: torch.from_numpy(x).to(device=device))

                with torch.no_grad():
                    obs_dict_input = {
                        'image': obs_dict['image'].unsqueeze(0),
                        'agent_pos': obs_dict['agent_pos'].unsqueeze(0),
                    }
                    action_dict = policy.predict_action(obs_dict_input)

                np_action_dict = dict_apply(action_dict,
                                            lambda x: x.detach().to('cpu').numpy())

                action = np_action_dict['action'].squeeze(0)
                obs, reward, done, info = env.step(action)
                num_goal_achieved += np.sum(info['goal_achieved'])
                done = np.all(done)

            all_success_rates.append(info['goal_achieved'])
            all_goal_achieved.append(num_goal_achieved)

        log_data = dict()
        log_data['mean_n_goal_achieved'] = np.mean(all_goal_achieved)
        log_data['mean_success_rates'] = np.mean(all_success_rates)
        log_data['test_mean_score'] = np.mean(all_success_rates)

        cprint(f"test_mean_score: {np.mean(all_success_rates)}", 'green')

        self.logger_util_test.record(np.mean(all_success_rates))
        self.logger_util_test10.record(np.mean(all_success_rates))
        log_data['SR_test_L3'] = self.logger_util_test.average_of_largest_K()
        log_data['SR_test_L5'] = self.logger_util_test10.average_of_largest_K()

        videos = env.env.get_video()
        if len(videos.shape) == 5:
            videos = videos[:, 0]
        if save_video:
            videos_wandb = wandb.Video(videos, fps=self.fps, format="mp4")
            log_data['sim_video_eval'] = videos_wandb

        _ = env.reset()
        return log_data
