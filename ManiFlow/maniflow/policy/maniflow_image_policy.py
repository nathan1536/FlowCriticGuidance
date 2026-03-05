from typing import Dict, Tuple
import torch
import torch.nn.functional as F
from einops import reduce
from termcolor import cprint

from maniflow.model.common.normalizer import LinearNormalizer
from maniflow.policy.base_policy import BasePolicy
from maniflow.common.pytorch_util import dict_apply
from maniflow.common.model_util import print_params
from maniflow.model.vision_2d.timm_obs_encoder import TimmObsEncoder
from maniflow.model.diffusion.ditx import DiTX
from maniflow.model.common.sample_util import *

class ManiFlowTransformerImagePolicy(BasePolicy):
    def __init__(self, 
             shape_meta: dict,
            horizon, 
            n_action_steps, 
            n_obs_steps,
            num_inference_steps=None,
            obs_as_global_cond=True,
            diffusion_timestep_embed_dim=256,
            diffusion_target_t_embed_dim=256,
            visual_cond_len=1024,
            n_layer=3,
            n_head=4,
            n_emb=256,
            qkv_bias=False,
            qk_norm=False,
            block_type="DiTX",
            obs_encoder: TimmObsEncoder = None,
            language_conditioned=False,
            # consistency flow training parameters
            flow_batch_ratio=0.75,
            consistency_batch_ratio=0.25,
            denoise_timesteps=10,
            sample_t_mode_flow="beta", 
            sample_t_mode_consistency="discrete",
            sample_dt_mode_consistency="uniform", 
            sample_target_t_mode="relative", # relative, absolute
            **kwargs):
        super().__init__()

        # parse shape_meta
        action_shape = shape_meta['action']['shape']
        self.action_shape = action_shape
        if len(action_shape) == 1:
            action_dim = action_shape[0]
        elif len(action_shape) == 2: # use multiple hands
            action_dim = action_shape[0] * action_shape[1]
        else:
            raise NotImplementedError(f"Unsupported action shape {action_shape}")
            
        obs_shape_meta = shape_meta['obs']
        obs_dict = dict_apply(obs_shape_meta, lambda x: x['shape'])
        
        # create ManiFlow model
        obs_feature_dim = obs_encoder.output_shape()[-1]
        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None
        if obs_as_global_cond:
            input_dim = action_dim
            global_cond_dim = obs_feature_dim

     
        cprint(f"[ManiFlowTransformerPointcloudPolicy] Using DiTX model", "red")
        model = DiTX(
            input_dim=input_dim,
            output_dim=action_dim,
            horizon=horizon,
            n_obs_steps=n_obs_steps,
            cond_dim=global_cond_dim,
            visual_cond_len=visual_cond_len,
            diffusion_timestep_embed_dim=diffusion_timestep_embed_dim,
            diffusion_target_t_embed_dim=diffusion_target_t_embed_dim,
            n_layer=n_layer,
            n_head=n_head,
            n_emb=n_emb,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            block_type=block_type,
            language_conditioned=language_conditioned,
        )
        
        self.obs_encoder = obs_encoder
        self.model = model
        
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.language_conditioned = language_conditioned
        self.kwargs = kwargs

        self.num_inference_steps = num_inference_steps
        self.flow_batch_ratio = flow_batch_ratio
        self.consistency_batch_ratio = consistency_batch_ratio
        assert flow_batch_ratio + consistency_batch_ratio == 1.0, "Sum of batch ratios should be equal to 1.0"
        self.denoise_timesteps = denoise_timesteps
        self.sample_t_mode_flow = sample_t_mode_flow
        self.sample_t_mode_consistency = sample_t_mode_consistency
        self.sample_dt_mode_consistency = sample_dt_mode_consistency
        self.sample_target_t_mode = sample_target_t_mode
        assert self.sample_target_t_mode in ["absolute", "relative"], "sample_target_t_mode must be either 'absolute' or 'relative'"
        
        cprint(f"[ManiFlowTransformerImagePolicy] Initialized with parameters:", "yellow")
        cprint(f"  - horizon: {self.horizon}", "yellow")
        cprint(f"  - n_action_steps: {self.n_action_steps}", "yellow")
        cprint(f"  - n_obs_steps: {self.n_obs_steps}", "yellow")
        cprint(f"  - num_inference_steps: {self.num_inference_steps}", "yellow")
        cprint(f"  - flow_batch_ratio: {self.flow_batch_ratio}", "yellow")
        cprint(f"  - consistency_batch_ratio: {self.consistency_batch_ratio}", "yellow")
        cprint(f"  - denoise_timesteps: {self.denoise_timesteps}", "yellow")
        cprint(f"  - sample_t_mode_flow: {self.sample_t_mode_flow}", "yellow")
        cprint(f"  - sample_t_mode_consistency: {self.sample_t_mode_consistency}", "yellow")
        cprint(f"  - sample_dt_mode_consistency: {self.sample_dt_mode_consistency}", "yellow")
        cprint(f"  - sample_target_t_mode: {self.sample_target_t_mode}", "yellow")

        print_params(self)
        
    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, 
            vis_cond=None,
            lang_cond=None,
            **kwargs
            ):
        
        noise = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=None)
        
        ode_traj = self.sample_ode(
            x0 = noise, 
            N = self.num_inference_steps,
            vis_cond=vis_cond,
            lang_cond=lang_cond,
           **kwargs)
        
        return ode_traj[-1] # sample ode returns the whole traj, return the last one
    


    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        vis_cond = None
        lang_cond = None

        if self.language_conditioned:
            # assume nobs has 'task_name' key for language condition
            lang_cond = nobs.get('task_name', None)
            assert lang_cond is not None, "Language goal is required"

        # condition through visual feature
        this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].to(device))
        nobs_features = self.obs_encoder(this_nobs).to(device) 
        vis_cond = nobs_features.reshape(B, -1, Do) # B, self.n_obs_steps*L, Do
        # empty data for action
        cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)

        # run sampling
        nsample = self.conditional_sample(
            cond_data, 
            vis_cond=vis_cond,
            lang_cond=lang_cond,
            **self.kwargs)
        
        # unnormalize prediction
        naction_pred = nsample[...,:Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:,start:end]
        
        # get prediction
        result = {
            'action': action,
            'action_pred': action_pred,
        }
        
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def get_optimizer(
            self, 
            lr: float,
            weight_decay: float,
            obs_encoder_lr: float = None,
            obs_encoder_weight_decay: float = None,
            betas: Tuple[float, float] = (0.9, 0.95)
        ) -> torch.optim.Optimizer:
        optim_groups = self.model.get_optim_groups(
            weight_decay=weight_decay)
        
        backbone_params = list()
        other_obs_params = list()
        if obs_encoder_lr is not None:
            cprint(f"[ManiFlowTransformerImagePolicy] Use different lr for obs_encoder: {obs_encoder_lr}", "yellow")
            for key, value in self.obs_encoder.named_parameters():
                if key.startswith('key_model_map'):
                    backbone_params.append(value)
                else:
                    other_obs_params.append(value)
            optim_groups.append({
                "params": backbone_params,
                "weight_decay": obs_encoder_weight_decay,
                "lr": obs_encoder_lr # for fine tuning
            })
            optim_groups.append({
                "params": other_obs_params,
                "weight_decay": obs_encoder_weight_decay
            })
        optimizer = torch.optim.AdamW(
            optim_groups, lr=lr, betas=betas
        )
        return optimizer
    
    def sample_t(self, batch_size, mode="uniform"):
        """
        Sample t for flow matching or consistency training.
        """
        if mode == "uniform":
            t = torch.rand((batch_size,), device=self.device)
        elif mode == "lognorm":
            t = sample_logit_normal(batch_size, m=self.lognorm_m, s=self.lognorm_s, device=self.device)
        elif mode == "mode":
            t = sample_mode(batch_size, s=self.mode_s, device=self.device)
        elif mode == "cosmap":
            t = sample_cosmap(batch_size, device=self.device)
        elif mode == "beta":
            t = sample_beta(batch_size, device=self.device)
        elif mode == "discrete":
            t = torch.randint(low=0, high=self.denoise_timesteps, size=(batch_size,)).float()
            t = t / self.denoise_timesteps
        else:
            raise ValueError(f" Unsupported sample_t_mode {mode}. Choose from 'uniform', 'lognorm', 'mode', 'cosmap', 'beta', 'discrete'.")
        return t

    def sample_dt(self, batch_size, sample_dt_mode="uniform"):
        """
        Sample dt for consistency training.
        """
        if sample_dt_mode == "uniform":
            dt = torch.rand((batch_size,), device=self.device)
        else:
            raise ValueError(f"Unsupported sample_dt_mode {sample_dt_mode}")
        
        return dt
    
    def linear_interpolate(self, noise, target, timestep, epsilon=0.0):
        """
        Linear interpolation between noise and target data with optional noise preservation.
        
        Args:
            noise (Tensor): Initial noise at t=0
            target (Tensor): Target data point at t=1  
            timestep (float): Interpolation parameter in [0, 1]
                            t=0 returns pure noise, t=1 returns target + epsilon*noise
            epsilon (float): Noise preservation factor. Controls minimum noise retained.
                            Default 0.0 for standard linear interpolation.
                            
        Returns:
            Tensor: Interpolated data point at given timestep
            
        Examples:
            >>> # Standard linear interpolation (epsilon=0)
            >>> result = linear_interpolate(noise, data, 0.5)  # 50% noise + 50% data
            
            >>> # With noise preservation (epsilon=0.01) 
            >>> result = linear_interpolate(noise, data, 1.0, epsilon=0.01)  # data + 1% noise
        """
        # Calculate noise coefficient with epsilon adjustment
        noise_coeff = 1.0 - (1.0 - epsilon) * timestep
        
        # Linear combination: preserved_noise + scaled_target
        interpolated_data_point = noise_coeff * noise + timestep * target
        
        return interpolated_data_point

    
    def get_flow_velocity(self, actions, **model_kwargs):
        """
        Get flow velocity targets for training.
        Flow training is used to train the model to predict instantaneous velocity given a timestep.
        """
        target_dict = {}
        
        # get visual and language conditions
        vis_cond = model_kwargs.get('vis_cond', None)
        lang_cond = model_kwargs.get('lang_cond', None)
        flow_batchsize = actions.shape[0]
        device = actions.device
        
        # sample t and dt for flow
        # dt is zero for flow, as we aim to predict the instantaneous velocity at t
        t_flow = self.sample_t(flow_batchsize, mode=self.sample_t_mode_flow).to(device)
        t_flow = t_flow.view(-1, 1, 1)
        dt_flow = torch.zeros((flow_batchsize,), device=device)
        
        # get target timestep
        # target_t_flow is the target timestep for the flow step
        # it can be either absolute or relative to t_flow
        # if absolute, it is t_flow + dt_flow
        # if relative, it is just dt_flow
        if self.sample_target_t_mode == "absolute":
            target_t_flow = t_flow.squeeze() + dt_flow
        elif self.sample_target_t_mode == "relative":
            target_t_flow = dt_flow
        
        # compute interpolated data points at t and predict flow velocity
        x_0_flow = torch.randn_like(actions, device=device) 
        x_1_flow = actions.to(device) 
        x_t_flow = self.linear_interpolate(x_0_flow, x_1_flow, t_flow, epsilon=0.0)
        v_t_flow = x_1_flow - x_0_flow

        target_dict['x_t'] = x_t_flow
        target_dict['t'] = t_flow
        target_dict['target_t'] = target_t_flow
        target_dict['v_target'] = v_t_flow
        target_dict['vis_cond'] = vis_cond
        target_dict['lang_cond'] = lang_cond

        return target_dict
    
    def get_consistency_velocity(self, actions, **model_kwargs):
        """
        Get consistency velocity targets for training.
        Consistency training is used to train the model to be consistent across different timesteps.
        """
        target_dict = {}
        
        # get visual and language conditions
        vis_cond = model_kwargs.get('vis_cond', None)
        lang_cond = model_kwargs.get('lang_cond', None)
        ema_model = model_kwargs.get('ema_model', None)
        consistency_batchsize = actions.shape[0]
        device = actions.device

        # sample t and dt for consistency training
        t_ct = self.sample_t(consistency_batchsize, mode=self.sample_t_mode_consistency).to(device)
        t_ct = t_ct.view(-1, 1, 1)
        delta_t1 = self.sample_dt(consistency_batchsize, sample_dt_mode=self.sample_dt_mode_consistency).to(device)
        # delta_t2 = self.sample_dt(consistency_batchsize, sample_dt_mode=self.sample_dt_mode_consistency).to(device)
        delta_t2 = delta_t1.clone() # use the same delta_t or resample a new one

        # compute next timestep
        t_next = t_ct.squeeze() + delta_t1
        t_next = torch.clamp(t_next, max=1.0) # clip t to ensure it does not exceed 1.0
        t_next = t_next.view(-1, 1, 1)
        
        # compute target timestep
        # target_t_next is the target timestep for the next step
        # it can be either absolute or relative to t_next
        # if absolute, it is t_next + delta_t2
        # if relative, it is just delta_t2
        if self.sample_target_t_mode == "absolute":
            target_t_next = t_next.squeeze() + delta_t2
        elif self.sample_target_t_mode == "relative":
            target_t_next = delta_t2

        # compute interpolated data points at timestep t and t_next
        x0_ct = torch.randn_like(actions, device=device) 
        x1_ct = actions.to(device) 
        x_t_ct = self.linear_interpolate(x0_ct, x1_ct, t_ct, epsilon=0.0)
        x_t_next = self.linear_interpolate(x0_ct, x1_ct, t_next, epsilon=0.0)

        # predict the average velocity from t_next toward next target (t_next + delta_t2)
        with torch.no_grad():
            v_avg_to_next_target = ema_model.model(
                sample=x_t_next, 
                timestep=t_next.squeeze(),
                target_t=target_t_next.squeeze(), 
                vis_cond=vis_cond[-consistency_batchsize:],
                lang_cond=lang_cond[-consistency_batchsize:] if lang_cond is not None else None,
            ) 
        # predict the target data point using the average velocity
        pred_x1_ct = x_t_next + (1 - t_next) * v_avg_to_next_target
        # estimate the velocity at t by using the predicted endpoint
        v_ct = (pred_x1_ct - x_t_ct) / (1 - t_ct)

        # target_t_ct is the target timestep for the current timestep t
        target_t_ct = delta_t1 if self.sample_target_t_mode == "relative" else t_next.squeeze()
        
        target_dict['x_t'] = x_t_ct
        target_dict['t'] = t_ct
        target_dict['target_t'] = target_t_ct
        target_dict['v_target'] = v_ct

        return target_dict
    
    @torch.no_grad()
    def sample_ode(self, x0=None, N=None, **model_kwargs):
        ### NOTE: Use Euler method to sample from the learned flow
        if N is None:
            N = self.num_inference_steps
        dt = 1./N
        traj = [] # to store the trajectory
        x = x0.detach().clone()
        batchsize = x.shape[0]

        t = torch.arange(0, N, device=x0.device, dtype=x0.dtype) / N
        traj.append(x.detach().clone())

        for i in range(N):
            ti = torch.ones((batchsize,), device=self.device) * t[i]
            if self.sample_target_t_mode == "absolute":
                target_t = ti + dt
            elif self.sample_target_t_mode == "relative":
                target_t = dt
            pred = self.model(x, ti, target_t=target_t, **model_kwargs)
            x = x.detach().clone() + pred * dt
            traj.append(x.detach().clone())

        return traj

    def few_step_sample_for_training(self, noise, vis_cond, lang_cond=None, num_steps=4):
        """
        Run few Euler steps with FULL gradient flow via gradient checkpointing.
        Memory cost ≈ same as no_grad version (recomputes forward during backward).
        """
        x = noise
        dt = 1.0 / num_steps
        batch_size = x.shape[0]
        
        for i in range(num_steps):
            t = torch.ones(batch_size, device=self.device) * (i / num_steps)
            if self.sample_target_t_mode == "absolute":
                target_t = t + dt
            elif self.sample_target_t_mode == "relative":
                target_t = torch.ones(batch_size, device=self.device) * dt
            
            # Wrap each Euler step in gradient checkpointing:
            # - Forward: runs normally, but intermediate activations are NOT stored
            # - Backward: recomputes the forward pass to get activations, then computes grads
            x = torch.utils.checkpoint.checkpoint(
                self._euler_step, x, t, target_t, vis_cond, lang_cond, dt,
                use_reentrant=False
            )
        
        return x

    def _euler_step(self, x, t, target_t, vis_cond, lang_cond, dt):
        """Single Euler step (factored out for checkpointing)."""
        v = self.model(x, t, target_t=target_t, vis_cond=vis_cond, lang_cond=lang_cond)
        return x + v * dt

    def compute_acgd_loss(
            self,
            batch,
            vis_cond,
            lang_cond,
            critics: dict,
            num_steps: int = 4,
            chunked_critic: bool = False,
    ):
        """
        Compute ACGD distillation loss: -Q(s, π_θ(s,o,t))
        Optimized with vectorization over time and task batching.

        When chunked_critic=True, critics expect Q(s_t, [a_t, a_{t+1}, ..., a_{t+k-1}])
        instead of single-step Q(s_t, a_t).
        """
        batch_size = vis_cond.shape[0]
        device = self.device

        # 1. Sample noise and generate student actions (same as before)
        noise = torch.randn(
            batch_size, self.horizon, self.action_dim,
            device=device, dtype=vis_cond.dtype
        )

        student_action_norm = self.few_step_sample_for_training(
            noise, vis_cond, lang_cond, num_steps=num_steps
        )

        student_action_raw = self.normalizer['action'].unnormalize(student_action_norm)
        student_action_raw = student_action_raw.clamp(-1.0, 1.0)
        student_action_raw = student_action_raw.to(device)

        # 2. Prepare data slices (Vectorized slicing)
        # Use the timestep that will actually be executed at inference
        exec_start = self.n_obs_steps - 1
        exec_end = exec_start + self.n_action_steps


        limit = batch['obs']['full_state'].shape[1]
        actual_end = min(exec_end, limit)

        relevant_states = batch['obs']['full_state'][:, exec_start:actual_end].to(device)
        relevant_student_actions = student_action_raw[:, exec_start:actual_end]
        relevant_expert_actions = batch['action'][:, exec_start:actual_end].to(device)

        task_names = batch['obs']['task_name']

        # 3. Group by Task to batch critic calls
        # We process unique tasks one by one, but batch all samples for that task
        unique_tasks = set(task_names)

        per_task_losses = []
        all_q_student = []
        all_q_expert = []

        for task_name in unique_tasks:
            critic = critics.get(task_name)
            if critic is None:
                continue

            # Find all batch indices belonging to this task
            task_indices = [i for i, t in enumerate(task_names) if t == task_name]
            task_indices_tensor = torch.tensor(task_indices, device=device)

            # Gather data for this task: (B_task, T_slice, Dim)
            task_states = relevant_states[task_indices_tensor]
            task_stud_acts = relevant_student_actions[task_indices_tensor]
            task_exp_acts = relevant_expert_actions[task_indices_tensor]

            if chunked_critic:
                # Chunked critic: Q(s_t, [a_t, a_{t+1}, ..., a_{t+k-1}])
                # Take first state, flatten action chunk
                sa_student = torch.cat([task_states[:, 0], task_stud_acts.reshape(task_stud_acts.shape[0], -1)], dim=-1)
                sa_expert = torch.cat([task_states[:, 0], task_exp_acts.reshape(task_exp_acts.shape[0], -1)], dim=-1)
            else:
                # Single-step critic: Q(s_t, a_t) — original path
                sa_student = torch.cat([task_states, task_stud_acts], dim=-1)
                sa_expert = torch.cat([task_states, task_exp_acts], dim=-1)
            
            # # Prepare inputs
            # sa_student = torch.cat([flat_states, flat_stud_acts], dim=-1)
            # sa_expert = torch.cat([flat_states, flat_exp_acts], dim=-1)
            
            # 4. Batched Critic Evaluation
            # Student Q (keep gradients)
            q_student_per_sample = critic(sa_student).mean(-1)

            # Expert Q (no gradients, for logging only)
            with torch.no_grad():
                q_expert_per_sample = critic(sa_expert).mean(-1)

            # 5. Compute Stats & Normalization
            #   α = η / E|Q|,  L_q = -α · E[Q(s, a_student)]
            # Scale-invariant across tasks and training stages.
            # η is hvia acgd_lambda in the outer loss.
            q_abs_mean = q_student_per_sample.detach().abs().mean().clamp(min=1.0)
            task_loss = -q_student_per_sample.mean() / q_abs_mean

            # # --- Previous
            # expert_mean = q_expert_per_sample.mean()
            # expert_std = q_expert_per_sample.std(unbiased=False)
            # expert_std = torch.clamp(expert_std, min=1.0)
            #
            # per_sample_gap = (q_expert_per_sample.detach() - q_student_per_sample) / expert_std
            # task_loss = F.softplus(per_sample_gap, beta=2.0).mean()
            # # --- End 

            per_task_losses.append(task_loss)
            all_q_student.append(q_student_per_sample.mean().detach())
            all_q_expert.append(q_expert_per_sample.mean().detach())

            # 6. Final Aggregation
            if len(per_task_losses) > 0:
                loss_distill = torch.stack(per_task_losses).mean()
                q_student_mean = torch.stack(all_q_student).mean()
                q_expert_mean = torch.stack(all_q_expert).mean()
            else:
                loss_distill = torch.tensor(0.0, device=device, requires_grad=True)
                q_student_mean = torch.tensor(0.0, device=device)
                q_expert_mean = torch.tensor(0.0, device=device)

            q_gap = q_expert_mean - q_student_mean

        return loss_distill, q_student_mean, q_expert_mean, q_gap

    def compute_critic_weights(
        self,
        batch,
        critics: dict,
        temperature: float = 1.0,
        weight_min: float = 0.01,
        weight_max: float = 5.0,
        chunked_critic: bool = False,
        chunk_size: int = 3,
    ):
        """
        Compute advantage weights using per-task Q-critics.

        If pre-computed V(s) is available in batch['obs']['v_value'],
        uses proper advantage: A(s,a) = Q(s,a) - V(s).
        Otherwise falls back to batch-normalized Q.

        When chunked_critic=True, evaluates Q(s_0, [a_0, ..., a_{k-1}]).
        """
        state_for_critic = batch['obs']['full_state']  # (B, T, state_dim)
        actions = batch['action']               # (B, T, action_dim)
        task_names = batch['obs']['task_name']  # list of B task names
        has_v = 'v_value' in batch['obs']

        batch_size = actions.shape[0]
        device = actions.device

        # Build critic input
        states_0 = state_for_critic[:, 0].to(device)   # (B, state_dim)
        if chunked_critic:
            # Flatten first chunk_size actions: (B, chunk_size * action_dim)
            action_chunk = actions[:, :chunk_size].reshape(batch_size, -1).to(device)
            sa_all = torch.cat([states_0, action_chunk], dim=-1)
        else:
            actions_0 = actions[:, 0].to(device)            # (B, action_dim)
            sa_all = torch.cat([states_0, actions_0], dim=-1)  # (B, state_dim+action_dim)

        q_values = torch.zeros(batch_size, device=device)
        valid_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)

        with torch.no_grad():
            unique_tasks = set(task_names)
            for task_name in unique_tasks:
                critic = critics.get(task_name)
                if critic is None:
                    continue
                task_idx = [i for i, t in enumerate(task_names) if t == task_name]
                idx_t = torch.tensor(task_idx, device=device)
                q_batch = critic(sa_all[idx_t]).squeeze(-1)
                q_values[idx_t] = q_batch
                valid_mask[idx_t] = True

        if not valid_mask.any():
            neutral = torch.ones(batch_size, device=device)
            return neutral, None, None

        # Fill invalid entries with mean of valid Q
        if not valid_mask.all():
            q_values[~valid_mask] = q_values[valid_mask].mean()

        # Compute advantage
        if has_v:
            v_values = batch['obs']['v_value'][:, 0].to(device)  # (B,)
            advantage = q_values - v_values
            # Normalize by advantage std for stable weighting
            adv_std = advantage.std().clamp(min=1e-4)
            advantage = advantage / adv_std
        else:
            advantage = (q_values - q_values.mean()) / (q_values.std() + 1e-8)

        weights = torch.exp(advantage / temperature).clamp(min=weight_min, max=weight_max)

        return weights, q_values, advantage

    def compute_loss(self, batch, ema_model=None, critics=None, critic_cfg=None, **kwargs):
        # normalize input
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action']).to(self.device)

        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        # handle different ways of passing observation
        local_cond = None
        vis_cond = None
        trajectory = nactions
        cond_data = trajectory
        lang_cond = None
        ema_model = ema_model

        if self.language_conditioned:
            # we assume language condition is passed as 'task_name'
            lang_cond = nobs.get('task_name', None)
            assert lang_cond is not None, "Language goal is required"

        # reshape B, T, ... to B*T
        this_nobs = dict_apply(nobs, 
            lambda x: x[:,:self.n_obs_steps,...].to(self.device))
        nobs_features = self.obs_encoder(this_nobs)
        vis_cond = nobs_features.reshape(batch_size, -1, self.obs_feature_dim)
        
        # Parse critic config FIRST (needed for batch split decision)
        use_critics = critics is not None and len(critics) > 0
        use_acgd = critic_cfg.get("acgd_enabled", False) if critic_cfg else False
        use_awr = use_critics and not use_acgd  # AWR only if ACGD is disabled
        
        # Determine ACGD warmup state early (needed for batch split)
        acgd_active = False
        if use_acgd and use_critics:
            acgd_warmup_epochs = critic_cfg.get("acgd_warmup_epochs", 0)
            current_epoch = kwargs.get("epoch", 0)
            acgd_active = current_epoch >= acgd_warmup_epochs
        
        # Set batch split: disable consistency when ACGD is active
        if use_acgd and acgd_active:
            # Disable consistency loss — the model already has few-step capability
            # from warmup training. Keeping it on fights the ACGD gradient.
            flow_batchsize = batch_size
            consistency_batchsize = 0
        else:
            flow_batchsize = int(batch_size * self.flow_batch_ratio)
            consistency_batchsize = int(batch_size * self.consistency_batch_ratio)

        # AWR: Compute critic-based advantage weights if using AWR mode
        weights = None
        q_values_for_logging = None
        advantage_for_logging = None
        if use_awr:
            temperature = critic_cfg.get("temperature", 1.0) if critic_cfg else 1.0
            weight_min = critic_cfg.get("weight_min", 0.01) if critic_cfg else 0.01
            weight_max = critic_cfg.get("weight_max", 5.0) if critic_cfg else 5.0
            is_chunked = critic_cfg.get("chunked_critic", False)
            c_size = critic_cfg.get("chunk_size", 3)
            weights, q_values_for_logging, advantage_for_logging = self.compute_critic_weights(
                batch, critics, temperature, weight_min, weight_max,
                chunked_critic=is_chunked, chunk_size=c_size,
            )

        # Get flow targets
        flow_target_dict = self.get_flow_velocity(nactions[:flow_batchsize], 
                                                    vis_cond=vis_cond[:flow_batchsize],
                                                    lang_cond=lang_cond[:flow_batchsize] if lang_cond is not None else None)
        v_flow_pred = self.model(
            sample=flow_target_dict['x_t'], 
            timestep=flow_target_dict['t'].squeeze(),
            target_t=flow_target_dict['target_t'].squeeze(),
            vis_cond=vis_cond[:flow_batchsize],
            lang_cond=flow_target_dict['lang_cond'][:flow_batchsize] if lang_cond is not None else None)
        v_flow_pred_magnitude = torch.sqrt(torch.mean(v_flow_pred ** 2)).item()

        # Get consistency targets (SKIP when consistency is disabled)
        v_ct_pred_magnitude = 0.0
        loss_ct_scalar = 0.0
        consistency_target_dict = None
        
        if consistency_batchsize > 0:
            consistency_target_dict = self.get_consistency_velocity(
                nactions[flow_batchsize:flow_batchsize+consistency_batchsize],
                vis_cond=vis_cond[flow_batchsize:flow_batchsize+consistency_batchsize],
                lang_cond=lang_cond[flow_batchsize:flow_batchsize+consistency_batchsize] if lang_cond is not None else None,
                ema_model=ema_model
            )
            v_ct_pred = self.model(
                sample=consistency_target_dict['x_t'], 
                timestep=consistency_target_dict['t'].squeeze(),
                target_t=consistency_target_dict['target_t'].squeeze(),
                vis_cond=vis_cond[flow_batchsize:flow_batchsize+consistency_batchsize],
                lang_cond=lang_cond[flow_batchsize:flow_batchsize+consistency_batchsize] if lang_cond is not None else None,
            )
            v_ct_pred_magnitude = torch.sqrt(torch.mean(v_ct_pred ** 2)).item()

        """Compute losses"""
        loss_bc = 0.

        # compute flow loss 
        v_flow_target = flow_target_dict['v_target']
        loss_flow = F.mse_loss(v_flow_pred, v_flow_target, reduction='none')
        loss_flow = reduce(loss_flow, 'b ... -> b', 'mean')  # (flow_batchsize,) - fully reduced per sample
        
        # Apply AWR critic weights to flow loss (only in AWR mode)
        if use_awr and weights is not None:
            flow_weights = weights[:flow_batchsize]
            weighted_loss_flow = (flow_weights * loss_flow).sum() / flow_weights.sum()
            loss_bc += weighted_loss_flow
            loss_flow_scalar = weighted_loss_flow.item()
        else:
            loss_bc += loss_flow.mean()
            loss_flow_scalar = loss_flow.mean().item()

        # compute consistency training loss (SKIP when consistency is disabled)
        if consistency_batchsize > 0 and consistency_target_dict is not None:
            v_ct_target = consistency_target_dict['v_target']
            loss_ct = F.mse_loss(v_ct_pred, v_ct_target, reduction='none')
            loss_ct = reduce(loss_ct, 'b ... -> b', 'mean')
            
            # Apply AWR critic weights to consistency loss (only in AWR mode)
            if use_awr and weights is not None:
                ct_weights = weights[flow_batchsize:flow_batchsize+consistency_batchsize]
                weighted_loss_ct = (ct_weights * loss_ct).sum() / ct_weights.sum()
                loss_bc += weighted_loss_ct
                loss_ct_scalar = weighted_loss_ct.item()
            else:
                loss_bc += loss_ct.mean()
                loss_ct_scalar = loss_ct.mean().item()

        loss_bc = loss_bc.mean()
        
        # ACGD: Add distillation loss that maximizes Q of student's predicted actions
        loss_distill = torch.tensor(0.0, device=self.device)
        acgd_q_student = 0.0
        acgd_q_expert = 0.0
        acgd_q_gap = 0.0
        
        if use_acgd and use_critics and acgd_active:
            acgd_alpha = critic_cfg.get("acgd_alpha", 1.0)
            acgd_lambda = critic_cfg.get("acgd_lambda", 0.3)
            acgd_sample_steps = critic_cfg.get("acgd_sample_steps", 4)
            
            # Compute ACGD distillation loss: -Q(s, π_θ(s,o,t))
            is_chunked = critic_cfg.get("chunked_critic", False)
            loss_distill, acgd_q_student, acgd_q_expert, acgd_q_gap = self.compute_acgd_loss(
                batch, vis_cond, lang_cond, critics, num_steps=acgd_sample_steps,
                chunked_critic=is_chunked,
            )
            
            # Total loss: α * L_flow + λ * L_distill (no consistency when ACGD active)
            loss = acgd_alpha * loss_bc + acgd_lambda * loss_distill
        else:
            loss = loss_bc
        
        loss_dict = {
                'loss_flow': loss_flow_scalar,
                'loss_ct': loss_ct_scalar,
                'v_flow_pred_magnitude': v_flow_pred_magnitude,
                'v_ct_pred_magnitude': v_ct_pred_magnitude,
                'bc_loss': loss_bc.item(),
        }
        
        # Log AWR weight stats (only in AWR mode)
        if use_awr and weights is not None:
            loss_dict['critic_weight_mean'] = weights.mean().item()
            loss_dict['critic_weight_std'] = weights.std().item()
            loss_dict['critic_weight_min'] = weights.min().item()
            loss_dict['critic_weight_max'] = weights.max().item()
            
            # Log Q-value and advantage statistics
            if q_values_for_logging is not None:
                loss_dict['critic_q_mean'] = q_values_for_logging.mean().item()
                loss_dict['critic_q_std'] = q_values_for_logging.std().item()
            if advantage_for_logging is not None:
                loss_dict['critic_advantage_mean'] = advantage_for_logging.mean().item()
                loss_dict['critic_advantage_std'] = advantage_for_logging.std().item()
        
        # Log ACGD stats (only in ACGD mode)
        if use_acgd and use_critics:
            loss_dict['acgd_active'] = 1.0 if acgd_active else 0.0  # Whether ACGD is enabled (after warmup)
            if acgd_active:
                loss_dict['acgd_loss_distill'] = loss_distill.item()
                loss_dict['acgd_q_student'] = acgd_q_student      # Q(s, student_action)
                loss_dict['acgd_q_expert'] = acgd_q_expert        # Q(s, expert_action)
                loss_dict['acgd_q_gap'] = acgd_q_gap              # expert - student (positive = critic prefers expert)
                loss_dict['acgd_total_loss'] = loss.item()
            

        return loss, loss_dict
