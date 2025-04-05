"""
This code is extended from https://github.com/gkswamy98/fast_irl/blob/master/learners/filt.py
and follows largely the same structure. The main difference is the addition of a HybridReplayBuffer
that is used to train the SAC agent under the HyPE algorithm.
"""

import os
from pathlib import Path
from typing import Any, Dict

import hydra
import gym
import numpy as np
import omegaconf
import torch
from stable_baselines3.common.evaluation import evaluate_policy
from tqdm import tqdm

from garage.models.discriminator import Discriminator, DiscriminatorEnsemble
from garage.models.sac import SAC
from garage.utils.common import MF_LOG_FORMAT, PROJECT_ROOT, rollout_agent_in_real_env
from garage.utils.gym_wrappers import (
    GoalWrapper,
    ResetWrapper,
    RewardWrapper,
    TremblingHandWrapper,
)
from garage.utils.logger import Logger
from garage.utils.nn_utils import gradient_penalty, linear_schedule
from garage.utils.oadam import OAdam
from garage.utils.replay_buffer import HybridReplayBuffer, QReplayBuffer


def train(cfg: omegaconf.DictConfig, demos_dict: Dict[str, Any]) -> None:
    """
    Main training loop for model-free inverse reinforcement learning.

    Args:
        cfg (omegaconf.DictConfig): Configuration for the experiment.
        demos_dict (Dict[str, Any]): Dictionary containing the expert demonstrations.

    Returns:
        None
    """

    device = cfg.device
    env_name = cfg.overrides.env
    is_maze = "maze" in env_name

    # --------------- Wrap environment and init discriminator ---------------
    env = gym.make(cfg.overrides.env)
    eval_env = gym.make(cfg.overrides.env)
    if is_maze:
        env = GoalWrapper(env, demos_dict["goals"][0][0])
        eval_env = GoalWrapper(eval_env, demos_dict["goals"][0][0])
    # Wrapper to reset to expert states with probability=reset_prob.
    # In standard IRL, reset_prob=0.0. For details, see second point here:
    # https://github.com/jren03/garage/tree/main/garage/algorithms#model-free-inverse-reinforcement-learning
    env = ResetWrapper(
        env,
        demos_dict["qpos"],
        demos_dict["qvel"],
        demos_dict["goals"],
        demos_dict["traj_obs"],
        demos_dict["traj_actions"],
        demos_dict["traj_seeds"],
        reset_prob=cfg.algorithm.reset_prob,
    )

    discriminator_cfg = cfg.overrides.discriminator
    if discriminator_cfg.ensemble_size > 1:
        f_net = DiscriminatorEnsemble(
            env,
            ensemble_size=discriminator_cfg.ensemble_size,
            clip_output=discriminator_cfg.clip_output,
        )
    else:
        f_net = Discriminator(env, clip_output=discriminator_cfg.clip_output)
    f_net.to(device)
    f_opt = OAdam(f_net.parameters(), lr=discriminator_cfg.lr)
    env = RewardWrapper(env, f_net)
    env = TremblingHandWrapper(env, cfg.overrides.p_tremble)
    eval_env = TremblingHandWrapper(eval_env, cfg.overrides.p_tremble)

    # --------------- Initialize Agent ---------------
    if is_maze:
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        expert_buffer = QReplayBuffer(state_dim, action_dim)
        expert_buffer.add_d4rl_dataset(demos_dict["dataset"])
        learner_buffer = QReplayBuffer(state_dim, action_dim)
        agent = hydra.utils.instantiate(
            cfg.algorithm.td3_agent,
            env=env,
            expert_buffer=expert_buffer,
            learner_buffer=learner_buffer,
            discriminator=f_net,
            cfg=cfg,
        )
        agent.learn(total_timesteps=cfg.algorithm.bc_init_steps, bc=True)
    else:
        sac_agent_cfg = cfg.algorithm.sac_agent
        agent = SAC(
            env=env,
            discriminator=f_net,
            learning_rate=linear_schedule(7.3e-4),
            bc_reg=sac_agent_cfg.bc_reg,
            bc_weight=sac_agent_cfg.bc_weight,
            policy=sac_agent_cfg.policy,
            verbose=sac_agent_cfg.verbose,
            ent_coef=sac_agent_cfg.ent_coef,
            train_freq=sac_agent_cfg.train_freq,
            gradient_steps=sac_agent_cfg.gradient_steps,
            gamma=sac_agent_cfg.gamma,
            tau=sac_agent_cfg.tau,
            device=device,
        )
        # Initialize replay buffer that can sample from both expert and learner data.
        # In standard IRL, only learner data is sampled from, so hybrid_sampling=False
        agent.replay_buffer = HybridReplayBuffer(
            buffer_size=agent.buffer_size,
            observation_space=agent.observation_space,
            action_space=agent.action_space,
            device=agent.device,
            n_envs=1,
            optimize_memory_usage=agent.optimize_memory_usage,
            expert_data=demos_dict["dataset"],
            hybrid_sampling=cfg.algorithm.hybrid_sampling,
            sampling_schedule=cfg.algorithm.sampling_schedule
            if cfg.algorithm.hybrid_sampling
            else None,
        )
        agent.actor.optimizer = OAdam(agent.actor.parameters())
        agent.critic.optimizer = OAdam(agent.critic.parameters())

    # --------------- Logging ---------------
    work_dir = os.getcwd()
    logger = Logger(work_dir, cfg)
    log_name = f"{cfg.algorithm.name}_{env_name}"
    logger.register_group(
        log_name,
        MF_LOG_FORMAT,
        color="green",
    )
    save_path = Path(
        PROJECT_ROOT,
        "garage",
        "experiment_results",
        env_name,
        f"{cfg.algorithm.name}_{cfg.seed}.npz",
    )
    save_path.parent.mkdir(exist_ok=True, parents=True)

    # ----------------- Train -----------------
    disc_steps = 0
    env_steps = 0
    mean_rewards, std_rewards = [], []
    kl_divs = []  # Add list for storing KL divergence history
    total_env_steps = cfg.algorithm.total_env_steps
    agent_train_steps = discriminator_cfg.train_every
    expert_sa_pairs = demos_dict["expert_sa_pairs"].to(device)
    tbar = tqdm(range(total_env_steps), ncols=0, desc="Env Interaction Steps")
    while env_steps < total_env_steps:
        if not disc_steps == 0:
            disc_lr = discriminator_cfg.lr / disc_steps
        else:
            disc_lr = discriminator_cfg.lr
        f_opt = OAdam(f_net.parameters(), lr=disc_lr)

        # --------------- Agent Training -----------------
        agent.learn(total_timesteps=agent_train_steps)
        env_steps += agent_train_steps
        tbar.update(agent_train_steps)

        # Update discriminator on data from the current policy.
        curr_states, curr_actions, _ = rollout_agent_in_real_env(
            env, agent, discriminator_cfg.num_sample_trajectories
        )
        learner_sa_pairs = torch.cat((curr_states, curr_actions), dim=1).to(device)

        # --------------- Discriminator Training ---------------
        for _ in range(discriminator_cfg.num_update_steps):
            learner_sa = learner_sa_pairs[
                np.random.choice(len(learner_sa_pairs), discriminator_cfg.batch_size)
            ]
            expert_sa = expert_sa_pairs[
                np.random.choice(len(expert_sa_pairs), discriminator_cfg.batch_size)
            ]
            f_opt.zero_grad()
            f_learner = f_net(learner_sa.float())
            f_expert = f_net(expert_sa.float())
            gp = gradient_penalty(learner_sa, expert_sa, f_net)
            loss = f_expert.mean() - f_learner.mean() + 10 * gp
            loss.backward()
            f_opt.step()
        disc_steps += 1

        if env_steps % cfg.overrides.eval_frequency == 0:
            if is_maze:
                mean_reward, std_reward = evaluate_policy(
                    agent, eval_env, n_eval_episodes=25
                )
                mean_reward = mean_reward * 100
                std_reward = std_reward * 100
            else:
                mean_reward, std_reward = evaluate_policy(
                    agent, eval_env, n_eval_episodes=10
                )
            mean_rewards.append(mean_reward)
            std_rewards.append(std_reward)
            
            # -------- Calculate KL divergence --------
            # Collect samples of expert and current policy action distributions
            kl_states = expert_sa_pairs[:1000, :env.observation_space.shape[0]].to(device)  # Use 1000 expert states
            
            # Get expert policy action distribution parameters
            # Note: Use expert demo actions as distribution center, with fixed variance
            expert_actions = expert_sa_pairs[:1000, env.observation_space.shape[0]:].to(device)
            expert_mean = expert_actions
            expert_std = torch.ones_like(expert_mean) * 0.1  # Assume fixed standard deviation
            
            # Get current policy action distribution parameters
            learner_actions = []
            learner_means = []
            learner_stds = []
            
            # Set evaluation mode
            agent.actor.eval()
            
            with torch.no_grad():
                for state in kl_states:
                    # Get policy action distribution
                    if hasattr(agent.actor, "get_distribution"):
                        # If actor has a method to get distribution
                        action_dist = agent.actor.get_distribution(state.unsqueeze(0))
                        mean = action_dist.mean.squeeze(0)
                        std = action_dist.stddev.squeeze(0)
                    else:
                        # If not, use predict method and assume fixed variance
                        action, _ = agent.predict(state.cpu().numpy(), deterministic=False)
                        action = torch.tensor(action, device=device)
                        mean = torch.tensor(action, device=device)
                        std = torch.ones_like(mean) * 0.1
                    
                    learner_means.append(mean)
                    learner_stds.append(std)
                    learner_actions.append(action)
            
            # Restore training mode
            agent.actor.train()
            
            # Convert lists to tensors
            learner_means = torch.stack(learner_means)
            learner_stds = torch.stack(learner_stds)
            
            # Calculate KL divergence (from expert to learner)
            # KL(p||q) = log(σ2/σ1) + (σ1^2 + (μ1-μ2)^2)/(2σ2^2) - 1/2
            variance_ratio = (learner_stds / expert_std).pow(2)
            mean_diff_squared = (expert_mean - learner_means).pow(2)
            kl_div = (torch.log(variance_ratio) + 
                      (1/variance_ratio) * (1 + mean_diff_squared/(expert_std.pow(2))) - 1)
            kl_div = 0.5 * kl_div.sum(dim=1).mean()  # Average KL across all dimensions and samples
            
            # Record KL divergence
            kl_divs.append(kl_div.item())
            
            logger.log_data(
                log_name,
                {
                    "env_steps": env_steps,
                    "mean_reward": mean_reward,
                    "std_reward": std_reward,
                    "kl_divergence": kl_div.item(),  # Add KL divergence
                },
            )
            eval_steps = list(range(cfg.overrides.eval_frequency, env_steps + 1, cfg.overrides.eval_frequency))
            # Save results
            np.savez(
                str(save_path),
                means=mean_rewards,
                stds=std_rewards,
                kl_divs=kl_divs,  # Add KL divergence history
                steps=eval_steps
            )

    # ------------- Save results -------------
    print(f"Results saved to {save_path}")
