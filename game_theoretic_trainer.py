#!/usr/bin/env python3
"""
game_theoretic_trainer.py

Train a single PPO agent (ego) against the GameTheoreticEnv opponent.
"""
import argparse
from typing import Tuple

import gymnasium as gym
import numpy as np
import torch

from ppo_trainer import PPOTrainer, seed_everything, MultiCategorical
from torch.distributions import Normal
from tqdm import trange

from loguru import logger


class PPOGameTrainer(PPOTrainer):
    def collect_rollout(self, max_steps: int = 2048):
        """Collect a rollout of experiences."""
        states = []
        actions = []
        rewards = []
        values = []
        log_probs = []
        dones = []

        state, info = self.env.reset()
        # state = state['state']  # Extract state from observation dict

        episode_rewards = []
        current_episode_reward = 0

        for _ in trange(max_steps, desc='Collect'):
            # Convert state to tensor
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

            # Get action from policy
            with torch.no_grad():
                if self.discrete:
                    logits = self.actor(state_tensor)
                    dist = MultiCategorical(logits, self.n_logits)
                else:
                    mean, log_std = self.actor(state_tensor)
                    std = log_std.exp()
                    dist = Normal(mean, std)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(dim=-1)
                value = self.critic(state_tensor)

            action = action.detach().cpu().numpy()[0]
            next_state, reward, terminated, truncated, info = self.env.step((action, dist))
            # next_state = next_state['state']  # Extract state from observation dict
            # done = terminated or truncated  # This is incorrect. Should use terminated.

            # Store experience
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            values.append(value.cpu().numpy()[0])
            log_probs.append(log_prob.cpu().numpy())
            # dones.append(done)
            dones.append(terminated)

            current_episode_reward += reward

            if terminated or truncated:  # Reset on either condition.
                episode_rewards.append(current_episode_reward)
                current_episode_reward = 0
                state, info = self.env.reset()
                # state = state['state']
            else:
                state = next_state

        # Get final value for GAE computation
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            final_value = self.critic(state_tensor).cpu().numpy()[0]

        # Log episode rewards
        if episode_rewards:
            self.writer.add_scalar('rollout/mean_episode_reward', np.mean(episode_rewards), self.episode_count)
            self.writer.add_scalar('rollout/max_episode_reward', np.max(episode_rewards), self.episode_count)
            self.writer.add_scalar('rollout/min_episode_reward', np.min(episode_rewards), self.episode_count)

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(values),
            np.array(log_probs),
            np.array(dones),
            final_value
        )

    def evaluate_agent(self):
        state, info = self.env.reset(options={'render': True})
        terminated, truncated = False, False
        min_rel_dist = np.inf
        episode_reward = 0

        with torch.no_grad():
            while not truncated and not terminated:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                if not self.discrete:
                    mean, log_std = self.actor(state_tensor)
                    dist = Normal(mean, log_std.exp())
                else:
                    logits = self.actor(state_tensor)
                    dist = MultiCategorical(logits, self.n_logits)
                action = dist.sample().cpu().numpy()[0]
                state, reward, terminated, truncated, info = self.env.step((action, dist))
                min_rel_dist = min(min_rel_dist, info['relative_distance'])
                episode_reward += reward

        # Log evaluation metrics
        self.writer.add_scalar('eval/episode_reward', episode_reward, self.episode_count)
        self.writer.add_scalar('eval/min_relative_distance', min_rel_dist, self.episode_count)

        if info['success']:
            self.success_count += 1
            logger.info(f"Successful overtaking! Episode {self.episode_count}")
            self.writer.add_scalar('eval/success', 1, self.episode_count)
        else:
            logger.info(f"Failed to overtake. Episode {self.episode_count}")
            logger.info(f"Min relative distance: {min_rel_dist}")
            self.writer.add_scalar('eval/success', 0, self.episode_count)
        logger.info(f"Success rate: {self.success_count}/{self.episode_count + 1}")

        # Log training time
        elapsed_time = time.time() - self.start_time
        self.writer.add_scalar('time/elapsed_seconds', elapsed_time, self.episode_count)
        self.writer.add_scalar('time/episodes_per_second', self.episode_count / elapsed_time, self.episode_count)


# from gym_carla.envs.barc.game_theoretic_env import GameTheoreticEnv

# def build_action_grid(action_space, num_bins_per_dim=5):
#     """Discretize a Box or MultiDiscrete into a list of actions."""
#     from gymnasium import spaces
#     if isinstance(action_space, spaces.Box):
#         dims = action_space.shape[0]
#         linspaces = [
#             np.linspace(action_space.low[i], action_space.high[i], num_bins_per_dim)
#             for i in range(dims)
#         ]
#         mesh = np.meshgrid(*linspaces, indexing='ij')
#         coords = np.stack([m.flatten() for m in mesh], axis=-1)
#         return [row for row in coords]
#     elif isinstance(action_space, spaces.MultiDiscrete):
#         grids = [np.arange(n) for n in action_space.nvec]
#         mesh = np.meshgrid(*grids, indexing='ij')
#         flat = np.stack([m.flatten() for m in mesh], axis=-1)
#         return [row for row in flat]
#     else:
#         raise NotImplementedError

# def opponent_reward_fn(obs, ego_action, opp_action):
#     """Front-car reward: maximize lead, maintain speed, penalize slow."""
#     rel = obs['relative_distance']
#     state = obs['state'].reshape(2, -1)
#     v_front = state[1, 0]
#     w_dist, w_speed, w_pen_slow, v_min = 1.0, 0.5, -1.0, 0.5
#     reward = w_dist * rel + w_speed * v_front
#     if v_front < v_min:
#         reward += w_pen_slow * (v_min - v_front)
#     return reward

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_epochs', type=int, default=1000)
    parser.add_argument('--max_steps', type=int, default=2048)
    parser.add_argument('-m', '--comment', default='game_theoretic')
    parser.add_argument('--sample_k', type=int, default=32)
    parser.add_argument('--bins', type=int, default=5)
    args = parser.parse_args()

    seed_everything(args.seed)

    env_name, track_name = "barc-v1-race", "L_track_barc"
    # build action grid
    # dummy = gym.make(env_name, opponent=None, track_name=track_name,
    #                  do_render=False, enable_camera=False, discrete_action=True)
    # action_grid = build_action_grid(dummy.action_space, num_bins_per_dim=args.bins)
    # dummy.close()

    # env with dual rewards
    gt_env = gym.make('barc-v2',  # GameTheoreticEnv(
                      track_name=track_name,
                      # ego_action_grid=action_grid,
                      # opp_action_grid=action_grid,
                      # opp_reward_fn=None,
                      sample_k=args.sample_k,
                      do_render=False,
                      enable_camera=False,
                      dt=0.1, dt_sim=0.01, max_steps=300
                      )

    # wrap so PPO sees only 'state' and ego reward
    class StateRewardWrapper(gym.Wrapper):
        def __init__(self, env):
            super().__init__(env)
            # expose only the 'state' observation
            # self.observation_space = env.observation_space.spaces
            # self.action_space = env.action_space

        def reset(self, **kwargs):
            obs_dict, info = self.env.reset(**kwargs)
            return obs_dict, info

        def step(self, action):
            obs_dict, rewards, done, trunc, info = self.env.step(action)
            ego_r, opp_r = rewards
            info['opp_reward'] = opp_r
            return obs_dict, ego_r, done, trunc, info

    env = StateRewardWrapper(gt_env)

    # ensure ppo_trainer module knows the env_name for PPOTrainer __init__
    # ppo_trainer.env_name = env_name
    trainer = PPOGameTrainer(env=env,
                             env_name='barc-v2',
                             model_name='ppo-game-theoretic',
                             comment=args.comment)
    try:
        trainer.train(num_iterations=args.n_epochs, max_steps=args.max_steps)
    finally:
        trainer.save_model(f'ppo_{args.comment}_latest.pth')
        trainer.close()


if __name__ == '__main__':
    main()
