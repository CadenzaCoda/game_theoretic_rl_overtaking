#!/usr/bin/env python3
"""
game_theoretic_trainer.py

Train a single PPO agent (ego) against the GameTheoreticEnv opponent.
"""
import argparse
import gymnasium as gym
import numpy as np
from ppo_trainer import PPOTrainer, seed_everything
import ppo_trainer
from gym_carla.envs.barc.game_theoretic_env import GameTheoreticEnv

def build_action_grid(action_space, num_bins_per_dim=5):
    """Discretize a Box or MultiDiscrete into a list of actions."""
    from gymnasium import spaces
    if isinstance(action_space, spaces.Box):
        dims = action_space.shape[0]
        linspaces = [
            np.linspace(action_space.low[i], action_space.high[i], num_bins_per_dim)
            for i in range(dims)
        ]
        mesh = np.meshgrid(*linspaces, indexing='ij')
        coords = np.stack([m.flatten() for m in mesh], axis=-1)
        return [row for row in coords]
    elif isinstance(action_space, spaces.MultiDiscrete):
        grids = [np.arange(n) for n in action_space.nvec]
        mesh = np.meshgrid(*grids, indexing='ij')
        flat = np.stack([m.flatten() for m in mesh], axis=-1)
        return [row for row in flat]
    else:
        raise NotImplementedError

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
    parser.add_argument('-m','--comment', default='game_theoretic')
    parser.add_argument('--sample_k', type=int, default=32)
    parser.add_argument('--bins', type=int, default=5)
    args = parser.parse_args()

    seed_everything(args.seed)

    env_name, track_name = "barc-v1-race", "L_track_barc"
    # build action grid
    dummy = gym.make(env_name, opponent=None, track_name=track_name,
                     do_render=False, enable_camera=False, discrete_action=True)
    action_grid = build_action_grid(dummy.action_space, num_bins_per_dim=args.bins)
    dummy.close()

    # env with dual rewards
    gt_env = GameTheoreticEnv(
        track_name=track_name,
        ego_action_grid=action_grid,
        opp_action_grid=action_grid,
        opp_reward_fn=None,
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
            self.observation_space = env.observation_space.spaces['state']
            self.action_space = env.action_space

        def reset(self, **kwargs):
            obs_dict, info = self.env.reset(**kwargs)
            return obs_dict['state'], info

        def step(self, action):
            obs_dict, rewards, done, trunc, info = self.env.step(action)
            ego_r, opp_r = rewards
            info['opp_reward'] = opp_r
            return obs_dict['state'], ego_r, done, trunc, info

    env = StateRewardWrapper(gt_env)

    # ensure ppo_trainer module knows the env_name for PPOTrainer __init__
    ppo_trainer.env_name = env_name
    trainer = PPOTrainer(env=env, model_name='ppo-game-theoretic', comment=args.comment)
    try:
        trainer.train(num_iterations=args.n_epochs, max_steps=args.max_steps)
    finally:
        trainer.save_model(f'ppo_{args.comment}_latest.pth')
        trainer.close()

if __name__ == '__main__':
    main()