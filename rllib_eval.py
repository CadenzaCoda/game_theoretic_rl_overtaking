import ray
# Initialize Ray
import os
import gymnasium as gym

from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.policy.policy import Policy
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env
import numpy as np
# Import your custom environment
from mpclab_common.track import get_track
from gym_carla.controllers.barc_pid import PIDWrapper
import gym_carla

import os
# from ray.tune.logger import UnifiedLogger
import torch

torch.autograd.set_detect_anomaly(True)

if torch.cuda.is_available():
    print("CUDA is available. Running on GPU.")
    print(f"Current device: {torch.cuda.get_device_name(0)}")
    print(f"Total GPUs available: {torch.cuda.device_count()}")
else:
    print("CUDA is not available. Running on CPU.")

def env_creator(config):
    print("Creating Environment")
    env_name = "barc-v1-race"
    track_name = "L_track_barc"
    opponent = PIDWrapper(dt=0.1, t0=0., track_obj=get_track(track_name))
    env = gym.make(env_name, opponent=opponent, track_name=track_name, do_render=True, enable_camera=False,
                   discrete_action=True)
    return env

register_env(name="my_barc", env_creator=env_creator)

# Define a function to instantiate the environment
CKPT_DIR = "/home/wilson/ray_results/PPO_my_barc_2025-05-05_15-59-40k8qe_whs/checkpoint_step_1000"
algo = Algorithm.from_checkpoint(CKPT_DIR)


results = algo.evaluate()                       # ← now uses the config above
print("Reward:", results["evaluation"]["episode_reward_mean"])
algo.stop()
ray.shutdown()

# env = env_creator(config=1)
# obs, info = env.reset(seed=0)
#
# done, truncated = False, False
# total_reward = 0.0
#
# # Inference‑only context (no grads)
# while not (done or truncated):
#     action = algo.compute_single_action(obs, explore=False)[0]
#     obs, reward, done, truncated, info = env.step(action)
#     total_reward += reward
# env.close()
# print("Episode reward:", total_reward)


