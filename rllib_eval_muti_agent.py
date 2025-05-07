import time

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
from gymnasium import spaces
# Import your custom environment
from ray.rllib.policy.policy import PolicySpec
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
    import gym_carla
    env = gym.make("barc-v1", track_name="L_track_barc", discrete_action=True, do_render=True)
    return env

register_env(name="barc_multi", env_creator=env_creator)

# observation_space = dict(
#             gps=spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
#             velocity=spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
#             state=spaces.Box(low=-np.inf, high=np.inf, shape=(2 * 9,), dtype=np.float32),  # Fixed to 2 vehicles
#         )
#
# observation_space = spaces.Dict(observation_space)
# _action_bounds = np.tile(np.array([2, 0.45]), [2, 1])
# action_space = spaces.Box(low=-_action_bounds, high=_action_bounds, dtype=np.float64)

observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2 * 9,), dtype=np.float32)
# observation_space = spaces.Dict(
#     {'ego': observation_space, 'oppo': observation_space}
# )
# Fixed action space for 2 vehicles
_action_bounds = np.tile(np.array([2, 0.45]), [2, 1])
u_a_space = np.linspace(-2, 2, 32, endpoint=True, dtype=np.float32)
u_steer_space = np.linspace(-0.45, 0.45, 32, endpoint=True, dtype=np.float32)  # Note: The choices are fixed for now. (10x10)
action_space = spaces.MultiDiscrete([len(u_a_space), len(u_steer_space)])

# action_space = spaces.Dict(
#     {'ego': action_space, 'oppo': action_space}
# )

policies = {
    "ego": PolicySpec(None, observation_space, action_space),
    "oppo": PolicySpec(None, observation_space, action_space),
}

def map_fn(agent_id, *_):
    return "ego" if agent_id == "ego" else "oppo"

config = (
    PPOConfig()
    .environment("barc_multi")
    .env_runners(
        num_env_runners=1,
        # num_cpus_per_env_runner=0,
        # num_gpus_per_env_runner=0.2,
        # rollout_fragment_length="auto",
        # batch_mode="complete_episodes",
        # sample_timeout_s=None,
        # observation_filter="MeanStdFilter"
    )
    .multi_agent(
        policies=policies,
        policy_mapping_fn=map_fn,
        policies_to_train=["ego", "oppo"]
    )
    .evaluation(
        evaluation_interval=1,  # Optional: for running evaluation episodes every N iterations
        evaluation_num_env_runners=0,
        evaluation_duration=1,                      # run exactly 1 episode
        evaluation_duration_unit="episodes",
        evaluation_config={
            "render_env": True,                     # call env.render()
            "create_env_on_driver": True,           # build eval env in driver
            "env_config": {"do_render": True},    # (if your env looks at this flag)
            # "explore": False
        }

    )

    .training(
        model={
            "fcnet_hiddens": [1024, 512, 256, 128],
            "vf_share_layers": False,
            "fcnet_activation": "relu"
               },

        train_batch_size=2048,  # Use a batch size that covers full episodes
        minibatch_size=1024,  # Adjust to a reasonable size based on your hardware

        lr=1e-4,
        entropy_coeff=[[0,0.01],[2048*1000,0.001],[2048*2000,0.0001]],
        gamma=0.99

    )
    .learners(
        num_learners=1,
        num_gpus_per_learner=1,
    )

    .framework("torch")  # or "tf2" if you prefer TensorFlow
)

# Build the algorithm
algo = config.build()

checkpoint_path = "/home/wilson/ray_results/PPO_barc_multi_2025-05-06_18-48-14dduhbo8j/checkpoint_step_3000"
algo.restore(checkpoint_path)

for i in range(50):
    print("Evaluation episode:", i+1)
    results = algo.evaluate()
    time.sleep(10)

print("Done")
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


