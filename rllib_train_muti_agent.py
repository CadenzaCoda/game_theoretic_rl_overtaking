import ray
# Initialize Ray
ray.init(
    include_dashboard=True,
    dashboard_host="0.0.0.0"
)
import os
import gymnasium as gym
from gymnasium import spaces
from ray import train
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env
import numpy as np
# Import your custom environment
from mpclab_common.track import get_track
from gym_carla.controllers.barc_pid import PIDWrapper
import gym_carla
from ray.rllib.policy.policy import PolicySpec

import os
# from ray.tune.logger import UnifiedLogger
import torch
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.callbacks import DefaultCallbacks

torch.autograd.set_detect_anomaly(True)

if torch.cuda.is_available():
    print("CUDA is available. Running on GPU.")
    print(f"Current device: {torch.cuda.get_device_name(0)}")
    print(f"Total GPUs available: {torch.cuda.device_count()}")
else:
    print("CUDA is not available. Running on CPU.")

# Define a function to instantiate the environment

NUM_TRAINING_STEPS = 3000
CHECKPOINT_INTERVAL = 200

def env_creator(config):
    print("Creating Environment")
    import gym_carla
    env = gym.make("barc-v1", track_name="L_track_barc", discrete_action=True)
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
    "oppo": PolicySpec(None, observation_space,  action_space),
}

def map_fn(agent_id, *_):
    return "ego" if agent_id=="ego" else "oppo"

config = (
    PPOConfig()
    .environment("barc_multi")
    .env_runners(
        num_env_runners=10,  # Reduce the number of rollout workers
        # num_cpus_per_env_runner=0,
        # num_gpus_per_env_runner=0.2,
        # rollout_fragment_length="auto",  # Adjust fragment size for better GPU utilization
        # batch_mode="complete_episodes",  # Ensure full episodes are used
        # sample_timeout_s=None,  # Increase the sample timeout to avoid timeouts
        # observation_filter="MeanStdFilter"
    )
    .multi_agent(
        policies=policies,
        policy_mapping_fn=map_fn,
        policies_to_train=["ego", "oppo"]
    )

    # .evaluation(
    #     # evaluation_interval=10,  # Optional: for running evaluation episodes every N iterations
    #     evaluation_duration=10,                      # run exactly 1 episode
    #     evaluation_duration_unit="episodes",
    #     evaluation_config={
    #         "render_env": True,                     # call env.render()
    #         "create_env_on_driver": True,           # build eval env in driver
    #         "env_config": {"do_render": True},    # (if your env looks at this flag)
    #     }
    # )

    .training(
        model={
            "fcnet_hiddens": [1024, 512, 256, 128],
            "vf_share_layers": False,
            "fcnet_activation": "relu"
               },

        train_batch_size=2048,  # Use a batch size that covers full episodes
        minibatch_size=1024,  # Adjust to a reasonable size based on your hardware

        lr=1e-4,
        entropy_coeff=[[0, 0.01], [2048 * 1000, 0.001], [2048 * 2000, 0.0001]],
        # entropy_coeff=0.01,
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

# checkpoint_path = "/home/wilson/ray_results/PPO_barc_multi_2025-05-06_15-39-03r5yn0ze2/checkpoint_step_1000"
# algo = Algorithm.from_checkpoint(checkpoint_path)

# Train the algorithm for a number of iterations
for train_step in range(1, NUM_TRAINING_STEPS + 1):  # Adjust the number of iterations as needed

    result = algo.train()
    # print("\n#############⬇️  Iteration %d ⬇️ #############" % train_step)
    # print(pretty_print(result))
    # print("#############⬆️  Iteration %d ⬆️ #############\n" % train_step)

    print("#############  Iteration %d  #############\n" % train_step)

    # Call `save()` to create a checkpoint.
    if train_step % CHECKPOINT_INTERVAL == 0 or train_step == NUM_TRAINING_STEPS:
        save_result = algo.save(
            checkpoint_dir=os.path.join(
                algo.logdir,
                f"checkpoint_step_{train_step}"
            )
        )
        path_to_checkpoint = save_result.checkpoint.path
        # print(f"{save_result}")
        print(
            "An Algorithm checkpoint has been created inside directory: "
            f"'{path_to_checkpoint}'."
        )

algo.stop()
ray.shutdown()
