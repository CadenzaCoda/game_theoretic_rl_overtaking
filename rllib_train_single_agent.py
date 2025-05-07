import ray
# Initialize Ray
ray.init(
    address="auto"
)
import os
import gymnasium as gym
from gymnasium import spaces
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env
import numpy as np
# Import your custom environment
from mpclab_common.track import get_track
# from gym_carla.controllers.barc_pid import PIDWrapper
from gym_carla.controllers.barc_pid_ref_tracking import PIDRacelineFollowerWrapper
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

# Define a function to instantiate the environment

NUM_TRAINING_STEPS = 3000
CHECKPOINT_INTERVAL = 200

def env_creator(config):
    print("Creating Environment")
    import gym_carla
    env_name = "barc-v1-race"
    track_name = "L_track_barc"
    # opponent = PIDWrapper(dt=0.1, t0=0., track_obj=get_track(track_name))
    opponent = PIDRacelineFollowerWrapper(dt=0.1, t0=0., track_obj=get_track(track_name))
    env = gym.make(env_name, opponent=opponent, track_name=track_name, do_render=False, enable_camera=False,
                   discrete_action=True)
    return env

register_env(name="my_barc", env_creator=env_creator)
config = (
    PPOConfig()
    .environment("my_barc")
    .env_runners(
        num_env_runners=10,  # Reduce the number of rollout workers
        # num_cpus_per_env_runner=0,
        # num_gpus_per_env_runner=0.2,
        # rollout_fragment_length="auto",  # Adjust fragment size for better GPU utilization
        # batch_mode="complete_episodes",  # Ensure full episodes are used
        # sample_timeout_s=None,  # Increase the sample timeout to avoid timeouts
        # observation_filter="MeanStdFilter"
    )

    # .evaluation(
    #     evaluation_interval=10,  # Optional: for running evaluation episodes every N iterations
    #     evaluation_duration=3,                      # run exactly 1 episode
    #     evaluation_duration_unit="episodes",
    #     evaluation_config={
    #         "render_env": True,                     # call env.render()
    #         "create_env_on_driver": True,           # build eval env in driver
    #         "env_config": { "do_render": True },    # (if your env looks at this flag)
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

# checkpoint_path = "/home/wilson/ray_results/PPO_min_speed_2025-02-25_15-28-5228wknq_n/checkpoint_step_105"
# algo.restore(checkpoint_path)

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
