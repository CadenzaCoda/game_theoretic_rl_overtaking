import ray
import os
import gymnasium as gym
from gymnasium import spaces
from ray import train
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env
import numpy as np
import torch
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import AgentID, PolicyID

torch.autograd.set_detect_anomaly(True)

if torch.cuda.is_available():
    print("CUDA is available. Running on GPU.")
    print(f"Current device: {torch.cuda.get_device_name(0)}")
    print(f"Total GPUs available: {torch.cuda.device_count()}")
else:
    print("CUDA is not available. Running on CPU.")

NUM_TRAINING_STEPS = 3000
CHECKPOINT_INTERVAL = 200

class SelfPlayCallback(DefaultCallbacks):
    """Custom callback to handle self-play training."""
    
    @override(DefaultCallbacks)
    def on_episode_start(self, *, worker, base_env, policies, episode, **kwargs):
        """Called at the start of each episode."""
        episode.user_data["ego_obs"] = []
        episode.user_data["oppo_obs"] = []
        episode.user_data["ego_actions"] = []
        episode.user_data["oppo_actions"] = []
        episode.user_data["ego_rewards"] = []
        episode.user_data["oppo_rewards"] = []
        episode.user_data["ego_dones"] = []
        episode.user_data["oppo_dones"] = []

    @override(DefaultCallbacks)
    def on_postprocess_trajectory(self, *, worker, episode, agent_id, policy_id, policies, postprocessed_batch, original_batches, **kwargs):
        """Called after trajectory is postprocessed."""
        # Store the trajectory data
        if agent_id == "ego":
            episode.user_data["ego_obs"].append(original_batches[policy_id]["obs"])
            episode.user_data["ego_actions"].append(original_batches[policy_id]["actions"])
            episode.user_data["ego_rewards"].append(original_batches[policy_id]["rewards"])
            episode.user_data["ego_dones"].append(original_batches[policy_id]["dones"])
        else:  # opponent
            episode.user_data["oppo_obs"].append(original_batches[policy_id]["obs"])
            episode.user_data["oppo_actions"].append(original_batches[policy_id]["actions"])
            episode.user_data["oppo_rewards"].append(original_batches[policy_id]["rewards"])
            episode.user_data["oppo_dones"].append(original_batches[policy_id]["dones"])

def env_creator(config):
    """Create the racing environment."""
    print("Creating Environment")
    import gym_carla
    env = gym.make("barc-laps-v1", track_name="L_track_barc", discrete_action=True)
    return env

register_env(name="barc_self_play", env_creator=env_creator)

# Define observation and action spaces
observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2 * 10,), dtype=np.float32)
_action_bounds = np.tile(np.array([2, 0.45]), [2, 1])
u_a_space = np.linspace(-2, 2, 32, endpoint=True, dtype=np.float32)
u_steer_space = np.linspace(-0.45, 0.45, 32, endpoint=True, dtype=np.float32)
action_space = spaces.MultiDiscrete([len(u_a_space), len(u_steer_space)])

# Define a single policy for both ego and opponent
policies = {
    "shared_policy": PolicySpec(None, observation_space, action_space),
}

def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    """Map both agents to the same policy."""
    return "shared_policy"

class SelfPlayEnvWrapper(gym.Wrapper):
    """Wrapper to handle self-play by flipping observations for opponent."""
    
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        
    def step(self, action):
        # Get ego action
        ego_action = action
        
        # Flip the observation for opponent
        flipped_obs = self._flip_observation(self.env.get_obs())
        
        # Get opponent action using the same policy
        oppo_action = self.env.get_opponent_action(flipped_obs)
        
        # Step the environment with both actions
        obs, reward, terminated, truncated, info = self.env.step({
            "ego": ego_action,
            "oppo": oppo_action
        })
        
        return obs, reward, terminated, truncated, info
    
    def _flip_observation(self, obs):
        """Flip the observation to get opponent's perspective."""
        # Reshape observation to separate ego and opponent states
        obs = obs.reshape(2, 9)
        # Swap ego and opponent states
        flipped = np.concatenate([obs[1:], obs[:1]])
        return flipped.reshape(-1)

# Configure the training
config = (
    PPOConfig()
    .environment("barc_self_play")
    .env_runners(
        num_env_runners=10,
    )
    .multi_agent(
        policies=policies,
        policy_mapping_fn=policy_mapping_fn,
        policies_to_train=["shared_policy"]
    )
    .training(
        model={
            "fcnet_hiddens": [1024, 512, 256, 128],
            "vf_share_layers": False,
            "fcnet_activation": "relu"
        },
        train_batch_size=2048,
        minibatch_size=1024,
        lr=1e-4,
        entropy_coeff=[[0, 0.01], [2048 * 1000, 0.001], [2048 * 2000, 0.0001]],
        gamma=0.99
    )
    .learners(
        num_learners=1,
        num_gpus_per_learner=1,
    )
    .framework("torch")
    .callbacks(SelfPlayCallback)
)

# Build and train the algorithm
algo = config.build()

for train_step in range(1, NUM_TRAINING_STEPS + 1):
    result = algo.train()
    print("#############  Iteration %d  #############\n" % train_step)

    if train_step % CHECKPOINT_INTERVAL == 0 or train_step == NUM_TRAINING_STEPS:
        save_result = algo.save(
            checkpoint_dir=os.path.join(
                algo.logdir,
                f"checkpoint_step_{train_step}"
            )
        )
        path_to_checkpoint = save_result.checkpoint.path
        print(
            "An Algorithm checkpoint has been created inside directory: "
            f"'{path_to_checkpoint}'."
        )

algo.stop()
ray.shutdown() 