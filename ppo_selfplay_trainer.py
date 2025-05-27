import gymnasium as gym
import numpy as np
import gym_carla
import torch
import torch.nn as nn
import torch.optim as optim
from numpy import floating
from tqdm import trange

import random
from loguru import logger
import os
from typing import Dict, Tuple, Optional, Union, Any
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Distribution, Categorical, Normal
import torch.nn.functional as F
import time
import datetime
from pathlib import Path

from mpclab_common.track import get_track

from torch.distributions import register_kl
from ray.rllib.env.multi_agent_env import MultiAgentEnv


def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.mps.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Scoreboard:
    def __init__(self):
        self.ego_wins = 0
        self.oppo_wins = 0
        self.draws = 0
        self.truncations = 0
        self.ego_losses = 0
        self.oppo_losses = 0
        self.ego_avg_speed = 0
        self.oppo_avg_speed = 0
        self.steps = 0

    def reset(self):
        self.ego_wins = 0
        self.oppo_wins = 0
        self.draws = 0
        self.truncations = 0
        self.ego_losses = 0
        self.oppo_losses = 0
        self.ego_avg_speed = 0
        self.oppo_avg_speed = 0
        self.steps = 0

    def log_speed(self, info):
        self.ego_avg_speed += info['ego']['vehicle_state'].v.v_long
        self.oppo_avg_speed += info['oppo']['vehicle_state'].v.v_long
        self.steps += 1

    def log_result(self, info):
        if info['ego']['success']:
            self.ego_wins += 1
        if info['ego']['failure']:
            self.ego_losses += 1
        elif info['oppo']['success']:
            self.oppo_wins += 1
        elif info['oppo']['failure']:
            self.oppo_losses += 1
        elif info['ego']['collision']:
            self.draws += 1
        else:
            self.truncations += 1

    def write_results(self, writer, episode_count, verbose=False):
        writer.add_scalar('rollout/ego_wins', self.ego_wins, episode_count)
        writer.add_scalar('rollout/oppo_wins', self.oppo_wins, episode_count)
        writer.add_scalar('rollout/ego_losses', self.ego_losses, episode_count)
        writer.add_scalar('rollout/oppo_losses', self.oppo_losses, episode_count)
        writer.add_scalar('rollout/draws', self.draws, episode_count)
        writer.add_scalar('rollout/truncations', self.truncations, episode_count)
        if self.steps > 0:
            writer.add_scalar('rollout/ego_avg_speed', self.ego_avg_speed / self.steps, episode_count)
            writer.add_scalar('rollout/oppo_avg_speed', self.oppo_avg_speed / self.steps, episode_count)
        if verbose:
            logger.info(self)

    # def print_results(self):
    def __repr__(self):
        return (f"EW: {self.ego_wins}, OW: {self.oppo_wins}, EL: {self.ego_losses}, OL: {self.oppo_losses}, "
                f"D: {self.draws}, G: {self.truncations}, " +
                (f"EV: {self.ego_avg_speed / self.steps:.2f}, OV: {self.oppo_avg_speed / self.steps:.2f}" if self.steps > 0 else ""))


class MultiCategorical(Distribution):
    arg_constraints = {}  # Optional: constraints on arguments (skip for now)
    has_rsample = False  # Cannot reparameterize sampling for discrete actions

    def __init__(self, logits: torch.Tensor, nvec: list):
        super().__init__(batch_shape=logits.shape[:-1], event_shape=torch.Size([len(nvec)]))
        self.nvec = list(nvec)
        self.split_logits = torch.split(logits, self.nvec, dim=-1)
        self.categoricals = [Categorical(logits=logit) for logit in self.split_logits]

    def sample(self, sample_shape=torch.Size()) -> torch.Tensor:
        samples = [dist.sample(sample_shape) for dist in self.categoricals]
        # Each sample has shape: sample_shape + batch_shape
        # Stack them to create final shape: sample_shape + batch_shape + event_shape
        return torch.stack(samples, dim=-1)

    def log_prob(self, actions: torch.Tensor):
        log_probs = [
            dist.log_prob(actions[..., i])
            for i, dist in enumerate(self.categoricals)
        ]
        return torch.stack(log_probs, dim=-1).sum(dim=-1)

    def entropy(self):
        entropies = [dist.entropy() for dist in self.categoricals]
        return torch.stack(entropies, dim=-1).sum(dim=-1)

    def mode(self):
        """Return the mode (argmax) of each categorical."""
        modes = [torch.argmax(logits, dim=-1) for logits in self.split_logits]
        return torch.stack(modes, dim=-1)


@register_kl(MultiCategorical, MultiCategorical)
def _kl_multi(m1, m2):
    # sum of per-dim KLs
    return sum(torch.distributions.kl_divergence(d1, d2)
               for d1, d2 in zip(m1.categoricals, m2.categoricals))


# Actor Network
class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, n_actions_per_dim: np.ndarray, hidden_dim: int = 256,
                 discrete: bool = False):
        super(Actor, self).__init__()
        self.action_dim = action_dim
        self.n_actions_per_dim = n_actions_per_dim
        self.discrete = discrete

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, np.sum(n_actions_per_dim) if discrete else action_dim * 2)
        )

    def forward(self, state: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        x = self.net(state)
        if not self.discrete:
            mean, log_std = torch.chunk(x, 2, dim=-1)
            log_std = torch.clamp(log_std, -20, 2)  # Prevent too small or large std.
            return mean, log_std  # These are the mean and log_std for Normal distributions.
        return x  # These are logits, and they will be used to form a MultiCategorical distribution later.


# Critic Network
class Critic(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 256):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state).squeeze(1)


class PPOSelfPlayTrainer:
    def __init__(
            self,
            # env: gym.Env,
            env: MultiAgentEnv,
            env_name: str = 'barc-laps-v1',
            hidden_dim: int = 256,
            learning_rate: float = 3e-4,
            gamma: float = 0.99,
            gae_lambda: float = 0.95,
            clip_ratio: float = 0.2,
            target_kl: float = 0.01,
            max_grad_norm: float = 0.5,
            device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
            log_dir: Optional[str] = None,
            model_name: str = 'ppo_selfplay',
            comment: Optional[str] = None,
            n_actions_per_dim: int = 10
    ):
        self.log_dir = None
        self.writer = None
        self.env = env

        # Get state and action dimensions from environment
        state_dim = self.env.observation_space['ego'].shape[0]
        if isinstance(self.env.action_space['ego'], gym.spaces.MultiDiscrete):
            self.n_logits = self.env.action_space['ego'].nvec
            self.action_dim = len(self.env.action_space['ego'].nvec)
            self.discrete = True
        elif isinstance(self.env.action_space['ego'], gym.spaces.Discrete):
            self.n_logits = np.array([self.env.action_space['ego'].n])
            self.action_dim = 1
            self.discrete = True
        elif isinstance(self.env.action_space['ego'], gym.spaces.Box):
            self.n_logits = None
            self.action_dim = self.env.action_space['ego'].shape[0]
            self.discrete = False
        else:
            raise NotImplementedError(f"Unsupported action space: {self.env.action_space['ego']}")

        # Initialize networks
        self.actor = Actor(state_dim, action_dim=self.action_dim, n_actions_per_dim=self.n_logits,
                           hidden_dim=hidden_dim, discrete=self.discrete).to(device)
        self.critic = Critic(state_dim, hidden_dim).to(device)

        # Initialize optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)

        # Store hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.target_kl = target_kl
        self.max_grad_norm = max_grad_norm
        self.device = device

        self.episode_count = 0
        self.success_count = 0

        # Store environment and model info
        self.env_name = env_name
        self.model_name = model_name

        # Setup TensorBoard
        self.set_log_dir(log_dir, comment)
        self.start_time = time.time()

    def set_log_dir(self, log_dir: Optional[str] = None, comment: Optional[str] = None):
        """
        Set the log directory with a formatted name based on model name, environment name, and optional comment.
        
        Args:
            log_dir: Optional base directory for logs. If None, uses 'runs'.
            comment: Optional comment to append to the log directory name.
        """
        # Create base log directory if not provided
        if log_dir is None:
            log_dir = 'runs'

        # Create timestamp for unique identification
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Build the log directory name
        log_name = f"{self.model_name}_{self.env_name}"
        if comment:
            log_name += f"_{comment}"
        log_name += f"_{timestamp}"

        # Create the full log directory path
        self.log_dir = os.path.join(log_dir, log_name)
        os.makedirs(self.log_dir, exist_ok=True)

        # Close existing writer if it exists
        if self.writer is not None:
            self.writer.close()

        # Create new writer
        self.writer = SummaryWriter(self.log_dir)
        logger.info(f"TensorBoard logs will be saved to: {self.log_dir}")

        # Log hyperparameters
        self.log_hyperparameters()

    def log_hyperparameters(self):
        """Log hyperparameters to TensorBoard."""
        hparams = {
            'env_name': self.env_name,
            'model_name': self.model_name,
            'gamma': self.gamma,
            'gae_lambda': self.gae_lambda,
            'clip_ratio': self.clip_ratio,
            'target_kl': self.target_kl,
            'max_grad_norm': self.max_grad_norm,
            'device': self.device,
            'actor_hidden_dim': self.actor.net[0].out_features,
            'critic_hidden_dim': self.critic.net[0].out_features,
        }

        # Add hyperparameters to TensorBoard
        self.writer.add_hparams(hparams, {'train/total_loss': 0})  # Placeholder metric

        # Log hyperparameters as text
        hparams_text = "\n".join([f"{k}: {v}" for k, v in hparams.items()])
        self.writer.add_text('hyperparameters', hparams_text)

    def compute_gae(
            self,
            rewards: torch.Tensor,
            values: torch.Tensor,
            dones: torch.Tensor,
            next_value: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Generalized Advantage Estimation."""
        advantages = torch.zeros_like(rewards)
        last_gae = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value_t = next_value
            else:
                next_value_t = values[t + 1]

            delta = rewards[t] + self.gamma * next_value_t * (1 - dones[t]) - values[t]
            advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae

        returns = advantages + values
        return advantages, returns

    def compute_ppo_loss(
            self,
            states: torch.Tensor,
            actions: torch.Tensor,
            old_log_probs: torch.Tensor,
            advantages: torch.Tensor,
            returns: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """Compute PPO loss for both actor and critic."""
        # Get current policy distribution
        if self.discrete:
            logits = self.actor(states)
            dist = MultiCategorical(logits=logits, nvec=self.n_logits)
        else:
            mean, log_std = self.actor(states)
            std = log_std.exp()
            dist = Normal(mean, std)

        # Compute new log probs and entropy
        if self.discrete:
            new_log_probs = dist.log_prob(actions)
        else:
            new_log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().mean()

        # Compute ratio and clipped surrogate loss
        ratio = torch.exp(new_log_probs - old_log_probs)
        clip_adv = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
        actor_loss = -torch.min(ratio * advantages, clip_adv).mean()

        # Compute value loss
        values = self.critic(states).squeeze()
        value_loss = nn.MSELoss()(values, returns)

        # Compute total loss
        total_loss = actor_loss + 0.5 * value_loss - 0.01 * entropy

        # Compute approximate KL divergence via Monte Carlo
        kl_div = (old_log_probs - new_log_probs).mean().item()

        return total_loss, value_loss, kl_div

    def train_step(
            self,
            states: np.ndarray,
            actions: np.ndarray,
            old_log_probs: np.ndarray,
            advantages: np.ndarray,
            returns: np.ndarray,
    ):
        """Perform one step of PPO training."""
        # Convert to tensors and move to device
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Track training metrics
        total_losses = []
        value_losses = []
        kl_divs = []

        # Perform multiple epochs of training
        for epoch in range(10):  # Number of epochs
            # Compute loss
            total_loss, value_loss, kl_div = self.compute_ppo_loss(
                states, actions, old_log_probs, advantages, returns
            )

            # Track metrics
            total_losses.append(total_loss.item())
            value_losses.append(value_loss.item())
            kl_divs.append(kl_div)

            # Early stopping if KL divergence is too high
            if kl_div > 1.5 * self.target_kl:
                logger.info(f"Early stopping at KL divergence: {kl_div:.3f}")
                break

            # Update networks
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()
            self.critic_optimizer.step()

        # Calculate average metrics
        avg_total_loss = np.mean(total_losses)
        avg_value_loss = np.mean(value_losses)
        avg_kl_div = np.mean(kl_divs)

        return {
            'total_loss': avg_total_loss,
            'value_loss': avg_value_loss,
            'kl_div': avg_kl_div
        }

    def collect_rollout(self, max_steps: int = 2048) -> Tuple[np.ndarray, ...]:
        """Collect a rollout of experiences using self-play."""
        states = []
        actions = []
        rewards = []
        values = []
        log_probs = []
        dones = []

        scoreboard = Scoreboard()

        state, info = self.env.reset()

        episode_rewards = []
        current_episode_reward = 0

        # Randomly choose from which vehicle's perspective to collect the rollout
        ego_vehicle = random.choice([True, False])

        for _ in trange(max_steps, desc='Collect'):
            # Infer the ego vehicle action
            # Convert state to tensor
            state_tensor = torch.FloatTensor(state['ego']).unsqueeze(0).to(self.device)

            # Get action from policy for ego vehicle
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

            # Infer the opponent vehicle action by flipping the observation
            flipped_state = state['oppo']
            flipped_state_tensor = torch.FloatTensor(flipped_state).unsqueeze(0).to(self.device)

            with torch.no_grad():
                if self.discrete:
                    flipped_logits = self.actor(flipped_state_tensor)
                    flipped_dist = MultiCategorical(flipped_logits, self.n_logits)
                else:
                    flipped_mean, flipped_log_std = self.actor(flipped_state_tensor)
                    flipped_std = flipped_log_std.exp()
                    flipped_dist = Normal(flipped_mean, flipped_std)
                flipped_action = flipped_dist.sample()
                flipped_log_prob = flipped_dist.log_prob(flipped_action).sum(dim=-1)
                flipped_value = self.critic(flipped_state_tensor)

            # Combine actions for both vehicles to get the final action
            combined_action = {
                'ego': action.detach().cpu().numpy()[0],
                'oppo': flipped_action.detach().cpu().numpy()[0]
            }

            # Step environment with the combined action
            next_state, reward, terminated, truncated, info = self.env.step(combined_action)

            # Store experience for ego vehicle
            if ego_vehicle:
                states.append(state['ego'])
                actions.append(action.detach().cpu().numpy()[0])
                rewards.append(reward['ego'])  # Use ego vehicle's reward
                values.append(value.cpu().numpy()[0])
                log_probs.append(log_prob.cpu().numpy())
                dones.append(terminated['ego'])  # Use ego vehicle's termination

            else:
                # Store experience for opponent vehicle
                states.append(state['oppo'])
                actions.append(flipped_action.detach().cpu().numpy()[0])
                rewards.append(reward['oppo'])  # Use opponent's reward
                values.append(flipped_value.cpu().numpy()[0])
                log_probs.append(flipped_log_prob.cpu().numpy())
                dones.append(terminated['oppo'])  # Use opponent's termination

            current_episode_reward += reward['ego'] if ego_vehicle else reward['oppo']
            scoreboard.log_speed(info)

            if terminated['ego'] or truncated['ego'] or terminated['oppo'] or truncated[
                'oppo']:  # Reset on either condition
                scoreboard.log_result(info)
                episode_rewards.append(current_episode_reward)
                current_episode_reward = 0
                ego_vehicle = random.choice([True, False])
                state, info = self.env.reset()
            else:
                state = next_state

        # Get final value for GAE computation
        with torch.no_grad():
            if ego_vehicle:
                state_tensor = torch.FloatTensor(state['ego']).unsqueeze(0).to(self.device)
            else:
                state_tensor = torch.FloatTensor(state['oppo']).unsqueeze(0).to(self.device)
            final_value = self.critic(state_tensor).cpu().numpy()[0]

        # Log episode rewards
        if episode_rewards:
            self.writer.add_scalar('rollout/mean_episode_reward', np.mean(episode_rewards), self.episode_count)
            self.writer.add_scalar('rollout/max_episode_reward', np.max(episode_rewards), self.episode_count)
            self.writer.add_scalar('rollout/min_episode_reward', np.min(episode_rewards), self.episode_count)
        scoreboard.write_results(self.writer, self.episode_count, verbose=True)

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(values),
            np.array(log_probs),
            np.array(dones),
            final_value
        )

    def _flip_observation(self, obs: np.ndarray) -> np.ndarray:
        """Flip the observation to get opponent's perspective."""
        # The observation contains state information for both vehicles
        # We need to swap the positions of ego and opponent in the observation
        obs = obs.reshape(2, -1)  # Reshape to (2, state_dim)
        flipped_obs = np.flip(obs, axis=0)  # Swap ego and opponent
        return flipped_obs.flatten()  # Flatten back to original shape

    def train(self, num_iterations: int = 1000, max_steps: int = 2048):
        """Train the PPO agent using self-play."""
        self.episode_count = 0
        while self.episode_count < num_iterations:
            # Collect rollout
            states, actions, rewards, values, log_probs, dones, final_value = self.collect_rollout(max_steps)

            # Compute advantages and returns
            advantages, returns = self.compute_gae(
                torch.FloatTensor(rewards),
                torch.FloatTensor(values),
                torch.FloatTensor(dones),
                torch.FloatTensor([final_value])
            )

            # Perform PPO update
            metrics = self.train_step(states, actions, log_probs, advantages, returns)

            # Log metrics to TensorBoard
            self.writer.add_scalar('train/total_loss', metrics['total_loss'], self.episode_count)
            self.writer.add_scalar('train/value_loss', metrics['value_loss'], self.episode_count)
            self.writer.add_scalar('train/kl_divergence', metrics['kl_div'], self.episode_count)
            self.writer.add_scalar('train/mean_advantage', advantages.mean().item(), self.episode_count)
            self.writer.add_scalar('train/mean_return', returns.mean().item(), self.episode_count)

            # Log metrics
            logger.info(f"Iteration {self.episode_count}")
            logger.info(f"Total Loss: {metrics['total_loss']:.3f}")
            logger.info(f"Value Loss: {metrics['value_loss']:.3f}")
            logger.info(f"KL Divergence: {metrics['kl_div']:.3f}")

            # Save model periodically
            if (self.episode_count + 1) % 100 == 0:
                self.save_model(f"ppo_selfplay_model_{self.episode_count + 1}_{self.env_name}_{self.model_name}.pt")

            self.evaluate_agent()
            self.episode_count += 1

    def evaluate_agent(self, scoreboard=None):
        state, info = self.env.reset(options={'render': True})
        terminated, truncated = False, False
        episode_reward = 0

        with torch.no_grad():
            while not truncated and not terminated:
                state_tensor = torch.FloatTensor(state['ego']).unsqueeze(0).to(self.device)

                # Get action for ego vehicle
                if not self.discrete:
                    mean, log_std = self.actor(state_tensor)
                    dist = Normal(mean, log_std.exp())
                else:
                    logits = self.actor(state_tensor)
                    dist = MultiCategorical(logits, self.n_logits)
                action = dist.sample().cpu().numpy()[0]

                # Get action for opponent by flipping observation
                flipped_state = state['oppo']
                flipped_state_tensor = torch.FloatTensor(flipped_state).unsqueeze(0).to(self.device)

                if not self.discrete:
                    flipped_mean, flipped_log_std = self.actor(flipped_state_tensor)
                    flipped_dist = Normal(flipped_mean, flipped_log_std.exp())
                else:
                    flipped_logits = self.actor(flipped_state_tensor)
                    flipped_dist = MultiCategorical(flipped_logits, self.n_logits)
                flipped_action = flipped_dist.sample().cpu().numpy()[0]

                # Combine actions
                combined_action = {
                    'ego': action,
                    'oppo': flipped_action
                }

                state, reward, terminated, truncated, info = self.env.step(combined_action)
                if scoreboard is not None:
                    scoreboard.log_speed(info)
                episode_reward += reward['ego']
                truncated = truncated['__all__']
                terminated = terminated['__all__']

        # Log evaluation metrics
        self.writer.add_scalar('eval/episode_reward', episode_reward, self.episode_count)

        if info['ego']['success']:
            self.success_count += 1
            logger.info(f"Successful episode! Episode {self.episode_count}")
            self.writer.add_scalar('eval/success', 1, self.episode_count)
        else:
            logger.info(f"Failed episode. Episode {self.episode_count}")
            self.writer.add_scalar('eval/success', 0, self.episode_count)
        logger.info(f"Success rate: {self.success_count}/{self.episode_count + 1}")

        # Log training time
        elapsed_time = time.time() - self.start_time
        self.writer.add_scalar('time/elapsed_seconds', elapsed_time, self.episode_count)
        self.writer.add_scalar('time/episodes_per_second', self.episode_count / elapsed_time, self.episode_count)

        return info

    def save_model(self, filename: str):
        """Save the model."""
        # Create directory if it doesn't exist
        checkpoint_root = Path.cwd() / 'checkpoints'
        os.makedirs(checkpoint_root, exist_ok=True)

        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict()
        }, checkpoint_root / filename)
        logger.info(f"Model saved to {checkpoint_root / filename}")

    def load_model(self, filename: str):
        """Load the model."""
        checkpoint_root = Path.cwd() / 'checkpoints'
        if not (checkpoint_root / filename).exists():
            raise ValueError(f"Checkpoint {checkpoint_root / filename} doesn't exist!")
        checkpoint = torch.load(checkpoint_root / filename)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        logger.info(f"Model loaded from {checkpoint_root / filename}")

    def close(self):
        """Close TensorBoard writer."""
        self.writer.close()


if __name__ == "__main__":
    # Create trainer with custom log directory
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--evaluation', action='store_true')
    parser.add_argument('--n_epochs', type=int, default=1000)
    parser.add_argument('--max_steps', type=int, default=2048)
    parser.add_argument('--resume', type=int, default=-1)
    parser.add_argument('-m', '--comment', type=str, default='experimental')
    params = parser.parse_args()

    seed_everything(params.seed)

    if params.evaluation:
        params.comment += '_evaluation'

    env_name = "barc-laps-v1"
    track_name = "L_track_barc"
    model_name = "ppo-selfplay"
    env = gym.make(env_name, track_name=track_name, do_render=False, enable_camera=False,
                   discrete_action=True)

    trainer = PPOSelfPlayTrainer(
        env=env,
        env_name=env_name,
        model_name=model_name,
        comment=params.comment
    )

    if params.evaluation:
        trainer.load_model('ppo_selfplay_model_1000_barc-laps-v1_ppo-selfplay.pt')
        scoreboard = Scoreboard()
        for _ in range(1):
            info = trainer.evaluate_agent()
            scoreboard.log_result(info)
        logger.info(scoreboard)
        exit(0)

    # Train the agent
    try:
        if params.resume > 0:
            trainer.load_model(f"ppo_selfplay_model_{params.resume}_{env_name}_{model_name}.pt")
            trainer.episode_count = params.resume
        trainer.train(num_iterations=params.n_epochs, max_steps=params.max_steps)
    finally:
        trainer.save_model(f'ppo_selfplay_{params.comment}_latest.pth')
        trainer.close()  # Close TensorBoard writer
