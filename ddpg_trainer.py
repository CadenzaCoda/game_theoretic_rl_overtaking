import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gym_carla
from gym_carla.controllers.barc_pid import PIDWrapper
from loguru import logger
import os
from typing import Dict, List, Tuple, Optional, Union, Any
from torch.utils.tensorboard import SummaryWriter
import time
import datetime
import pdb
from collections import deque
import random

# Actor Network for DDPG
class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256, max_action: float = 1.0):
        super(Actor, self).__init__()
        self.max_action = max_action
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # Output in [-1, 1]
        )
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.max_action * self.net(state)

# Critic Network for DDPG
class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=1)
        return self.net(x)

# Replay Buffer for DDPG
class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size: int) -> Tuple:
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (np.array(state), np.array(action), np.array(reward), 
                np.array(next_state), np.array(done))
        
    def __len__(self) -> int:
        return len(self.buffer)

class DDPGTrainer:
    def __init__(
        self,
        env_name: str = 'barc-v1',
        track_name: str = 'L_track_barc',
        hidden_dim: int = 256,
        actor_lr: float = 1e-4,
        critic_lr: float = 1e-3,
        gamma: float = 0.99,
        tau: float = 0.005,
        batch_size: int = 256,
        buffer_size: int = int(1e6),
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        log_dir: Optional[str] = None,
        model_name: str = 'ddpg',
        comment: Optional[str] = None
    ):
        self.env = gym.make(env_name, track_name=track_name, do_render=False, enable_camera=False)
        self.opponent = PIDWrapper(dt=0.1, t0=0., track_obj=self.env.unwrapped.get_track())
        
        # Get state and action dimensions from environment
        state_dim = self.env.observation_space['state'].shape[0]
        action_dim = self.env.action_space.shape[1]
        
        # Get action bounds - assuming symmetric bounds around 0
        max_action = 1.0  # Most continuous control tasks use normalized actions [-1, 1]
        
        logger.debug(f"State dimension: {state_dim}")
        logger.debug(f"Action dimension: {action_dim}")
        logger.debug(f"Max action value: {max_action}")
        
        # Initialize networks
        self.actor = Actor(state_dim, action_dim, hidden_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, hidden_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        self.critic = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Initialize optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Store hyperparameters
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
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
        """Set the log directory with a formatted name."""
        if log_dir is None:
            log_dir = 'runs'
            
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_name = f"{self.model_name}_{self.env_name}"
        if comment:
            log_name += f"_{comment}"
        log_name += f"_{timestamp}"
        
        self.log_dir = os.path.join(log_dir, log_name)
        os.makedirs(self.log_dir, exist_ok=True)
        
        if hasattr(self, 'writer'):
            self.writer.close()
            
        self.writer = SummaryWriter(self.log_dir)
        logger.info(f"TensorBoard logs will be saved to: {self.log_dir}")
        
        self.log_hyperparameters()
        
    def log_hyperparameters(self):
        """Log hyperparameters to TensorBoard."""
        hparams = {
            'env_name': self.env_name,
            'model_name': self.model_name,
            'gamma': self.gamma,
            'tau': self.tau,
            'batch_size': self.batch_size,
            'device': self.device,
            'actor_hidden_dim': self.actor.net[0].out_features,
            'critic_hidden_dim': self.critic.net[0].out_features,
        }
        
        self.writer.add_hparams(hparams, {'train/total_loss': 0})
        hparams_text = "\n".join([f"{k}: {v}" for k, v in hparams.items()])
        self.writer.add_text('hyperparameters', hparams_text)
        
    def select_action(self, state: np.ndarray, noise_std: float = 0.1) -> np.ndarray:
        """Select action with exploration noise."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action = self.actor(state_tensor).cpu().numpy()[0]
            if noise_std > 0:
                action += np.random.normal(0, noise_std, size=action.shape)
                action = np.clip(action, -1, 1)
        return action
        
    def train_step(self, batch_size: int) -> Dict[str, float]:
        """Perform one step of DDPG training."""
        if len(self.replay_buffer) < batch_size:
            return {
                'actor_loss': 0.0,
                'critic_loss': 0.0,
                'q_value': 0.0
            }
            
        # Sample from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Compute target Q value
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = self.critic_target(next_states, next_actions)
            target_q = rewards + (1 - dones) * self.gamma * target_q
            
        # Update critic
        current_q = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_q, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update actor
        actor_loss = -self.critic(states, self.actor(states)).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update target networks
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'q_value': current_q.mean().item()
        }
        
    def collect_experience(self, max_steps: int = 2048, noise_std: float = 0.1):
        """Collect experience for training."""
        state, info = self.env.reset()
        self.opponent.reset()
        state = state['state']
        
        episode_rewards = []
        current_episode_reward = 0
        steps_taken = 0
        
        while steps_taken < max_steps:
            # Select action
            action = self.select_action(state, noise_std)
            
            # Get opponent action
            opponent_action, _ = self.opponent.step(info['vehicle_state'][1], terminated=info['terminated'][1], lap_no=info['lap_no'][1])
            
            # Take step in environment
            next_state, reward, terminated, truncated, info = self.env.step(np.stack([action, opponent_action], axis=0))
            next_state = next_state['state']
            done = terminated or truncated
            
            # Store experience
            self.replay_buffer.push(state, action, reward, next_state, done)
            
            current_episode_reward += reward
            steps_taken += 1
            
            if done:
                episode_rewards.append(current_episode_reward)
                current_episode_reward = 0
                state, info = self.env.reset()
                state = state['state']
                self.opponent.reset()
            else:
                state = next_state
                
        # Log episode rewards
        if episode_rewards:
            self.writer.add_scalar('rollout/mean_episode_reward', np.mean(episode_rewards), self.episode_count)
            self.writer.add_scalar('rollout/max_episode_reward', np.max(episode_rewards), self.episode_count)
            self.writer.add_scalar('rollout/min_episode_reward', np.min(episode_rewards), self.episode_count)
            
    def train(self, num_iterations: int = 1000, max_steps: int = 2048):
        """Train the DDPG agent."""
        self.episode_count = 0
        while self.episode_count < num_iterations:
            # Collect experience
            self.collect_experience(max_steps)
            
            # Perform multiple training steps
            metrics_list = []
            for _ in range(50):  # Number of training steps per iteration
                metrics = self.train_step(self.batch_size)
                metrics_list.append(metrics)
                
            # Calculate average metrics
            avg_metrics = {
                k: np.mean([m[k] for m in metrics_list])
                for k in metrics_list[0].keys()
            }
            
            # Log metrics
            self.writer.add_scalar('train/actor_loss', avg_metrics['actor_loss'], self.episode_count)
            self.writer.add_scalar('train/critic_loss', avg_metrics['critic_loss'], self.episode_count)
            self.writer.add_scalar('train/q_value', avg_metrics['q_value'], self.episode_count)
            
            logger.info(f"Iteration {self.episode_count}")
            logger.info(f"Actor Loss: {avg_metrics['actor_loss']:.3f}")
            logger.info(f"Critic Loss: {avg_metrics['critic_loss']:.3f}")
            logger.info(f"Q Value: {avg_metrics['q_value']:.3f}")
            
            # Save model periodically
            if (self.episode_count + 1) % 100 == 0:
                self.save_model(f"ddpg_model_{self.episode_count + 1}_{self.env_name}_{self.model_name}.pt")
                
            self.evaluate_agent()
            self.episode_count += 1
            
    def evaluate_agent(self):
        """Evaluate the agent's performance."""
        state, info = self.env.reset(options={'render': True})
        self.opponent.reset()
        terminated, truncated = False, False
        min_rel_dist = np.inf
        episode_reward = 0
        
        with torch.no_grad():
            while not truncated:
                state_tensor = torch.FloatTensor(state['state']).unsqueeze(0).to(self.device)
                action = self.actor(state_tensor).cpu().numpy()[0]
                opponent_action, _ = self.opponent.step(info['vehicle_state'][1], terminated=info['terminated'][1], lap_no=info['lap_no'][1])
                next_state, reward, terminated, truncated, info = self.env.step(np.stack([action, opponent_action], axis=0))
                min_rel_dist = min(min_rel_dist, info['relative_distance'])
                episode_reward += reward
                state = next_state  # Update state for next iteration
                
        # Log evaluation metrics
        self.writer.add_scalar('eval/episode_reward', episode_reward, self.episode_count)
        self.writer.add_scalar('eval/min_relative_distance', min_rel_dist, self.episode_count)
        
        if terminated:
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
        
    def save_model(self, filename: str):
        """Save the model."""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'actor_target_state_dict': self.actor_target.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict()
        }, filename)
        logger.info(f"Model saved to {filename}")
        
    def load_model(self, filename: str):
        """Load the model."""
        checkpoint = torch.load(filename)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.actor_target.load_state_dict(checkpoint['actor_target_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        logger.info(f"Model loaded from {filename}")
        
    def close(self):
        """Close TensorBoard writer."""
        self.writer.close()

if __name__ == "__main__":
    # Create trainer with custom log directory
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--comment', type=str, default='experimental')
    params = parser.parse_args()

    # Create checkpoints directory if it doesn't exist
    os.makedirs('checkpoints', exist_ok=True)

    # Suppress warnings about barc3d
    import warnings
    warnings.filterwarnings('ignore', message='Barc3d is not available')
    warnings.filterwarnings('ignore', message='surface_lib functions import failed')
    warnings.filterwarnings('ignore', message='barc3d related package import failed')

    trainer = DDPGTrainer(
        model_name="ddpg",
        env_name="barc-v1",
        track_name="L_track_barc",
        comment=params.comment
    )
    
    # Train the agent
    try:
        trainer.train(num_iterations=1000, max_steps=2048)
    finally:
        trainer.save_model('checkpoints/ddpg.pth')
        trainer.close()  # Close TensorBoard writer
