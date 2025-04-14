import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import gym_carla
from src.carla_gym.controllers.barc_pid import PIDWrapper
from loguru import logger
import os
from typing import Dict, List, Tuple

# Actor Network
class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim * 2)  # Mean and log_std for each action dimension
        )
        
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.net(state)
        mean, log_std = torch.chunk(x, 2, dim=-1)
        log_std = torch.clamp(log_std, -20, 2)  # Prevent too small or large std
        return mean, log_std

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

class PPOTrainer:
    def __init__(
        self,
        env_name: str = 'barc-v1',
        track_name: str = 'L_track_barc',
        hidden_dim: int = 256,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        target_kl: float = 0.01,
        max_grad_norm: float = 0.5,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.env = gym.make(env_name, track_name=track_name, do_render=False, enable_camera=False)
        self.opponent = PIDWrapper(dt=0.1, t0=0., track_obj=self.env.unwrapped.get_track())
        # self.env.unwrapped.bind_controller(self.opponent)
        
        # Get state and action dimensions from environment
        state_dim = self.env.observation_space['state'].shape[0]
        action_dim = self.env.action_space.shape[1]
        logger.debug(f"State dimension: {state_dim}")
        logger.debug(f"Action dimension: {action_dim}")
        
        # Initialize networks
        self.actor = Actor(state_dim, action_dim, hidden_dim).to(device)
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
        mean, log_std = self.actor(states)
        std = log_std.exp()
        dist = Normal(mean, std)
        
        # Compute new log probs and entropy
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
        
        # Compute approximate KL divergence
        kl_div = ((old_log_probs - new_log_probs) ** 2).mean().item()
        
        return total_loss, value_loss, kl_div
    
    def train_step(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor
    ) -> Dict[str, float]:
        """Perform one step of PPO training."""
        # Convert to tensors and move to device
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Perform multiple epochs of training
        for _ in range(10):  # Number of epochs
            # Compute loss
            total_loss, value_loss, kl_div = self.compute_ppo_loss(
                states, actions, old_log_probs, advantages, returns
            )
            
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
        
        return {
            'total_loss': total_loss.item(),
            'value_loss': value_loss.item(),
            'kl_div': kl_div
        }
    
    def collect_rollout(self, max_steps: int = 2048) -> Tuple[np.ndarray, ...]:
        """Collect a rollout of experiences."""
        states = []
        actions = []
        rewards = []
        values = []
        log_probs = []
        dones = []
        
        state, info = self.env.reset()
        self.opponent.reset()
        state = state['state']  # Extract state from observation dict
        
        for _ in range(max_steps):
            # Convert state to tensor
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            # Get action from policy
            with torch.no_grad():
                mean, log_std = self.actor(state_tensor)
                std = log_std.exp()
                dist = Normal(mean, std)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(dim=-1)
                value = self.critic(state_tensor)
            
            # Take action in environment
            opponent_action, _ = self.opponent.step(info['vehicle_state'][1], terminated=info['terminated'][1], lap_no=info['lap_no'][1])
            next_state, reward, terminated, truncated, info = self.env.step(np.stack([action.cpu().numpy()[0], opponent_action], axis=0))
            next_state = next_state['state']  # Extract state from observation dict
            done = terminated or truncated
            
            # Store experience
            states.append(state)
            actions.append(action.cpu().numpy()[0])
            rewards.append(reward)
            values.append(value.cpu().numpy()[0])
            log_probs.append(log_prob.cpu().numpy()[0])
            dones.append(done)
            
            if done:
                state, info = self.env.reset()
                state = state['state']
            else:
                state = next_state
        
        # Get final value for GAE computation
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            final_value = self.critic(state_tensor).cpu().numpy()[0]
        
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(values),
            np.array(log_probs),
            np.array(dones),
            final_value
        )
    
    def train(self, num_iterations: int = 1000, max_steps: int = 2048):
        """Train the PPO agent."""
        for i in range(num_iterations):
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
            
            # Log metrics
            logger.info(f"Iteration {i}")
            logger.info(f"Total Loss: {metrics['total_loss']:.3f}")
            logger.info(f"Value Loss: {metrics['value_loss']:.3f}")
            logger.info(f"KL Divergence: {metrics['kl_div']:.3f}")
            
            # Save model periodically
            if (i + 1) % 100 == 0:
                self.save_model(f"ppo_model_{i+1}.pt")
    
    def save_model(self, filename: str):
        """Save the model."""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict()
        }, filename)
        logger.info(f"Model saved to {filename}")
    
    def load_model(self, filename: str):
        """Load the model."""
        checkpoint = torch.load(filename)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        logger.info(f"Model loaded from {filename}")

if __name__ == "__main__":
    # Create trainer
    trainer = PPOTrainer()
    
    # Train the agent
    trainer.train(num_iterations=1000, max_steps=2048) 
