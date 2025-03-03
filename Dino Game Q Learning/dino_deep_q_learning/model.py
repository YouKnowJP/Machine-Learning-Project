# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
import os


class DQN(nn.Module):
    """Deep Q-Network model architecture."""

    def __init__(self, input_dim, output_dim, hidden_dims=None):
        super(DQN, self).__init__()

        # Fix the mutable default argument
        if hidden_dims is None:
            hidden_dims = [128, 64]

        # Build fully connected layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass through the network."""
        return self.model(x)


class DQNAgent:
    """DQN agent for playing the Dino game."""

    def __init__(self, state_dim, action_dim, config):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() and config.use_cuda else "cpu")

        # Create Q-networks
        self.policy_net = DQN(state_dim, action_dim, config.hidden_dims).to(self.device)
        self.target_net = DQN(state_dim, action_dim, config.hidden_dims).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Experience replay buffer
        self.memory = deque(maxlen=config.buffer_size)

        # Training parameters
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=config.learning_rate)
        self.batch_size = config.batch_size
        self.gamma = config.gamma  # Discount factor
        self.epsilon = config.epsilon_start
        self.epsilon_end = config.epsilon_end
        self.epsilon_decay = config.epsilon_decay
        self.target_update = config.target_update
        self.train_step = 0

        # Path for saving models
        self.checkpoint_dir = config.checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def select_action(self, state, training=True):
        """
        Select action using epsilon-greedy policy.

        Args:
            state (np.ndarray): Current state observation
            training (bool): Whether in training mode (use epsilon) or not

        Returns:
            int: Selected action
        """
        if training and random.random() < self.epsilon:
            # Random action
            return random.randrange(self.action_dim)
        else:
            # Greedy action
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                return q_values.max(1)[1].item()

    def update_epsilon(self):
        """Update epsilon value using decay schedule."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory."""
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        """Train the model on a batch of experiences from the replay buffer."""
        if len(self.memory) < self.batch_size:
            return

        # Sample random batch from memory
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Compute current Q values
        current_q = self.policy_net(states).gather(1, actions)

        # Compute next Q values using target network
        with torch.no_grad():
            # Double DQN: use policy net to select actions, target net to evaluate them
            next_actions = self.policy_net(next_states).max(1)[1].unsqueeze(1)
            next_q = self.target_net(next_states).gather(1, next_actions)
            target_q = rewards + (1 - dones) * self.gamma * next_q

        # Compute loss and update weights
        loss = F.smooth_l1_loss(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

        # Update target network periodically
        self.train_step += 1
        if self.train_step % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        # Update epsilon
        self.update_epsilon()

        return loss.item()

    def save_model(self, filename):
        """Save model weights to file."""
        save_path = os.path.join(self.checkpoint_dir, filename)
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'train_step': self.train_step
        }, save_path)
        print(f"Model saved to {save_path}")

    def load_model(self, filename):
        """Load model weights from file."""
        load_path = os.path.join(self.checkpoint_dir, filename)
        if os.path.exists(load_path):
            checkpoint = torch.load(load_path, map_location=self.device)
            self.policy_net.load_state_dict(checkpoint['policy_net'])
            self.target_net.load_state_dict(checkpoint['target_net'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.epsilon = checkpoint['epsilon']
            self.train_step = checkpoint['train_step']
            print(f"Model loaded from {load_path}")
            return True
        else:
            print(f"No model found at {load_path}")
            return False
