import os
import numpy as np
import random
from collections import deque
import config

import torch
import torch.nn as nn
import torch.optim as optim

# Example PyTorch QNetwork; you can adapt it to your needs
class QNetwork(nn.Module):
    def __init__(self, num_channels=4, num_actions=3):
        super(QNetwork, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(num_channels, 32, kernel_size=8, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=5, stride=2),
            nn.Conv2d(32, 64, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=7, stride=3),
            nn.Conv2d(64, 128, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=3),
        )
        # Flatten -> FC layers
        self.fc1 = nn.Linear(384, 64)  # Might need adjusting if shape mismatch
        self.fc2 = nn.Linear(64, 8)
        self.fc3 = nn.Linear(8, num_actions)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = nn.ReLU()(self.fc1(x))
        x = nn.ReLU()(self.fc2(x))
        x = self.fc3(x)
        return x


class Agent:
    def __init__(
        self,
        input_shape=(76, 384, 4),
        memory_size=config.MEMORY_SIZE,
        batch_size=config.BATCH_SIZE,
        lr=1e-4,
        gamma=0.95,
        alpha=0.9,
    ):
        self.model = QNetwork(num_channels=input_shape[2], num_actions=3)

        # Check if pretrained weights file exists
        if os.path.isfile(config.PRETRAINED_WEIGHTS):
            # If it's a TensorFlow/Keras file, you'd need a different approach to load it in PyTorch.
            # If it's a PyTorch .pt or .pth file, do something like:
            # self.model.load_state_dict(torch.load(config.PRETRAINED_WEIGHTS))
            # For demonstration, we'll just print:
            print(f"Found pretrained weights: {config.PRETRAINED_WEIGHTS}")
            print("Loading is not implemented in PyTorch code if it's .h5!")
        else:
            print("No pretrained weights found. Training from scratch.")

        # Replay buffer
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size
        self.loss_history = []
        self.location = 0

        self.gamma = gamma
        self.alpha = alpha

        # PyTorch optimizer & loss
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def predict(self, state):
        self.model.eval()
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0)  # shape (1, 76, 384, 4)
            q_values = self.model(state_t)  # shape (1, 3)
        return q_values.cpu().numpy()

    def act(self, state, epsilon=0.004):
        if random.random() > epsilon:
            qval = self.predict(state)
            return np.argmax(qval.flatten())
        else:
            return random.choice([0, 1, 2])

    def remember(self, state, next_state, action, reward, done, location):
        self.location = location
        self.memory.append((state, next_state, action, reward, done))

    def learn(self):
        if len(self.memory) < self.batch_size:
            print("Not enough samples to learn.")
            return

        batch = random.sample(self.memory, self.batch_size)
        states = np.array([sample[0] for sample in batch], dtype=np.float32)
        next_states = np.array([sample[1] for sample in batch], dtype=np.float32)
        actions = np.array([sample[2] for sample in batch])
        rewards = np.array([sample[3] for sample in batch], dtype=np.float32)
        dones = np.array([sample[4] for sample in batch], dtype=np.bool_)

        states_t = torch.FloatTensor(states)
        next_states_t = torch.FloatTensor(next_states)
        actions_t = torch.LongTensor(actions)
        rewards_t = torch.FloatTensor(rewards)
        dones_t = torch.BoolTensor(dones)

        self.model.train()
        q_current = self.model(states_t)  # (batch_size, 3)
        q_current_a = q_current.gather(1, actions_t.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            q_next = self.model(next_states_t)  # (batch_size, 3)
            q_next_max = q_next.max(dim=1)[0]

        target = rewards_t + (~dones_t) * (self.gamma * q_next_max)
        new_q = q_current_a + self.alpha * (target - q_current_a)

        loss = self.criterion(q_current_a, new_q.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        print("LOSS:", loss.item())
        self.loss_history.append(loss.item())
