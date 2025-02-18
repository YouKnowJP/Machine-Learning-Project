# deep_q_model.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Union, Any
import config

class Linear_QNet(nn.Module):
    """
    A simple two-layer neural network for Q-learning.
    """
    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = config.MODEL_FOLDER
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_path = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_path)
        print(f"Model saved to {file_path}")

    def load(self, file_name='model.pth'):
        model_folder_path = config.MODEL_FOLDER
        file_path = os.path.join(model_folder_path, file_name)
        if os.path.exists(file_path):
            self.load_state_dict(torch.load(file_path))
            print(f"Model loaded from {file_path}")
        else:
            print("No existing model found. Training from scratch.")

class QTrainer:
    """
    Q-learning trainer that performs training steps and wraps saving.
    """
    def __init__(self, model: Linear_QNet, lr: float, gamma: float) -> None:
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(
        self,
        state: Union[torch.Tensor, Any],
        action: Union[torch.Tensor, Any],
        reward: Union[torch.Tensor, Any],
        next_state: Union[torch.Tensor, Any],
        done: Union[bool, list]
    ) -> None:
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        # Ensure tensors have a batch dimension.
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
            action = action.unsqueeze(0)
            reward = reward.unsqueeze(0)
            done = (done,)

        pred = self.model(state)
        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                with torch.no_grad():
                    next_pred = self.model(next_state[idx])
                Q_new = reward[idx] + self.gamma * torch.max(next_pred)
            target[idx][torch.argmax(action[idx]).item()] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()

    def save_model(self, file_name='model.pth'):
        self.model.save(file_name)
