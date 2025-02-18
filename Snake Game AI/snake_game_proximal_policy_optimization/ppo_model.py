# ppo_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class PPOModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, action_dim):
        super(PPOModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        # Actor head: outputs logits for each action (for 3 moves)
        self.fc_actor = nn.Linear(hidden_dim, action_dim)
        # Critic head: outputs a single state value
        self.fc_critic = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        policy_logits = self.fc_actor(x)
        value = self.fc_critic(x)
        return policy_logits, value

    def save(self, file_path):
        torch.save(self.state_dict(), file_path)
        print(f"Model saved to {file_path}")

    def load(self, file_path):
        self.load_state_dict(torch.load(file_path))
        print(f"Model loaded from {file_path}")
