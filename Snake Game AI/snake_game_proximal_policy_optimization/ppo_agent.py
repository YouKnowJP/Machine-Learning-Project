# ppo_agent.py

import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from game import SnakeGameAI, Direction, Point
import config
from ppo_model import PPOModel

class PPOAgent:
    def __init__(self):
        self.game = SnakeGameAI()
        self.input_dim = 11    # Number of features in the state representation
        self.hidden_dim = 256
        self.action_dim = 3    # Three possible actions
        self.model = PPOModel(self.input_dim, self.hidden_dim, self.action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.LR)
        self.clip_param = config.CLIP_PARAM
        self.gamma = config.GAMMA
        self.entropy_coef = config.ENTROPY_COEF
        self.value_coef = config.VALUE_COEF
        self.epochs = config.PPO_EPOCHS

    def get_state(self):
        head = self.game.snake[0]
        block_size = config.BLOCK_SIZE
        point_l = Point(head.x - block_size, head.y)
        point_r = Point(head.x + block_size, head.y)
        point_u = Point(head.x, head.y - block_size)
        point_d = Point(head.x, head.y + block_size)

        # Direction flags
        dir_l = self.game.direction == Direction.LEFT
        dir_r = self.game.direction == Direction.RIGHT
        dir_u = self.game.direction == Direction.UP
        dir_d = self.game.direction == Direction.DOWN

        # Danger checks: straight, right, left
        danger_straight = ((dir_r and self.game.is_collision(point_r)) or
                           (dir_l and self.game.is_collision(point_l)) or
                           (dir_u and self.game.is_collision(point_u)) or
                           (dir_d and self.game.is_collision(point_d)))
        danger_right = ((dir_u and self.game.is_collision(point_r)) or
                        (dir_d and self.game.is_collision(point_l)) or
                        (dir_l and self.game.is_collision(point_u)) or
                        (dir_r and self.game.is_collision(point_d)))
        danger_left = ((dir_d and self.game.is_collision(point_r)) or
                       (dir_u and self.game.is_collision(point_l)) or
                       (dir_r and self.game.is_collision(point_u)) or
                       (dir_l and self.game.is_collision(point_d)))

        state = [
            int(danger_straight),
            int(danger_right),
            int(danger_left),
            int(dir_l),
            int(dir_r),
            int(dir_u),
            int(dir_d),
            int(self.game.food.x < head.x),  # Food is to the left
            int(self.game.food.x > head.x),  # Food is to the right
            int(self.game.food.y < head.y),  # Food is above
            int(self.game.food.y > head.y)   # Food is below
        ]
        return np.array(state, dtype=int)

    def select_action(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32)
        logits, value = self.model(state_tensor)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob, value

    def compute_returns(self, rewards, dones, last_value):
        returns = []
        R = last_value
        # Compute discounted returns (backwards)
        for r, done in zip(reversed(rewards), reversed(dones)):
            if done:
                R = 0
            R = r + self.gamma * R
            returns.insert(0, R)
        return returns

    def train(self, rollout_steps=config.ROLLOUT_STEPS):
        states = []
        actions = []
        log_probs = []
        rewards = []
        dones = []
        values = []

        state = self.get_state()
        for _ in range(rollout_steps):
            action, log_prob, value = self.select_action(state)
            # Convert action integer into one-hot encoding for game.play_step
            action_onehot = [0] * self.action_dim
            action_onehot[action] = 1
            reward, done, score = self.game.play_step(action_onehot)

            states.append(state)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            dones.append(done)
            values.append(value)

            state = self.get_state()
            if done:
                self.game.reset()

        # Convert collected data to tensors
        states_tensor = torch.tensor(states, dtype=torch.float32)
        actions_tensor = torch.tensor(actions, dtype=torch.int64)
        old_log_probs = torch.stack(log_probs)
        values_tensor = torch.cat(values).detach().squeeze()

        # Estimate value for the last state in the rollout
        state_tensor = torch.tensor(state, dtype=torch.float32)
        _, last_value = self.model(state_tensor)
        last_value = last_value.item()

        returns = self.compute_returns(rewards, dones, last_value)
        returns_tensor = torch.tensor(returns, dtype=torch.float32)

        advantages = returns_tensor - values_tensor

        # PPO update: perform multiple epochs over the collected rollout
        for _ in range(self.epochs):
            logits, values_new = self.model(states_tensor)
            dist = torch.distributions.Categorical(logits=logits)
            new_log_probs = dist.log_prob(actions_tensor)
            ratio = torch.exp(new_log_probs - old_log_probs)

            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = F.mse_loss(values_new.squeeze(), returns_tensor)
            entropy_loss = -dist.entropy().mean()

            loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        print(f"PPO update completed. Total reward in rollout: {sum(rewards)}")
