# deep_q_agent.py
import os
import torch
import random
import numpy as np
import pickle  # Use pickle for saving/loading the memory
from collections import deque
from typing import List
from game import SnakeGameAI, Direction, Point
from deep_q_model import Linear_QNet, QTrainer
from helper import plot
import config

# Use configuration values from config.py
MAX_MEMORY = config.MAX_MEMORY
BATCH_SIZE = config.BATCH_SIZE
LR = config.LR
GAMMA = config.GAMMA
EPSILON_DECAY = config.EPSILON_DECAY
MAX_GAMES = config.MAX_GAMES
SAVE_INTERVAL = config.SAVE_INTERVAL
MODEL_FOLDER = config.MODEL_FOLDER
MODEL_FILE = os.path.join(MODEL_FOLDER, config.MODEL_FILE)
MEMORY_FILE = os.path.join(MODEL_FOLDER, config.MEMORY_FILE)
GAME_COUNT_FILE = os.path.join(MODEL_FOLDER, config.GAME_COUNT_FILE)


class Agent:
    """
    Reinforcement Learning agent for the Snake game using Q-learning.
    """

    def __init__(self) -> None:
        self.n_games = 0
        self.epsilon = 0  # Exploration rate
        self.gamma = GAMMA
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game: SnakeGameAI) -> np.ndarray:
        head = game.snake[0]
        block_size = config.BLOCK_SIZE
        point_l = Point(head.x - block_size, head.y)
        point_r = Point(head.x + block_size, head.y)
        point_u = Point(head.x, head.y - block_size)
        point_d = Point(head.x, head.y + block_size)

        # Direction flags
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        # Danger checks: straight, right, left
        danger_straight = ((dir_r and game.is_collision(point_r)) or
                           (dir_l and game.is_collision(point_l)) or
                           (dir_u and game.is_collision(point_u)) or
                           (dir_d and game.is_collision(point_d)))
        danger_right = ((dir_u and game.is_collision(point_r)) or
                        (dir_d and game.is_collision(point_l)) or
                        (dir_l and game.is_collision(point_u)) or
                        (dir_r and game.is_collision(point_d)))
        danger_left = ((dir_d and game.is_collision(point_r)) or
                       (dir_u and game.is_collision(point_l)) or
                       (dir_r and game.is_collision(point_u)) or
                       (dir_l and game.is_collision(point_d)))

        state = [
            int(danger_straight),
            int(danger_right),
            int(danger_left),
            int(dir_l),
            int(dir_r),
            int(dir_u),
            int(dir_d),
            int(game.food.x < head.x),  # Food left
            int(game.food.x > head.x),  # Food right
            int(game.food.y < head.y),  # Food up
            int(game.food.y > head.y)  # Food down
        ]
        return np.array(state, dtype=int)

    def remember(self, state: np.ndarray, action: List[int],
                 reward: float, next_state: np.ndarray, done: bool) -> None:
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self) -> None:
        if len(self.memory) > 0:
            mini_sample = (random.sample(self.memory, BATCH_SIZE)
                           if len(self.memory) > BATCH_SIZE else list(self.memory))
            states, actions, rewards, next_states, dones = zip(*mini_sample)
            self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state: np.ndarray, action: List[int],
                           reward: float, next_state: np.ndarray, done: bool) -> None:
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state: np.ndarray) -> List[int]:
        self.epsilon = max(EPSILON_DECAY - self.n_games, 0)
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32)
            prediction = self.model(state_tensor)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        return final_move


def train() -> None:
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0

    agent = Agent()
    game = SnakeGameAI()

    # Check if the model file exists; if yes, load model weights,
    # then load the game count and experience memory.
    if os.path.exists(MODEL_FILE):
        print("Model found. Loading existing weights...")
        agent.model.load()
        if os.path.exists(GAME_COUNT_FILE):
            with open(GAME_COUNT_FILE, "r") as f:
                agent.n_games = int(f.read().strip())
            print(f"Resuming from game {agent.n_games}")
        if os.path.exists(MEMORY_FILE):
            print("Loading experience memory...")
            with open(MEMORY_FILE, "rb") as f:
                loaded_memory = pickle.load(f)
            agent.memory = deque(loaded_memory, maxlen=MAX_MEMORY)
    else:
        print("No existing model found. Training from scratch.")

    # Training loop
    while agent.n_games < MAX_GAMES:
        state_old = agent.get_state(game)
        final_move = agent.get_action(state_old)
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)
        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score

            print(f"Game: {agent.n_games}, Score: {score}, Record: {record}")
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

            # Save a checkpoint once per finished game if it's time.
            if agent.n_games % SAVE_INTERVAL == 0:
                agent.trainer.save_model()
                print(f"Checkpoint saved at {agent.n_games} games.")
                with open(GAME_COUNT_FILE, "w") as f:
                    f.write(str(agent.n_games))
                with open(MEMORY_FILE, "wb") as f:
                    pickle.dump(list(agent.memory), f)

    # Final save after training is complete.
    agent.trainer.save_model()
    print(f"Training finished after {MAX_GAMES} games. Model saved.")
    with open(GAME_COUNT_FILE, "w") as f:
        f.write(str(agent.n_games))
    with open(MEMORY_FILE, "wb") as f:
        pickle.dump(list(agent.memory), f)


if __name__ == '__main__':
    train()
