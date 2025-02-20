#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 23:11:36 2025

@author: youknowjp
"""

import os
import torch
import random
import numpy as np
import pickle
from collections import deque
from typing import List
from game import SnakeGameAI, Direction, Point
from deep_q_model import Linear_QNet, QTrainer
from helper import plot
import config

MAX_MEMORY = config.MAX_MEMORY
BATCH_SIZE = config.BATCH_SIZE
LR = config.LR
GAMMA = config.GAMMA
MAX_GAMES = config.MAX_GAMES
SAVE_INTERVAL = config.SAVE_INTERVAL
MODEL_FOLDER = config.MODEL_FOLDER
MODEL_FILE = os.path.join(MODEL_FOLDER, config.MODEL_FILE)
MEMORY_FILE = os.path.join(MODEL_FOLDER, config.MEMORY_FILE)
GAME_COUNT_FILE = os.path.join(MODEL_FOLDER, config.GAME_COUNT_FILE)

class Agent:
    def __init__(self, use_astar: bool = False) -> None:
        self.use_astar = use_astar
        self.n_games = 0
        self.epsilon = config.INITIAL_EPSILON
        self.gamma = GAMMA
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(12, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game: SnakeGameAI) -> np.ndarray:
        head = game.snake[0]
        block_size = config.BLOCK_SIZE
        point_l = Point(head.x - block_size, head.y)
        point_r = Point(head.x + block_size, head.y)
        point_u = Point(head.x, head.y - block_size)
        point_d = Point(head.x, head.y + block_size)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        danger_straight = (
            (dir_r and game.is_collision(point_r))
            or (dir_l and game.is_collision(point_l))
            or (dir_u and game.is_collision(point_u))
            or (dir_d and game.is_collision(point_d))
        )
        danger_right = (
            (dir_u and game.is_collision(point_r))
            or (dir_d and game.is_collision(point_l))
            or (dir_l and game.is_collision(point_u))
            or (dir_r and game.is_collision(point_d))
        )
        danger_left = (
            (dir_d and game.is_collision(point_r))
            or (dir_u and game.is_collision(point_l))
            or (dir_r and game.is_collision(point_u))
            or (dir_l and game.is_collision(point_d))
        )

        max_distance = (game.w + game.h) / block_size
        current_distance = (abs(head.x - game.food.x) + abs(head.y - game.food.y)) / block_size
        normalized_distance = current_distance / max_distance

        state = [
            int(danger_straight),
            int(danger_right),
            int(danger_left),
            int(dir_l),
            int(dir_r),
            int(dir_u),
            int(dir_d),
            int(game.food.x < head.x),
            int(game.food.x > head.x),
            int(game.food.y < head.y),
            int(game.food.y > head.y),
            normalized_distance
        ]
        return np.array(state, dtype=float)

    def remember(self, state: np.ndarray, action: List[int], reward: float, next_state: np.ndarray, done: bool) -> None:
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self) -> None:
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = list(self.memory)
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state: np.ndarray, action: List[int], reward: float, next_state: np.ndarray, done: bool) -> None:
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state: np.ndarray, game: SnakeGameAI) -> List[int]:
        """
        Returns the next move. If A* is enabled and a valid path is found,
        the action is determined by the first step of that path; otherwise,
        it falls back to the deep Q-learning policy.
        """
        if self.use_astar:
            path = game.get_astar_path()
            if len(path) > 1:
                next_point = path[1]  # The immediate next cell in the path
                dx = next_point.x - game.head.x
                dy = next_point.y - game.head.y

                # Determine desired direction based on the relative movement
                desired_direction = game.direction
                if dx > 0:
                    desired_direction = Direction.RIGHT
                elif dx < 0:
                    desired_direction = Direction.LEFT
                elif dy > 0:
                    desired_direction = Direction.DOWN
                elif dy < 0:
                    desired_direction = Direction.UP

                # Map the absolute direction to a relative move (straight/right/left)
                clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
                current_index = clock_wise.index(game.direction)
                if desired_direction == game.direction:
                    return [1, 0, 0]  # move straight
                elif clock_wise[(current_index + 1) % 4] == desired_direction:
                    return [0, 1, 0]  # turn right
                elif clock_wise[(current_index - 1) % 4] == desired_direction:
                    return [0, 0, 1]  # turn left

        # Fallback: use epsilon-greedy deep Q-learning policy.
        self.epsilon = max(config.MIN_EPSILON, config.INITIAL_EPSILON * (config.DECAY_RATE ** self.n_games))
        if random.random() < self.epsilon:
            move = random.randint(0, 2)
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32)
            prediction = self.model(state_tensor)
            move = torch.argmax(prediction).item()
        final_move = [0, 0, 0]
        final_move[move] = 1
        return final_move

def train() -> None:
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0

    # Enable A* integration by setting use_astar=True.
    agent = Agent(use_astar=True)
    game = SnakeGameAI()

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
                agent.memory = deque(pickle.load(f), maxlen=MAX_MEMORY)
    else:
        print("No existing model found. Training from scratch.")

    while agent.n_games < MAX_GAMES:
        state_old = agent.get_state(game)
        final_move = agent.get_action(state_old, game)  # Pass game instance here
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)
        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            previous_record = record
            record = max(record, score)
            print(f"Game: {agent.n_games}, Score: {score}, Record: {record}")

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

            if score > previous_record:
                agent.trainer.save_model()
                print(f"New high score! Model saved at game {agent.n_games} with score {score}.")
                with open(GAME_COUNT_FILE, "w") as f:
                    f.write(str(agent.n_games))
                with open(MEMORY_FILE, "wb") as f:
                    pickle.dump(list(agent.memory), f)

    agent.trainer.save_model()
    print(f"Training finished after {MAX_GAMES} games. Model saved.")
    with open(GAME_COUNT_FILE, "w") as f:
        f.write(str(agent.n_games))
    with open(MEMORY_FILE, "wb") as f:
        pickle.dump(list(agent.memory), f)

if __name__ == "__main__":
    train()
