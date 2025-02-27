#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 23:11:36 2025

@author: youknowjp
"""

# Training configuration
MAX_GAMES: int = 200            # Total number of games to train
MAX_MEMORY: int = 100_000       # Maximum size for the replay memory
BATCH_SIZE: int = 1_000         # Batch size for training from replay memory
LR: float = 0.001               # Learning rate
GAMMA: float = 0.9              # Discount factor for Q-learning

# Exploration parameters
INITIAL_EPSILON: float = 1.0    # Starting exploration rate
MIN_EPSILON: float = 0.05       # Minimum exploration rate
DECAY_RATE: float = 0.995       # Decay rate per game for epsilon

# Model saving/loading configuration
MODEL_FOLDER: str = "models"    # Path to the model folder
MODEL_FILE: str = "model.pth"   # Model file name
SAVE_INTERVAL: int = 50         # Save model every N games

# Files for persisting additional training state
MEMORY_FILE: str = "memory.npy"          # File to save the experience memory
GAME_COUNT_FILE: str = "game_count.txt"  # File to save the number of games played

# Game configuration
BLOCK_SIZE: int = 20          # Size of one block (in pixels)
SPEED: int = 500              # Game speed (frames per second)

# Additional state representation configuration
STACK_SIZE: int = 3           # Number of frames to stack for richer state representation
