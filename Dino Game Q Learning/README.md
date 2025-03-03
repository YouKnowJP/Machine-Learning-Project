# Dino Game AI

An implementation of a Deep Q-Network (DQN) reinforcement learning agent that learns to play a custom version of the Chrome Dino game.

![Dino Game](https://storage.googleapis.com/gweb-uniblog-publish-prod/original_images/Social_dino-with-hat.gif)

## Overview

This project uses deep reinforcement learning to train an AI agent to play the classic Chrome T-Rex (Dino) game. The agent uses a Deep Q-Network (DQN) to learn optimal strategies for jumping over cacti, ducking under birds, and collecting power-ups.

### Features

- Custom implementation of the Chrome Dino game in Pygame
- Deep Q-Network (DQN) implementation with PyTorch
- Experience replay for efficient learning
- Configurable hyperparameters for experimentation
- Training and evaluation scripts with visualization
- Checkpointing system to save and load model progress

## Requirements

- Python 3.6+
- PyTorch
- Pygame
- NumPy
- Matplotlib

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/dino-game-ai.git
   cd dino-game-ai
   ```

2. Install the required packages:
   ```
   pip install torch pygame numpy matplotlib
   ```

## Project Structure

- `main.py`
