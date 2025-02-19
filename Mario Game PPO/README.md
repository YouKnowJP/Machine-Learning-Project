# Mario RL Agent

This repository contains code for training a reinforcement learning (RL) agent to play *Super Mario Bros.* using the Proximal Policy Optimization (PPO) algorithm.

## Table of Contents

- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [PPO Explanation](#ppo-explanation)
- [File Structure](#file-structure)
- [Contributing](#contributing)
- [License](#license)

## Introduction

This project demonstrates how to train an RL agent to play *Super Mario Bros.* using the PPO algorithm from the `Stable Baselines3` library. The code is modularized for better maintainability and configurability.

## Prerequisites

- Python 3.6+
- `pip` package manager

## Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/yourusername/mario-rl-agent.git
    cd mario-rl-agent
    ```

2. **Create a virtual environment (recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Linux/macOS
    venv\Scripts\activate  # On Windows
    ```

3. **Install required packages:**

    ```bash
    pip install -r requirements.txt
    ```

    **`requirements.txt` should contain:**
    ```
    gym_super_mario_bros==7.3.0
    nes_py
    stable-baselines3[extra]
    torch==1.10.1+cu113
    torchvision==0.11.2+cu113
    torchaudio==0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
    matplotlib
    ```

## Usage

1. **Run the main script:**

    ```bash
    python main.py
    ```

    This will:
    - Create and preprocess the Mario environment.
    - Train the PPO model.
    - Test the trained model.

2. **Optional steps:**
    - To run the game with random actions for visualization, uncomment the `run_random_actions(config)` line in `main.py`.
    - To visualize the stacked frames, uncomment the `visualize_state(state)` lines in `main.py`.

## Configuration

The configuration settings are located in `config.py`. You can modify the following parameters:

- `game_name`: Name of the *Super Mario Bros.* environment.
- `simple_movement`: Use simplified controls.
- `grayscale`: Convert the environment to grayscale.
- `frame_stack`: Number of stacked frames.
- `checkpoint_dir`: Directory to save model checkpoints.
- `log_dir`: Directory to store TensorBoard logs.
- `model_save_path`: Path to save the final trained model.
- `learning_rate`: Learning rate for the PPO optimizer.
- `n_steps`: Number of steps per training update.
- `total_timesteps`: Total number of timesteps for training.
- `check_freq`: Frequency of saving model checkpoints.
- `render_test`: Boolean to render the test.
- `max_test_steps`: Maximum steps for the test run.

## PPO Explanation

**Proximal Policy Optimization (PPO)** is a widely used policy gradient algorithm in reinforcement learning. It is designed for stability and sample efficiency. Below are the key concepts:

1. **Policy Gradient:**
   - PPO directly optimizes the policy (agent's strategy) based on observed rewards.
2. **Objective Function:**
   - PPO maximizes an objective function that promotes high-reward actions.
3. **Clipping Mechanism:**
   - Limits the magnitude of policy updates to ensure stable training.
   - Compares new action probabilities with old probabilities and clips the ratio within a specified range.
4. **Surrogate Objective:**
   - Uses a surrogate function with clipping to approximate the true objective and maintain stability.
5. **Advantages:**
   - Stable and easy to implement.
   - Performs well with relatively few hyperparameters.
   - Sample-efficient, meaning it can learn effectively with a moderate amount of data.
6. **Why PPO is used here:**
   - Suitable for complex environments like *Super Mario Bros.*
   - Stable Baselines3 provides an optimized implementation.
   - Balances performance and ease of use.

In summary, PPO is a reliable RL algorithm that efficiently balances exploration and exploitation, making it well-suited for training agents in complex environments like *Super Mario Bros.*.

## File Structure

```
├── mario-rl-agent
│   ├── main.py               # Main training script
│   ├── config.py             # Configuration settings
│   ├── model.py              # PPO model implementation
│   ├── utils.py              # Utility functions
│   ├── requirements.txt      # Required dependencies
│   ├── README.md             # Documentation
│   ├── saved_models/         # Directory to store trained models
│   ├── logs/                 # Directory for TensorBoard logs
```
