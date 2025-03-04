# train.py

import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import os
import argparse
import time

from model import DQNAgent
from game_env import DinoGameEnv
from config import Config


def plot_metrics(rewards, avg_rewards, filename):
    """Plot rewards and save to file."""
    plt.figure(figsize=(12, 5))
    plt.plot(rewards, alpha=0.3, label='Rewards')
    plt.plot(avg_rewards, label='Average Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training Rewards')
    plt.legend()
    plt.savefig(filename)
    plt.close()


def train(config_params, load_checkpoint=None):
    """
    Train the DQN agent on the Dino game.

    Args:
        config_params: Configuration parameters
        load_checkpoint: Filename to load checkpoint from, if any
    """
    # Set up environment
    env = DinoGameEnv(render_mode='human' if config_params.render_training else None)

    # Set up agent
    agent = DQNAgent(
        state_dim=env.observation_space_shape[0],
        action_dim=env.action_space,
        config=config_params
    )

    # Load checkpoint if specified
    if load_checkpoint:
        agent.load_model(load_checkpoint)

    # Training metrics
    rewards = []
    avg_rewards = []
    best_eval_reward = -float('inf')
    reward_history = deque(maxlen=100)

    # Create checkpoint directory
    os.makedirs(config_params.checkpoint_dir, exist_ok=True)

    print("Starting training...")
    start_time = time.time()

    for episode in range(1, config_params.episodes + 1):
        state = env.reset()
        total_reward = 0
        episode_info = {}

        # Show episode info if rendering
        if config_params.render_training:
            print(f"Starting Episode {episode}/{config_params.episodes}, Epsilon: {agent.epsilon:.4f}")
            # Small pause between episodes for better visibility
            time.sleep(1)

        for step in range(config_params.max_steps):
            # Select action
            action = agent.select_action(state)

            # Take action
            next_state, reward, episode_done, episode_info = env.step(action)

            # Store experience
            agent.remember(state, action, reward, next_state, episode_done)

            # Update state and reward
            state = next_state
            total_reward += reward

            # Train the model
            if len(agent.memory) >= agent.batch_size:
                _ = agent.replay()

            if episode_done:
                break

        # Record metrics
        rewards.append(total_reward)
        reward_history.append(total_reward)
        avg_rewards.append(np.mean(reward_history))

        # Print progress
        if episode % 10 == 0 or config_params.render_training:
            print(f"Episode: {episode}, Score: {episode_info.get('score', 0)}, Reward: {total_reward:.2f}, "
                  f"Avg Reward: {np.mean(reward_history):.2f}, Epsilon: {agent.epsilon:.4f}")

        # Periodically save checkpoint
        if episode % config_params.checkpoint_frequency == 0:
            agent.save_model(f"dino_dqn_episode_{episode}.pth")

            # Plot and save metrics
            plot_metrics(rewards, avg_rewards, f"{config_params.checkpoint_dir}/rewards_episode_{episode}.png")

        # Periodically evaluate the agent
        if episode % config_params.eval_frequency == 0:
            eval_reward = evaluate(agent, config_params, episode)
            if eval_reward > best_eval_reward:
                best_eval_reward = eval_reward
                agent.save_model("dino_dqn_best.pth")

    # Save final model
    agent.save_model("dino_dqn_final.pth")

    # Plot final metrics
    plot_metrics(rewards, avg_rewards, f"{config_params.checkpoint_dir}/rewards_final.png")

    print(f"Training completed in {(time.time() - start_time) / 60:.2f} minutes")
    print(f"Best evaluation reward: {best_eval_reward:.2f}")


def evaluate(agent, config_params, episode):
    """
    Evaluate the agent's performance.

    Args:
        agent: DQN agent
        config_params: Configuration parameters
        episode: Current episode number

    Returns:
        float: Average reward over evaluation episodes
    """
    print(f"\nEvaluating agent at episode {episode}...")
    eval_env = DinoGameEnv(render_mode='human' if config_params.render_eval else None)
    eval_rewards = []

    for eval_ep in range(config_params.eval_episodes):
        state = eval_env.reset()
        total_reward = 0
        eval_done = False
        eval_info = {}

        # Small delay to see the start of each evaluation episode
        if config_params.render_eval:
            time.sleep(0.5)

        while not eval_done:
            # Select action (no exploration)
            action = agent.select_action(state, training=False)

            # Take action
            next_state, reward, eval_done, eval_info = eval_env.step(action)

            # Update state and reward
            state = next_state
            total_reward += reward

        eval_rewards.append(total_reward)
        print(f"Eval Episode {eval_ep + 1}/{config_params.eval_episodes}, Score: {eval_info.get('score', 0)}, Reward: {total_reward:.2f}")

    avg_reward = np.mean(eval_rewards)
    print(f"Evaluation complete. Average reward: {avg_reward:.2f}\n")
    return avg_reward


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train DQN agent for Dino game')
    parser.add_argument('--load', type=str, help='Load model from checkpoint')
    parser.add_argument('--render', action='store_true', help='Enable rendering during training')

    args = parser.parse_args()
    config = Config()

    # Override config with command line arguments
    if args.render:
        config.render_training = True

    train(config, args.load)
