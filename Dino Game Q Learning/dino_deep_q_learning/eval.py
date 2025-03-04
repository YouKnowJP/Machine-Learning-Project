# eval.py

import argparse
import numpy as np
import time

from model import DQNAgent
from game_env import DinoGameEnv
from config import Config


def evaluate(model_path, num_episodes=10, render=True):
    """
    Evaluate a trained model on the Dino game.

    Args:
        model_path: Path to the saved model checkpoint
        num_episodes: Number of episodes to run
        render: Whether to render the game
    """
    # Load configuration
    config = Config()

    # Set up environment
    env = DinoGameEnv(render_mode='human' if render else None)

    # Set up agent
    agent = DQNAgent(
        state_dim=env.observation_space_shape[0],
        action_dim=env.action_space,
        config=config
    )

    # Load model
    if not agent.load_model(model_path):
        print(f"Could not load model from {model_path}")
        return

    # Run evaluation episodes
    scores = []
    rewards = []

    for episode in range(1, num_episodes + 1):
        state = env.reset()
        done = False
        episode_reward = 0
        info = {'score': 0}  # Initialize info with a default value

        # Delay to let player see the game
        if render:
            time.sleep(1)

        while not done:
            # Select action (no exploration)
            action = agent.select_action(state, training=False)

            # Take action
            next_state, reward, done, info = env.step(action)

            # Update state and reward
            state = next_state
            episode_reward += reward

            # Optional: slow down for better visualization
            if render:
                time.sleep(0.01)

        scores.append(info['score'])
        rewards.append(episode_reward)

        print(f"Episode {episode}/{num_episodes}, Score: {info['score']}, Reward: {episode_reward:.2f}")

    # Print summary statistics
    print("\nEvaluation Summary:")
    print(f"Average Score: {np.mean(scores):.2f} ± {np.std(scores):.2f}")
    print(f"Average Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"Max Score: {max(scores)}")
    print(f"Min Score: {min(scores)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate DQN agent for Dino game')
    parser.add_argument('--model', type=str, required=True, help='Path to model file')
    parser.add_argument('--episodes', type=int, default=10, help='Number of episodes to run')
    parser.add_argument('--no-render', action='store_true', help='Disable rendering')

    args = parser.parse_args()
    evaluate(args.model, args.episodes, not args.no_render)
