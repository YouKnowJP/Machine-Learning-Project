# main.py

import argparse
import sys

from train import train
from eval import evaluate
from config import Config


def main():
    """Main entry point for training or evaluating the DQN agent."""
    parser = argparse.ArgumentParser(description='Dino Game DQN Agent')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Train command
    train_parser = subparsers.add_parser('train', help='Train the DQN agent')
    train_parser.add_argument('--load', type=str, help='Load model from checkpoint')
    train_parser.add_argument('--episodes', type=int, help='Number of episodes to train')
    train_parser.add_argument('--render', action='store_true', help='Render during training')

    # Evaluate command
    eval_parser = subparsers.add_parser('eval', help='Evaluate a trained agent')
    eval_parser.add_argument('--model', type=str, required=True, help='Path to model file')
    eval_parser.add_argument('--episodes', type=int, default=10, help='Number of episodes to run')
    eval_parser.add_argument('--no-render', action='store_true', help='Disable rendering')

    args = parser.parse_args()
    config = Config()

    if args.command == 'train':
        # Update config with command line arguments
        if hasattr(args, 'episodes') and args.episodes is not None:
            config.episodes = args.episodes
        if hasattr(args, 'render'):
            config.render_training = args.render

        # Run training
        train(config, args.load)

    elif args.command == 'eval':
        # Run evaluation
        evaluate(args.model, args.episodes, not args.no_render)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
