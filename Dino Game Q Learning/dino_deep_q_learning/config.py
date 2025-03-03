"""
Configuration parameters for the Dino game DQN agent.
"""


class Config:
    # Model parameters
    hidden_dims = [128, 64]

    # Training parameters
    batch_size = 64
    buffer_size = 100000
    gamma = 0.99  # Discount factor
    learning_rate = 0.0005

    # Exploration parameters
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay = 0.995

    # Target network update frequency
    target_update = 100

    # Training settings
    episodes = 2000
    max_steps = 100000
    eval_frequency = 50
    checkpoint_frequency = 100

    # Checkpoint directory
    checkpoint_dir = "Checkpoints path"

    # CUDA settings
    use_cuda = True

    # Rendering settings - Changed to True to enable visualization
    render_training = True
    render_eval = True

    # Evaluation settings
    eval_episodes = 5
