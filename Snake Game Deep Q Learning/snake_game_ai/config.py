# config.py
# Training configuration
MAX_GAMES = 500            # Total number of games to train
MAX_MEMORY = 100_000       # Maximum size for the replay memory
BATCH_SIZE = 1_000         # Batch size for training from replay memory
LR = 0.001                 # Learning rate
GAMMA = 0.9                # Discount factor for Q-learning
EPSILON_DECAY = 80         # Base for epsilon decay: epsilon = max(EPSILON_DECAY - n_games, 0)

# Model saving/loading configuration
MODEL_FOLDER = "Model Folder File Path"
MODEL_FILE = "model.pth"   # Only the file name; folder is specified above
SAVE_INTERVAL = 50  # Save model every 1,000,000 games

# Files for persisting additional training state
MEMORY_FILE = "memory.npy"          # To save the experience memory
GAME_COUNT_FILE = "game_count.txt"  # To save the number of games played

# Game configuration
BLOCK_SIZE = 20          # Size of one block (in pixels)
SPEED = 500              # Game speed (frames per second)
