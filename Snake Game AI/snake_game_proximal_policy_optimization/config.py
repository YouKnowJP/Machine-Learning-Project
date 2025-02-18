# config.py

# -----------------
# Game configuration
# -----------------
BLOCK_SIZE = 20          # Size of one block (in pixels)
SPEED = 200              # Game speed (frames per second)

# --------------------------
# PPO training configuration
# --------------------------
LR = 0.001               # Learning rate
GAMMA = 0.99             # Discount factor for PPO
SAVE_INTERVAL = 10       # Save model every 10 PPO updates
CLIP_PARAM = 0.2         # PPO clipping parameter
ENTROPY_COEF = 0.01      # Coefficient for entropy bonus
VALUE_COEF = 0.5         # Coefficient for value loss
PPO_EPOCHS = 4           # Number of epochs per PPO update
ROLLOUT_STEPS = 2000     # Number of steps to rollout per update
NUM_UPDATES = 1000       # Total number of PPO updates

# -----------------
# Model saving
# -----------------
MODEL_FOLDER = "ppo_model"
MODEL_FILE = "ppo_model.pth"
