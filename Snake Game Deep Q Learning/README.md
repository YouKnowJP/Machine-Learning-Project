# Deep Q Learning Snake Game

This project implements a Deep Q-Learning agent that learns to play the classic Snake Game using reinforcement learning techniques. The agent is built with PyTorch and leverages experience replay, an epsilon-greedy policy, and a simple two-layer neural network to approximate Q-values.

## Project Structure

- **agent.py**  
  Contains the `Agent` class that manages the training loop, memory replay, action selection, and saving/loading checkpoints. It uses Deep Q-Learning to train an agent for the Snake game.

- **config.py**  
  Holds all the configuration parameters such as training settings (e.g., `MAX_GAMES`, `BATCH_SIZE`, `LR`, etc.), file paths for saving the model and training state, and game-specific parameters (e.g., `BLOCK_SIZE`, `SPEED`).

- **game.py**  
  Implements the Snake game using Pygame. It defines the game logic including snake movement, collision detection, and food placement.

- **helper.py**  
  Provides helper functions (e.g., `plot`) to display live training progress using Matplotlib and IPython's display.

- **model.py**  
  Defines the neural network (`Linear_QNet`) used for approximating Q-values, as well as the `QTrainer` class that handles the training step.

- **main.py**  
  The entry point for training. Running this file starts the training loop where the agent interacts with the Snake game, learns from its experiences, and periodically saves checkpoints.

## How It Works

1. **State Representation:**  
   The agent represents the game state using a fixed-size vector that includes:
   - Danger in three directions (straight, right, left)
   - Current moving direction (one-hot encoded)
   - Food location relative to the snake's head

2. **Deep Q-Learning:**  
   The agent uses a neural network to approximate Q-values for each possible action. The training uses experience replay (a memory of past state-action transitions) to sample batches of experiences for stable learning.

3. **Epsilon-Greedy Policy:**  
   During training, the agent selects actions according to an epsilon-greedy strategy. This ensures that the agent explores the environment sufficiently before exploiting learned strategies.

4. **Checkpointing and Resuming Training:**  
   - The project saves the model weights, the number of games played, and the replay memory to disk.
   - Checkpoints are saved every `SAVE_INTERVAL` games (as defined in `config.py`).
   - When training resumes, the agent loads these saved states to continue from where it left off.

## Setup and Installation

### Prerequisites

- Python 3.7+
- [PyTorch](https://pytorch.org/)
- [Pygame](https://www.pygame.org/)
- [NumPy](https://numpy.org/)
- [Matplotlib](https://matplotlib.org/)
- [IPython](https://ipython.org/)

### Installation

1. **Clone the repository:**

   ```bash
   git clone <repository-url>
   cd "Snake Game Deep Q Learning"
   ```

2. **Install required packages:**

   If you have a `requirements.txt`, run:
   ```bash
   pip install -r requirements.txt
   ```

   Otherwise, install packages manually:
   ```bash
   pip install torch pygame numpy matplotlib ipython
   ```

3. **Configure the project:**

   Open `config.py` and adjust the parameters as needed. For example, you can change:
   - `MAX_GAMES`: Total number of games to train.
   - `SAVE_INTERVAL`: Interval (in number of games) to save checkpoints.
   - `MODEL_FOLDER`: The directory where the model and training state are saved.

## Running the Project

To start training the agent, run:

```bash
python main.py
```

- **Training Progress:**  
  The training progress will be displayed in a live plot (using Matplotlib) showing the score and the mean score over games.

- **Checkpoints:**  
  The model is automatically saved at regular intervals (as specified by `SAVE_INTERVAL` in `config.py`). If a saved model is detected when the training starts, the agent will resume training from the saved state.

## Technical Details

- **State Vector:**  
  The state vector is an 11-dimensional NumPy array that captures the environment's state, including collision risks and food location.

- **Neural Network:**  
  The `Linear_QNet` is a simple two-layer fully-connected neural network with ReLU activation in the hidden layer. It outputs Q-values for three possible actions.

- **Training:**  
  The agent uses two types of memory:
  - **Short-term memory:** Training is done immediately after each move.
  - **Long-term memory:** Batches of experiences are sampled randomly from a replay memory to update the model.

- **Saving & Loading:**  
  The project uses Python's `pickle` module to save and load the replay memory, and PyTorch's `torch.save`/`torch.load` for the model weights.

## Future Improvements

- **Improved Network Architecture:** Experiment with deeper or convolutional architectures.
- **Hyperparameter Tuning:** Adjust parameters like learning rate, epsilon decay, and batch size for improved performance.
- **Additional Features:** Implement more sophisticated exploration strategies (e.g., prioritized experience replay).

## Acknowledgments

- This project is inspired by many Deep Q-Learning tutorials and the classic Snake game.
- Special thanks to the [Pygame community](https://www.pygame.org/contribute.html) for their contributions.
