# **Deep Q-Learning + A* Snake Game**  

This project implements a hybrid Deep Q-Learning and A* pathfinding agent that learns to play the classic Snake Game using reinforcement learning techniques. The agent is built with PyTorch and leverages experience replay, an epsilon-greedy policy, and a simple two-layer neural network to approximate Q-values. Additionally, the A* search algorithm is integrated to compute optimal paths when appropriate, enhancing decision-making.  

---

## **Project Structure**  

-   **`deep_q_agent.py`**: Contains the `Agent` class that manages training, memory replay, action selection, and saving/loading checkpoints. The agent can either use Deep Q-Learning or the A* algorithm for decision-making.  
-   **`astar.py`**: Implements the A* search algorithm to compute the shortest path from the snake’s head to the food while avoiding obstacles.  
-   **`config.py`**: Holds configuration parameters such as training settings (`MAX_GAMES`, `BATCH_SIZE`, `LR`, etc.), file paths for saving models and game state, and game parameters (`BLOCK_SIZE`, `SPEED`).  
-   **`game.py`**: Implements the Snake game using Pygame, handling movement, collision detection, food placement, and the A* pathfinding integration.  
-   **`helper.py`**: Provides utility functions (e.g., `plot`) for visualizing training progress using Matplotlib.  
-   **`deep_q_model.py`**: Defines the `Linear_QNet` neural network used for approximating Q-values, along with the `QTrainer` class that handles training.  
-   **`main.py`**: The entry point for training. Running this file starts the training loop where the agent interacts with the Snake game, learns from its experiences, and periodically saves checkpoints.  

---

## **How It Works**  

### **State Representation**  
The agent represents the game state using a fixed-size vector that includes:  
-   Danger in three directions (straight, right, left).  
-   Current moving direction (one-hot encoded).  
-   Food location relative to the snake's head.  
-   Normalized Manhattan distance to the food.  

### **Hybrid Decision-Making**  
The agent can choose actions based on either **Deep Q-Learning** or **A* search**:  

1. **Deep Q-Learning**:  
   - Uses a neural network to approximate Q-values for each possible action.  
   - Employs experience replay (a memory of past transitions) to sample batches of experiences for stable learning.  
   - Uses an **epsilon-greedy policy** for exploration and exploitation.  

2. **A* Pathfinding** (Optional):  
   - When enabled, the agent uses A* search to find the shortest, safest path to the food.  
   - If a valid path exists, the agent follows it; otherwise, it defaults to Deep Q-Learning.  

### **Checkpointing and Resuming Training**  
-   Saves model weights, replay memory, and the number of games played.  
-   Checkpoints are saved every `SAVE_INTERVAL` games (`config.py`).  
-   If training is restarted, the agent loads its saved state and continues from where it left off.  

---

## **Setup and Installation**  

### **Prerequisites**  

Ensure you have the following installed:  
-   Python 3.7+  
-   [PyTorch](https://pytorch.org/)  
-   [Pygame](https://www.pygame.org/)  
-   [NumPy](https://numpy.org/)  
-   [Matplotlib](https://matplotlib.org/)  
-   [IPython](https://ipython.org/)  

### **Installation**  

1.  **Clone the repository:**  

    ```bash
    git clone <repository-url>
    cd snake-game-deep-q-learning
    ```

    *(Replace `<repository-url>` with your repository URL.)*  

2.  **Install dependencies:**  

    If a `requirements.txt` is available:  

    ```bash
    pip install -r requirements.txt
    ```

    Otherwise, manually install:  

    ```bash
    pip install torch pygame numpy matplotlib ipython
    ```

3.  **Configure settings:**  
    Modify `config.py` to adjust:  
    -   `MAX_GAMES`: Number of training games.  
    -   `SAVE_INTERVAL`: Games per checkpoint save.  
    -   `MODEL_FOLDER`: Directory for model storage.  

---

## **Running the Project**  

### **Train the Agent**  
To start training, run:  

```bash
python main.py
```

### **Enable A* Pathfinding**  
To train the agent with **A* search enabled**, modify **`deep_q_agent.py`**:  

```python
agent = Agent(use_astar=True)
```

This will make the agent prioritize A* pathfinding when possible.  

---

## **Future Enhancements**  
-   Fine-tune hyperparameters to balance Deep Q-Learning and A* search.  
-   Test different heuristics for A* pathfinding to improve adaptability.  
-   Implement additional reinforcement learning strategies like Double Q-Learning or Dueling Q-Networks.  

---

This project serves as a hybrid approach to reinforcement learning and algorithmic pathfinding, demonstrating how **Deep Q-Learning** and **A* search** can work together to solve a classic game problem. 🚀