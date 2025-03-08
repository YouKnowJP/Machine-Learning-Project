# **Pong AI Using NEAT (NeuroEvolution of Augmenting Topologies)**
🚀 **An AI-powered Pong game where neural networks evolve over time using the NEAT algorithm. The AI learns to play Pong by competing against itself, improving its paddle control and reaction time.**

---

## **Table of Contents**
- [Overview](#overview)
- [Installation](#installation)
- [How It Works](#how-it-works)
- [Running the Project](#running-the-project)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Training the AI](#training-the-ai)
- [Playing Against the AI](#playing-against-the-ai)
- [Known Issues & Troubleshooting](#known-issues--troubleshooting)
- [Contributing](#contributing)
- [License](#license)

---

## **Overview**
This project implements a **Pong AI** using the **NEAT (NeuroEvolution of Augmenting Topologies)** algorithm. The AI starts with random paddle movements and **evolves through self-play**, improving over generations.

### **Key Features**
- **NEAT Algorithm** for evolving neural networks
- **AI vs AI Training**: Two AI agents play against each other to evolve
- **Human vs AI Mode**: Test AI against a human player
- **Customizable Training Parameters** in `config.txt`
- **Checkpoints** to resume training

---

## ⚡ **Installation**
To run this project, install the required dependencies:

```sh
git clone https://github.com/YouKnowJP/Machine-Learning-Project.git
cd Machine-Learning-Project/Pong-AI

# Install required libraries
pip install pygame neat-python
```

---

## **How It Works**
1. **AI Training**
   - Two AI players start with random actions.
   - NEAT evaluates their performance using a **fitness function**.
   - The **best networks** evolve over multiple generations.

2. **Human vs AI Mode**
   - The trained AI model is loaded from `best.pickle`.
   - The AI controls the **right paddle** while the human plays using:
     - `W` = Move up
     - `S` = Move down

3. **Stopping Condition**
   - The game stops if a player scores **or** if the total ball hits reach **50**.

---

## **Running the Project**
To start the **training process**, run:

```sh
python main.py
```

You’ll be asked to **train** or **play against AI**:
```
Enter 'train' to train a new AI or 'play' to play against the best AI: 
```

### **Train the AI**
```sh
python main.py
# Then enter:
train
```
- This runs **NEAT training**, evolving the AI over generations.
- The best AI is saved to **`best.pickle`**.

### **Play Against AI**
```sh
python main.py
# Then enter:
play
```
- The AI loads from `best.pickle` and plays **against a human**.
- You control the **left paddle** using `W` and `S`.

---

## **Project Structure**
```
Pong-AI/
│── pong.py                # Pong game logic
│── main.py                # Main script for training & playing
│── config.txt             # NEAT algorithm configuration
│── best.pickle            # Best trained AI model
│── neat-checkpoint-*      # Checkpoints for resuming training
│── README.md              # Documentation
```

---

## **Configuration**
You can tweak the **training settings** in `config.txt`. Some key parameters:

```ini
[DefaultGenome]
num_inputs          = 3
num_outputs         = 3
fitness_threshold   = 100  # AI stops training when reaching this fitness
```

---

## **Training the AI**
1. **Start training**:
   ```sh
   python main.py
   ```
2. **How AI learns**:
   - AI plays **against itself**.
   - Fitness increases based on **ball hits and survival time**.
   - Best AI gets **saved automatically**.
3. **Resume training from checkpoint**:
   ```sh
   python main.py
   ```
   - NEAT automatically loads from `neat-checkpoint-*`.

---

## **Playing Against the AI**
Once trained, you can play against the AI:
```sh
python main.py
```
- AI **controls right paddle**.
- **You control the left paddle** using:
  - `W` = Move Up
  - `S` = Move Down

---

## **Known Issues & Troubleshooting**
| Issue | Possible Cause | Solution |
|--------|-----------------|-------------|
| **Game does not start** | Missing `pygame` | Run `pip install pygame` |
| **403 Git Permission Error** | No write access | Fork the repo & push to your fork |
| **NEAT training is slow** | Large population size | Reduce `pop_size` in `config.txt` |
