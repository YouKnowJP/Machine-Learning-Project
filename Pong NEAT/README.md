# NEAT-Pong-Python  

This project implements the **NeuroEvolution of Augmenting Topologies (NEAT)** algorithm to train an AI agent to play **Pong** in Python. The AI evolves its decision-making ability through generations, learning how to control the paddle and respond to the ball's movement efficiently.

## What is NEAT?  
NEAT (**NeuroEvolution of Augmenting Topologies**) is an evolutionary algorithm designed to train artificial neural networks (ANNs) dynamically. Developed by **Kenneth Stanley**, NEAT enhances both the **structure** and **weights** of a neural network through genetic evolution. Unlike traditional machine learning models, NEAT starts with minimal networks and gradually **evolves complexity** while maintaining population diversity.  

### Key Features of NEAT:  
- **Dynamic Topology Evolution** – Networks start with simple structures and grow as necessary.  
- **Speciation** – Similar networks are grouped to encourage diverse solutions.  
- **Crossover & Mutation** – Neural networks inherit traits from top performers, allowing for gradual improvement.  
- **Fitness-Based Selection** – The most successful networks survive and evolve into better models over generations.  

## How NEAT is Applied in Pong  

### **Understanding the Inputs and Outputs**  
In the Pong game, we control a **single paddle** (left or right). The AI must decide:  
1. **Move Up**  
2. **Move Down**  
3. **Stay Still**  

To make these decisions, the AI needs information about the game state. The **input layer** consists of:  
- **Paddle’s Y-position** (since it moves vertically, X remains constant)  
- **Ball’s Y-position**  
- **Ball’s X-position** (relative to the paddle)  

These **three inputs** are passed through the neural network, which processes them via **hidden layers**. The network's **output layer** then determines which action the paddle should take.

### **Neural Network Structure**  
- **Input Layer (3 nodes)**  
  - **Paddle’s Y-position**  
  - **Ball’s Y-position**  
  - **Ball’s X-position**  

- **Hidden Layers**  
  - The NEAT algorithm optimizes these dynamically, adding nodes and connections as needed.

- **Output Layer (3 nodes)**  
  - **Move Up**  
  - **Move Down**  
  - **Stay Still**  

### **How Training Works**  
1. **Random Initialization** – The first generation starts with simple networks that make random decisions.  
2. **Evaluation (Fitness Function)** – Each network is tested on its ability to hit the ball and prevent it from passing.  
3. **Selection & Evolution** – The best-performing networks are kept, while weaker ones are replaced through **crossover** and **mutation**.  
4. **Progressive Learning** – Over many generations, the AI improves its gameplay strategy, leading to an optimal solution.  

## Applications of NEAT  
Beyond Pong, NEAT is widely used for:  
- **Game AI** – Teaching agents to play Flappy Bird, Mario, and other games.  
- **Robotics** – Evolving control strategies for autonomous systems.  
- **Optimization Problems** – Solving complex decision-making tasks without explicit programming.  

For more details, check out the original **NEAT paper** by Kenneth Stanley:  
📄 [NEAT: NeuroEvolution of Augmenting Topologies](http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf)  
