# Machine Learning Project

A collection of machine learning implementations and game AI experiments.
Each project keeps its source, configuration, documentation, and local assets
together. Existing project directory names are preserved.

## Project catalog

| Area | Project | Start here |
| --- | --- | --- |
| Game AI | Dino: deep Q-learning | [Guide](Game%20AI/Dino%20Game%20Q%20Learning/README.md), [agent code](Game%20AI/Dino%20Game%20Q%20Learning/dino_deep_q_learning/) |
| Game AI | Pong: NEAT | [Guide](Game%20AI/Pong%20NEAT/README.md), [implementation notes](Game%20AI/Pong%20NEAT/DOCUMENTATION.md) |
| Game AI | Snake: deep Q-learning and A* | [Guide](Game%20AI/Snake%20Game%20Q%20Learning/snake_game_deep_q_learning/README.md), [human game](Game%20AI/Snake%20Game%20Q%20Learning/snake_game_human.py) |
| Algorithms from scratch | Convolutional neural network | [Guide](Machine%20Learning%20Algorithm%20From%20Scratch/Convolutional%20Neural%20Network/README.md) |
| Algorithms from scratch | Linear regression | [Guide](Machine%20Learning%20Algorithm%20From%20Scratch/Linear%20Regression/README.md) |
| Algorithms from scratch | Logistic regression | [Guide](Machine%20Learning%20Algorithm%20From%20Scratch/Logistic%20Regression/README.md) |

## Working with projects

Clone the repository and use the individual project guides for dependencies.

```sh
git clone https://github.com/YouKnowJP/Machine-Learning-Project.git
cd Machine-Learning-Project
```

Run a project's entry point from that project's code directory so its sibling
imports, font files, configuration, and saved model paths resolve as intended.
For example, the Dino CLI lives in
`Game AI/Dino Game Q Learning/dino_deep_q_learning/`.

Browse [game AI](Game%20AI/README.md),
[algorithms from scratch](Machine%20Learning%20Algorithm%20From%20Scratch/README.md),
or the [complete local file index](docs/FILE_INDEX.md).
