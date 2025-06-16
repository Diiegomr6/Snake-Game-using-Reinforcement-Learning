# Snake Game with Q-Learning

This project implements the classic Snake game using **PyGame**, and teaches an AI agent to play it using the **Q-learning algorithm**.

Created for Machine Learning Classes at *University Carlos III of Madrid*, by José Luis Perán.

## Overview

An AI agent plays Snake by learning through trial and error. The snake is rewarded for moving toward food and penalized for collisions or inefficient movements.

- **Framework**: PyGame
- **Algorithm**: Q-learning
- **State Encoding**: Binary representation of food direction and possible safe moves

---

## How It Works

- **Environment**: `snake_env.py`  
  A custom-built simulation of the Snake game, returning states, rewards, and terminal signals.

- **Q-Learning Agent**: `q_learning.py`  
  Maintains and updates a Q-table to learn the best action for each state.

- **Training Loop**: `SnakeGame.py`  
  Connects the environment and Q-learning agent. Trains the model over multiple episodes.


## Files

- `SnakeGame.py`: Main training and game loop.
- `snake_env.py`: Snake game environment (observation, action, reward logic).
- `q_learning.py`: Q-learning implementation (epsilon-greedy, Q-table updates).
- `q_table.txt`: Saved Q-table (generated after training).


## Features

- Q-learning with ε-greedy exploration
- Reward structure:
  - +10 for eating food
  - -100 for dying
  - +1 for moving closer to food, -1 for moving away
- Smart binary state encoding based on food and danger directions


## Training Parameters

- **Episodes**: 2000 (configurable)
- **States**: 256 (2⁸ binary features)
- **Actions**: 4 (Up, Down, Left, Right)
- **Learning rate** (α): 0.1
- **Discount factor** (γ): 0.9
- **Exploration rate** (ε): starts at 0.9, decays over time


## Notes

- The snake's state is represented by 8 binary flags (food direction and move safety).
- The environment prevents the snake from reversing direction immediately.
- Set `growing_body = False` during testing for more stable evaluation.


