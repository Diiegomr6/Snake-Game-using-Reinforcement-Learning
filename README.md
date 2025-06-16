# Snake Game with Q-Learning

This project implements the classic Snake game using the PyGame library and trains an AI agent using **Q-learning**, a reinforcement learning algorithm. The agent learns how to play by interacting with the environment, receiving rewards or penalties, and updating its knowledge accordingly. The final goal is for the agent to maximize its score by eating food and avoiding collisions.


## How It Works

The project consists of three main files:

- `snake_env.py`: defines the **Snake game environment** and the game logic.
- `q_learning.py`: implements the **Q-learning algorithm**, including the Q-table and update rules.
- `SnakeGame.py`: the **main script** that runs the training or testing loop.


## State Representation

The snake's state is represented as an **8-bit binary vector**. These binary variables encode two types of information:

1. **Food Direction**:
   - `food_north`: `1` if food is above the snake, else `0`
   - `food_south`: `1` if food is below
   - `food_west`: `1` if food is to the left
   - `food_east`: `1` if food is to the right

2. **Safe Moves**:
   - `valid_step_west`: `1` if the snake can move left without dying or getting trapped
   - `valid_step_east`: `1` if it can move right safely
   - `valid_step_north`: `1` if it can move up safely
   - `valid_step_south`: `1` if it can move down safely

These 8 binary values are concatenated and converted into a single **integer state** using binary encoding. This results in `2^8 = 256` unique states.


## Action Space

There are **4 possible actions** the agent can take at each step:

- `0`: UP
- `1`: DOWN
- `2`: LEFT
- `3`: RIGHT

The environment ensures that the snake doesn't reverse direction (e.g., if it’s going RIGHT, it can't suddenly go LEFT).


## Reward System

The agent receives rewards based on its behavior:

- `+10`: eating food
- `-100`: dying (colliding with wall or self)
- `+1`: moving closer to the food (based on Manhattan distance)
- `-1`: moving farther from the food

This encourages the snake to seek food while avoiding dangerous or inefficient paths.


## Q-Learning Parameters

In `q_learning.py`, the Q-learning algorithm maintains a Q-table with dimensions `[256 x 4]` (states x actions).

Key parameters include:

- `alpha (α = 0.1)`: learning rate — how much to update the Q-value
- `gamma (γ = 0.9)`: discount factor — future reward importance
- `epsilon (ε = 0.1 initially)`: exploration rate — how often the agent chooses a random action
- `epsilon_decay (0.9995)`: how quickly `ε` decreases over time
- `epsilon_min (0.01)`: minimum exploration rate

The agent uses an **ε-greedy strategy**, meaning it will mostly choose the best-known action but sometimes explore.


## Training Logic

The main training happens in `SnakeGame.py`.

1. **Initialize** the environment and Q-learning agent.
2. For each episode:
   - Reset the environment.
   - While the game is not over:
     - Use the agent’s policy (`choose_action`) to pick an action.
     - Execute the action in the environment (`env.step()`).
     - Get the new state, reward, and terminal flag.
     - Update the Q-table with `update_q_table()`.
     - Render the game (if `render_game = True`).
3. After each episode, save the Q-table (`q_table.txt`) and print the total reward.
4. At the end, print the average reward over all episodes.

```python
training = True         # Toggle learning on or off
render_game = True      # Show visual game window
growing_body = True     # If True, the snake grows when eating
num_episodes = 2000     # Total training episodes
```

During **training**, the snake learns by trial and error. During **testing**, you can turn off training (`training = False`) to evaluate its performance using the learned Q-table.

---

## Persistence

The Q-table is saved to a file named `q_table.txt` after every episode using NumPy. If the file exists, it’s loaded on startup so training can continue from previous runs.

```python
def save_q_table(self, filename="q_table.txt"):
    np.savetxt(filename, self.q_table)

def load_q_table(self, filename="q_table.txt"):
    try:
        self.q_table = np.loadtxt(filename)
    except IOError:
        self.q_table = np.zeros((self.n_states, self.n_actions))
```
