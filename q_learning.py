"""
Snake Eater Q learning basic algorithm
Made with PyGame
Last modification in April 2024 by José Luis Perán
Machine Learning Classes - University Carlos III of Madrid
"""
import numpy as np
import random
import json
import time

class QLearning:
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.9, epsilon=0.1, epsilon_min=0.01, epsilon_decay=0.9995):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.load_q_table()

    def choose_action(self, state, allowed_actions, training=True):
        if training and np.random.uniform(0, 1) < self.epsilon:
            action = random.choice(allowed_actions)  # Explore
        else:
            action = np.argmax(self.q_table[state])  # Exploit

        # Only decay epsilon during training
        if training:
            self.epsilon = max(self.epsilon_min, self.epsilon_decay * self.epsilon)

        return action

 
    def update_q_table(self, state, action, reward, next_state, terminal_state=False):
        current_q = self.q_table[state][action]

        if terminal_state:
            new_q = (1 - self.alpha) * current_q + self.alpha * reward
        else:
            max_future_q = np.max(self.q_table[next_state])
            new_q = (1 - self.alpha) * current_q + self.alpha * (reward + self.gamma * max_future_q)

        self.q_table[state][action] = new_q

    def save_q_table(self, filename="q_table.txt"):
        np.savetxt(filename, self.q_table)

    def load_q_table(self, filename="q_table.txt"):
        try:
            self.q_table = np.loadtxt(filename)
        except IOError:
            # If the file doesn't exist, initialize Q-table with zeros as per dimensions
            self.q_table = np.zeros((self.n_states, self.n_actions))
