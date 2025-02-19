from collections import defaultdict
import gymnasium as gym
import numpy as np
import random

class Agent:
    def __init__(
        self,
        env: gym.Env,
        learning_method: str,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float,
    ):
        self.env = env
        self.learning_method = learning_method
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))
        self.lr = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.training_error = []

    def select_action(self, obs: tuple[int, int, bool]) -> int:
        if self.learning_method in ["q-learning", "sarsa"]:
            return self.select_action_epsilon_greedy(obs)
        else:
            raise ValueError(f"Learning method {self.learning_method} not supported")
        
    def update(self, obs: tuple[int, int, bool], action: int, reward: float, terminated: bool, next_obs: tuple[int, int, bool]):
        if self.learning_method == "q-learning":
            self.update_q_table(obs, action, reward, terminated, next_obs)
        elif self.learning_method == "sarsa":
            next_action = self.select_action_epsilon_greedy(next_obs)
            self.update_sarsa(obs, action, reward, terminated, next_obs, next_action)
        else:
            raise ValueError(f"Learning method {self.learning_method} not supported")

    def select_action_epsilon_greedy(self, obs: tuple[int, int, bool]) -> int:
        rv = random.uniform(0, 1)
        if rv < self.epsilon:
            return self.env.action_space.sample()
        else: 
            return int(np.argmax(self.q_values[obs]))

    def update_q_table(
        self,
        obs: tuple[int, int, bool],
        action: int,
        reward: float,
        terminated: bool,
        next_obs: tuple[int, int, bool],
    ):
        future_q_value = (not terminated) * np.max(self.q_values[next_obs])
        temporal_difference = (reward + self.discount_factor * future_q_value - self.q_values[obs][action])
        self.q_values[obs][action] = (self.q_values[obs][action] + self.lr * temporal_difference)
        self.training_error.append(temporal_difference)

    def update_sarsa(
        self,
        obs: tuple[int, int, bool],
        action: int,
        reward: float,
        terminated: bool,
        next_obs: tuple[int, int, bool],
        next_action: int,
    ):
        future_q_value = (not terminated) * self.q_values[next_obs][next_action]
        temporal_difference = (reward + self.discount_factor * future_q_value - self.q_values[obs][action])
        self.q_values[obs][action] = (self.q_values[obs][action] + self.lr * temporal_difference)
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

if __name__ == "__main__":
    print("This is the agent module, to run the simulation, use simulation.py")
