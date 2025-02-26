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
        if isinstance(env.observation_space, gym.spaces.Box):
            self.num_states = (env.observation_space.high - env.observation_space.low)*np.array([10, 100])
            self.num_states = np.round(self.num_states, 0).astype(int) + 1
            self.q_values = np.zeros([self.num_states[0], self.num_states[1], env.action_space.n])
        else:
            self.num_states = env.observation_space.n
            self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))
        self.learning_method = learning_method
        self.lr = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.training_error = []

    def select_action(self, state_adj: tuple[int, int, bool]) -> int:
        if self.learning_method in ["q-learning", "sarsa"]:
            return self.select_action_epsilon_greedy(state_adj)
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

    def select_action_epsilon_greedy(self, state_adj: tuple[int, int, bool]) -> int:
        if isinstance(self.env.observation_space, gym.spaces.Box):
            state_idx = (state_adj[0], state_adj[1])
        else:
            state_idx = state_adj
        if random.uniform(0, 1) < 1 - self.epsilon:
            return np.argmax(self.q_values[state_idx])
        return np.random.randint(0, self.env.action_space.n)

    def update_q_table(
        self,
        obs: tuple[int, int, bool],
        action: int,
        reward: float,
        terminated: bool,
        next_obs: tuple[int, int, bool],
    ):
        if isinstance(self.env.observation_space, gym.spaces.Box):
            obs_idx = (obs[0], obs[1])
            next_obs_idx = (next_obs[0], next_obs[1])
        else:
            obs_idx = obs
            next_obs_idx = next_obs
        future_q_value = (not terminated) * np.max(self.q_values[next_obs_idx])
        temporal_difference = (reward + self.discount_factor * future_q_value - self.q_values[obs_idx][action])
        self.q_values[obs_idx][action] = (self.q_values[obs_idx][action] + self.lr * temporal_difference)
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
        if isinstance(self.env.observation_space, gym.spaces.Box):
            obs_idx = (obs[0], obs[1])
            next_obs_idx = (next_obs[0], next_obs[1])
        else:
            obs_idx = obs
            next_obs_idx = next_obs
            
        future_q_value = (not terminated) * self.q_values[next_obs_idx][next_action]
        temporal_difference = (reward + self.discount_factor * future_q_value - self.q_values[obs_idx][action])
        self.q_values[obs_idx][action] = (self.q_values[obs_idx][action] + self.lr * temporal_difference)
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

if __name__ == "__main__":
    print("This is the agent module, to run the simulation, use simulation.py")
