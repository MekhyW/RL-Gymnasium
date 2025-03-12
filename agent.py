from collections import defaultdict
import gymnasium as gym
import numpy as np
import random
from collections import deque
from keras import Sequential
from keras.layers import Dense
from keras.activations import relu, linear
from tensorflow.keras.optimizers import Adam

def initialize_deep_model(env, LEARNING_RATE):
    model = Sequential()
    model.add(Dense(512, activation=relu, input_dim=env.observation_space.shape[0]))
    model.add(Dense(256, activation=relu))
    model.add(Dense(env.action_space.n, activation=linear))
    model.summary()
    model.compile(loss='mse', optimizer=Adam(learning_rate=LEARNING_RATE))
    return model

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
        memory_length: int,
        batch_size: int,
    ):
        self.env = env
        if isinstance(env.observation_space, gym.spaces.Box):
            obs_dims = env.observation_space.shape[0]
            scaling_factors = np.ones(obs_dims) * 10
            state_ranges = np.abs(env.observation_space.high - env.observation_space.low)
            state_ranges[np.isinf(state_ranges)] = 100
            self.num_states = state_ranges * scaling_factors
            self.num_states = np.round(self.num_states, 0).astype(int) + 1
            self.num_states = np.maximum(self.num_states, 1)
            print("NUMBER OF STATES: ", self.num_states)
            self.q_values = np.zeros([*self.num_states, env.action_space.n])
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
        if self.learning_method == "deep-q-learning":
            self.model = initialize_deep_model(env, learning_rate)
            self.memory = deque(maxlen=memory_length)
            self.batch_size = batch_size

    def select_action(self, state_adj: tuple[int, int, bool]) -> int:
        if self.learning_method in ["q-learning", "sarsa"]:
            return self.select_action_epsilon_greedy(state_adj)
        elif self.learning_method == "deep-q-learning":
            return self.select_action_deep_q_learning(state_adj)
        else:
            raise ValueError(f"Learning method {self.learning_method} not supported")
        
    def update(self, obs: tuple[int, int, bool], action: int, reward: float, terminated: bool, next_obs: tuple[int, int, bool]):
        if self.learning_method == "q-learning":
            self.update_q_table(obs, action, reward, terminated, next_obs)
        elif self.learning_method == "sarsa":
            next_action = self.select_action_epsilon_greedy(next_obs)
            self.update_sarsa(obs, action, reward, terminated, next_obs, next_action)
        elif self.learning_method == "deep-q-learning":
            self.update_deep_q_learning(obs, action, reward, next_obs, terminated)
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
    
    def select_action_deep_q_learning(self, state_adj: tuple[int, int, bool]) -> int:
        if np.random.rand() < self.epsilon:
            return random.randrange(self.env.action_space.n)
        action = self.model.predict(state_adj, verbose=0)
        return np.argmax(action[0])

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

    def update_deep_q_learning(self, state, action, reward, next_state, terminal):
        self.memory.append((state, action, reward, next_state, terminal))
        if len(self.memory) <= self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size) 
        states = np.array([i[0] for i in batch])
        actions = np.array([i[1] for i in batch])
        rewards = np.array([i[2] for i in batch])
        next_states = np.array([i[3] for i in batch])
        terminals = np.array([i[4] for i in batch])
        states = np.squeeze(states)
        next_states = np.squeeze(next_states)
        next_max = np.amax(self.model.predict_on_batch(next_states), axis=1)
        targets = rewards + self.discount_factor * (next_max) * (1 - terminals)
        targets_full = self.model.predict_on_batch(states)
        indexes = np.array([i for i in range(self.batch_size)])
        targets_full[[indexes], [actions]] = targets
        self.model.fit(states, targets_full, epochs=1, verbose=0)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon * self.epsilon_decay)

if __name__ == "__main__":
    print("This is the agent module, to run the simulation, use simulation.py")
