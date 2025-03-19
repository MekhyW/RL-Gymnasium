from collections import defaultdict
import gymnasium as gym
import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim

class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

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
        steps_per_update: int,
    ):
        self.env = env
        self.learning_method = learning_method
        if isinstance(env.observation_space, gym.spaces.Box):
            obs_dims = env.observation_space.shape[0]
            scaling_factors = np.ones(obs_dims) * 10
            state_ranges = np.abs(env.observation_space.high - env.observation_space.low)
            state_ranges[np.isinf(state_ranges)] = 100
            self.num_states = state_ranges * scaling_factors
            self.num_states = np.round(self.num_states, 0).astype(int) + 1
            self.num_states = np.maximum(self.num_states, 1)
            print("NUMBER OF STATES: ", self.num_states)
            self.q_values = np.zeros([*self.num_states, env.action_space.n]) if self.learning_method in ["q-learning", "sarsa"] else None
        else:
            self.num_states = env.observation_space.n
            self.q_values = defaultdict(lambda: np.zeros(env.action_space.n)) if self.learning_method in ["q-learning", "sarsa"] else None
        self.lr = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.training_error = []
        self.steps_per_update = steps_per_update
        if self.learning_method == "deep-q-learning":
            self.model = QNetwork(env.observation_space.shape[0], env.action_space.n)
            self.memory = deque(maxlen=memory_length)
            self.batch_size = batch_size
            self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
            self.loss_fn = nn.MSELoss()
            self.target_model = QNetwork(env.observation_space.shape[0], env.action_space.n)
            self.target_model.load_state_dict(self.model.state_dict())
            self.target_model.eval()

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
    
    def select_action_deep_q_learning(self, state: tuple[int, int, bool]) -> int:
        if np.random.rand() < self.epsilon:
            return random.randrange(self.env.action_space.n)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        self.model.eval()
        with torch.no_grad():
            q_values = self.model(state_tensor)
        action = torch.argmax(q_values, dim=1).item()
        return action

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

    def experience(self, state, action, reward, next_state, terminal):
        self.memory.append((state, action, reward, next_state, terminal))

    def experience_replay(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states = np.array([exp[0] for exp in batch])
        actions = np.array([exp[1] for exp in batch])
        rewards = np.array([exp[2] for exp in batch])
        next_states = np.array([exp[3] for exp in batch])
        terminals = np.array([exp[4] for exp in batch]).astype(np.uint8)
        states_tensor = torch.FloatTensor(states).squeeze(1)         # shape: (batch_size, input_dim)
        next_states_tensor = torch.FloatTensor(next_states).squeeze(1)   # shape: (batch_size, input_dim)
        actions_tensor = torch.LongTensor(actions).unsqueeze(1)          # shape: (batch_size, 1)
        rewards_tensor = torch.FloatTensor(rewards).unsqueeze(1)
        terminals_tensor = torch.FloatTensor(terminals).unsqueeze(1)
        self.model.train()
        q_values = self.model(states_tensor)
        with torch.no_grad():
            next_actions = torch.argmax(self.model(next_states_tensor), dim=1, keepdim=True)
            next_q_values_target = self.target_model(next_states_tensor)
            next_q_values = torch.gather(next_q_values_target, 1, next_actions)
        targets = rewards_tensor + self.discount_factor * next_q_values * (1 - terminals_tensor)
        q_selected = torch.gather(q_values, 1, actions_tensor)
        loss = self.loss_fn(q_selected, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_deep_q_learning(self, state, action, reward, next_state, terminal):
        self.experience(state, action, reward, next_state, terminal)
        state = next_state
        self.experience_replay()

    def decay_epsilon(self, i):
        self.epsilon = max(self.final_epsilon, self.epsilon * self.epsilon_decay)
        if i % self.steps_per_update == 0:
            self.target_model.load_state_dict(self.model.state_dict())

if __name__ == "__main__":
    print("This is the agent module, to run the simulation, use simulation.py")
