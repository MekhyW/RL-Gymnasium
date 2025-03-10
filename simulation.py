import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from gymnasium.wrappers import RecordVideo
from agent import Agent
import numpy as np

EXPERIMENT_NAME = "FrozenLake-qlearning-01-1-0995-03-099"
ENV_NAME = "FrozenLake-v1"
ENV_PARAMS = {'render_mode': "rgb_array", "map_name": "4x4", "is_slippery": True}
LEARNING_METHOD = "q-learning"
TRAINING_EPISODES = 2000
TESTING_EPISODES = 100
MAX_STEPS = 100
LEARNING_RATE = 0.1
INITIAL_EPSILON = 1.0
EPSILON_DECAY = 0.995
FINAL_EPSILON = 0.3
DISCOUNT_FACTOR = 0.99
env = gym.make(ENV_NAME, **ENV_PARAMS).env
env = TimeLimit(env, max_episode_steps=MAX_STEPS)
env = RecordVideo(env, video_folder="videos", episode_trigger=lambda t: t % int(TRAINING_EPISODES/10) == 0)
agent = Agent(env, LEARNING_METHOD, LEARNING_RATE, INITIAL_EPSILON, EPSILON_DECAY, FINAL_EPSILON, DISCOUNT_FACTOR)

def reset_log_files():
    with open(f"logging/{EXPERIMENT_NAME}-q_table.txt", "w") as f:
        f.write("")
    with open(f"logging/{EXPERIMENT_NAME}-episodes.csv", "w") as f:
        f.write("episode,reward,steps,is_training\n")

def log_q_table():
    with open(f"logging/{EXPERIMENT_NAME}-q_table.txt", "a") as f:
        f.write(str(agent.q_values))
        f.write("\n")

def log_episodes(episode, accumulated_reward, length, is_training):
    with open(f"logging/{EXPERIMENT_NAME}-episodes.csv", "a") as f:
        f.write(f"{episode},{accumulated_reward},{length},{is_training}\n")

def transform_state(state):
    if isinstance(env.observation_space, gym.spaces.Box):
        state_adj = (state - env.observation_space.low)*np.array([10, 100])
        return np.round(state_adj, 0).astype(int)
    else:
        return state

def run_episodes(env, agent, num_episodes, is_training, start_episode):
    if not is_training:
        agent.epsilon = 0
    for i in range(num_episodes):
        print(f"{"Training" if is_training else "Testing"} Episode {i}")
        state, _ = env.reset()
        state_adj = transform_state(state)
        done = False
        accumulated_reward = 0
        length = 0
        while not done:
            action = agent.select_action(state_adj)
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state_adj = transform_state(next_state)
            done = terminated or truncated
            if is_training:
                agent.update(state_adj, action, reward, done, next_state_adj)
            env.render()
            state_adj = next_state_adj
            accumulated_reward += reward
            length += 1
        log_episodes(start_episode + i, accumulated_reward, length, is_training)
        if is_training:
            agent.decay_epsilon()
            log_q_table()

if __name__ == "__main__":
    reset_log_files()
    print("Training...")
    run_episodes(env, agent, TRAINING_EPISODES, is_training=True, start_episode=0)
    print("Testing...")
    run_episodes(env, agent, TESTING_EPISODES, is_training=False, start_episode=TRAINING_EPISODES)
    print("Done!")
    env.close()
