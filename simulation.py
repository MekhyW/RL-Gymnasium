import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from gymnasium.wrappers import RecordVideo
from agent import Agent

EXPERIMENT_NAME = "taxi-low-alpha-high-gamma"
ENV_NAME = "Taxi-v3"
MAX_EPISODES = 1000
MAX_STEPS = 200
LEARNING_RATE = 0.1
INITIAL_EPSILON = 1.0
EPSILON_DECAY = INITIAL_EPSILON / (MAX_EPISODES / 2)
FINAL_EPSILON = 0.01
DISCOUNT_FACTOR = 0.95
env = gym.make(ENV_NAME, render_mode="rgb_array").env
env = TimeLimit(env, max_episode_steps=MAX_STEPS)
env = RecordVideo(env, video_folder="videos", episode_trigger=lambda t: t % int(MAX_EPISODES/10) == 0)
agent = Agent(env, LEARNING_RATE, INITIAL_EPSILON, EPSILON_DECAY, FINAL_EPSILON, DISCOUNT_FACTOR)

def reset_log_files():
    with open(f"logging/{EXPERIMENT_NAME}-q_table.txt", "w") as f:
        f.write("")
    with open(f"logging/{EXPERIMENT_NAME}-episodes.csv", "w") as f:
        f.write("episode,reward,steps\n")

def log_q_table():
    with open(f"logging/{EXPERIMENT_NAME}-q_table.txt", "a") as f:
        f.write(str(agent.q_values))
        f.write("\n")

def log_episodes(episode, accumulated_reward, length):
    with open(f"logging/{EXPERIMENT_NAME}-episodes.csv", "a") as f:
        f.write(f"{episode},{accumulated_reward},{length}\n")

reset_log_files()
for i in range(MAX_EPISODES):
    print(f"Episode {i}")
    state, info = env.reset()
    done = False
    accumulated_reward = 0
    length = 0
    while not done:
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        agent.update(state, action, reward, done, next_state)
        env.render()
        state = next_state
        accumulated_reward += reward
        length += 1
    agent.decay_epsilon()
    log_episodes(i, accumulated_reward, length)
    log_q_table()

env.close()