import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from gymnasium.wrappers import RecordVideo
from agent import Agent

ENV_NAME = "Taxi-v3"
MAX_EPISODES = 1000
MAX_STEPS = 200
LEARNING_RATE = 0.001
INITIAL_EPSILON = 1.0
EPSILON_DECAY = INITIAL_EPSILON / (MAX_EPISODES / 2)
FINAL_EPSILON = 0.01
DISCOUNT_FACTOR = 0.95
env = gym.make(ENV_NAME, render_mode="rgb_array").env
env = TimeLimit(env, max_episode_steps=MAX_STEPS)
env = RecordVideo(env, video_folder="videos", episode_trigger=lambda t: t % (MAX_STEPS/2) == 0)
agent = Agent(env, LEARNING_RATE, INITIAL_EPSILON, EPSILON_DECAY, FINAL_EPSILON, DISCOUNT_FACTOR)

def log_experience(state, action, reward, next_state, done):
    experience = {
        "state": state,
        "action": action,
        "reward": reward,
        "next_state": next_state,
        "done": done
    }
    with open("experiences.txt", "a") as f:
        f.write(str(experience))
        f.write("\n")

def log_q_table():
    with open("q_table.txt", "a") as f:
        f.write(str(agent.q_values))
        f.write("\n")

done = False
for i in range(MAX_EPISODES):
    print(f"Episode {i}")
    state, info = env.reset()
    while not done:
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        agent.update(state, action, reward, done, next_state)
        log_experience(state, action, reward, next_state, done)
        log_q_table()
        env.render()
        done = terminated or truncated
        state = next_state
    agent.decay_epsilon()

env.close()