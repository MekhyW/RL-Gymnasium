import gymnasium as gym
from agent import Agent

env_name = input("Enter the name of the environment [Taxi-v3, LunarLander-v3, CartPole-v1]: ")
env = gym.make(env_name, render_mode="human").env
MAX_EPISODES = 10000
LEARNING_RATE = 0.001
INITIAL_EPSILON = 1.0
EPSILON_DECAY = 0.999
FINAL_EPSILON = 0.01
DISCOUNT_FACTOR = 0.95
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
    state = env.reset()
    while not done:
        action = agent.select_action(state)
        next_state, reward, done, truncated, info = env.step(action)
        agent.update(state, action, reward, done, next_state)
        log_experience(state, action, reward, next_state, done)
        log_q_table()
        env.render()
        state = next_state
    agent.decay_epsilon()

env.close()