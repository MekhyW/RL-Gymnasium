import gymnasium as gym
env_name = "LunarLander-v3"
env = gym.make(env_name, render_mode="human")
MAX_EPISODES = 1000

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

observation, info = env.reset(seed=42)
for _ in range(MAX_EPISODES):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    log_experience(observation, action, reward, observation, terminated)
    if terminated or truncated:
        observation, info = env.reset()

env.close()