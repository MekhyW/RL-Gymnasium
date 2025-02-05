import gymnasium as gym
env_name = "Taxi-v3"
env = gym.make(env_name, render_mode="human").env
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

def select_action(state):
    return env.action_space.sample()

done = False
for i in range(MAX_EPISODES):
    state = env.reset()
    while not done:
        action = select_action(state)
        next_state, reward, done, truncated, info = env.step(action)
        log_experience(state, action, reward, next_state, done)
        env.render()

env.close()