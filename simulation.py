import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from gymnasium.wrappers import RecordVideo
from agent import Agent

EXPERIMENT_NAME = "FrozenLake-sarsa"
ENV_NAME = "FrozenLake-v1"
ENV_PARAMS = {'map_name': '4x4', 'is_slippery': True, 'render_mode': "rgb_array"}
LEARNING_METHOD = "sarsa"
TRAINING_EPISODES = 5000
TESTING_EPISODES = 100
MAX_STEPS = 200
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

def run_episodes(env, agent, num_episodes, is_training, start_episode):
    if not is_training:
        agent.epsilon = 0
    for i in range(num_episodes):
        print(f"{"Training" if is_training else "Testing"} Episode {i}")
        state, info = env.reset()
        done = False
        accumulated_reward = 0
        length = 0
        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            if is_training:
                agent.update(state, action, reward, done, next_state)
            env.render()
            state = next_state
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
