import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from gymnasium.wrappers import RecordVideo
from agent import Agent
import numpy as np

EXPERIMENT_NAME = "LunarLander-deep-q-learning"
ENV_NAME = "LunarLander-v3"
ENV_PARAMS = {'render_mode': "rgb_array"}
LEARNING_METHOD = "deep-q-learning"
TRAINING_EPISODES = 5000
TESTING_EPISODES = 100
MAX_STEPS = 1000
LEARNING_RATE = 0.001
INITIAL_EPSILON = 1.0
EPSILON_DECAY = 0.999
FINAL_EPSILON = 0.01
DISCOUNT_FACTOR = 0.99
MEMORY_LENGTH = 100000
BATCH_SIZE = 128
STEPS_PER_UPDATE = 50
env = gym.make(ENV_NAME, **ENV_PARAMS).env
env = TimeLimit(env, max_episode_steps=MAX_STEPS)
env = RecordVideo(env, video_folder="videos", episode_trigger=lambda t: t % int(TRAINING_EPISODES/10) == 0)
agent = Agent(env, LEARNING_METHOD, LEARNING_RATE, INITIAL_EPSILON, EPSILON_DECAY, FINAL_EPSILON, DISCOUNT_FACTOR, MEMORY_LENGTH, BATCH_SIZE, STEPS_PER_UPDATE)
np.random.seed(0)

def reset_log_files():
    if LEARNING_METHOD == "deep-q-learning":
        with open(f"logging/{EXPERIMENT_NAME}-model_weights.txt", "w") as f:
            f.write("")
    else:
        with open(f"logging/{EXPERIMENT_NAME}-q_table.txt", "w") as f:
            f.write("")
    with open(f"logging/{EXPERIMENT_NAME}-episodes.csv", "w") as f:
        f.write("episode,reward,steps,is_training\n")

def log_q_table_weights():
    if LEARNING_METHOD == "deep-q-learning":
        with open(f"logging/{EXPERIMENT_NAME}-model_weights.txt", "a") as f:
            model_params = {}
            for name, param in agent.model.named_parameters():
                model_params[name] = param.data.detach().numpy()
            f.write(str(model_params).replace("\n", ""))
            f.write("\n")
    else:
        with open(f"logging/{EXPERIMENT_NAME}-q_table.txt", "a") as f:
            f.write(str(agent.q_values))
            f.write("\n")

def log_episodes(episode, accumulated_reward, length, is_training):
    with open(f"logging/{EXPERIMENT_NAME}-episodes.csv", "a") as f:
        f.write(f"{episode},{accumulated_reward},{length},{is_training}\n")

def transform_state(state, env, num_states):
    if not isinstance(env.observation_space, gym.spaces.Box):
        return state
    obs_dims = env.observation_space.shape[0]
    scaling_factors = np.ones(obs_dims) * 10
    state_adj = (state - env.observation_space.low) * scaling_factors
    state_adj = np.round(state_adj, 0).astype(int)
    upper_bounds = num_states - 1
    state_adj = np.clip(state_adj, 0, upper_bounds)
    return state_adj

def run_episodes(env, agent, num_episodes, is_training, start_episode):
    if not is_training:
        agent.epsilon = 0
    for i in range(num_episodes):
        print(f"{"Training" if is_training else "Testing"} Episode {i}")
        state, _ = env.reset()
        state_adj = transform_state(state, env, agent.num_states)
        done = False
        accumulated_reward = 0
        length = 0
        while not done:
            action = agent.select_action(state_adj)
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state_adj = transform_state(next_state, env, agent.num_states)
            done = terminated or truncated
            if is_training:
                agent.update(state_adj, action, reward, done, next_state_adj)
            env.render()
            state_adj = next_state_adj
            accumulated_reward += reward
            length += 1
        log_episodes(start_episode + i, accumulated_reward, length, is_training)
        if is_training:
            agent.decay_epsilon(i)
            log_q_table_weights()

if __name__ == "__main__":
    reset_log_files()
    print("Training...")
    run_episodes(env, agent, TRAINING_EPISODES, is_training=True, start_episode=0)
    print("Testing...")
    run_episodes(env, agent, TESTING_EPISODES, is_training=False, start_episode=TRAINING_EPISODES)
    print("Done!")
    env.close()
