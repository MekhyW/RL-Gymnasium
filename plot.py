import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np

dfs = [pd.read_csv(f"logging/{file}") for file in os.listdir("logging") if file.endswith(".csv")]
dfs_names = [file.split(".")[0] for file in os.listdir("logging") if file.endswith(".csv")]

# First plot - Rewards (Training episodes only)
plt.figure(figsize=(12, 8))
for df, name in zip(dfs, dfs_names):
    training_df = df[df['is_training'] == True]
    rolling_reward = training_df["reward"].rolling(window=100).mean()
    plt.plot(training_df["episode"], rolling_reward, label=name)
plt.legend()
plt.title("Running Average of Training Reward per Episode (window=100)")
plt.xlabel("Episode")
plt.ylabel("Accumulated Reward")
plt.savefig("plots/reward_plot.png", dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# Second plot - Steps (Training episodes only)
plt.figure(figsize=(12, 8))
for df, name in zip(dfs, dfs_names):
    training_df = df[df['is_training'] == True]
    rolling_steps = training_df["steps"].rolling(window=100).mean()
    plt.plot(training_df["episode"], rolling_steps, label=name)
plt.legend()
plt.title("Running Average of Training Steps per Episode (window=100)")
plt.xlabel("Episode")
plt.ylabel("Number of Steps")
plt.savefig("plots/steps_plot.png", dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# Statistics for test episodes
for df, name in zip(dfs, dfs_names):
    test_df = df[df['is_training'] == False]
    print(f"\nTest Statistics for {name}:")
    print(f"Average actions taken: {test_df['steps'].mean():.2f}")
    print(f"Standard deviation actions taken: {test_df['steps'].std():.2f}")
    print(f"Average rewards: {test_df['reward'].mean():.2f}")
    print(f"Standard deviation rewards: {test_df['reward'].std():.2f}")
    print(f"Episodes with positive reward result: {test_df[test_df['reward'] > 0]['episode'].nunique()}/{len(test_df)}")