import matplotlib.pyplot as plt
import pandas as pd
import os

dfs = [pd.read_csv(f"logging/{file}") for file in os.listdir("logging") if file.endswith(".csv")]
dfs_names = [file.split(".")[0] for file in os.listdir("logging") if file.endswith(".csv")]

# First plot - Rewards
plt.figure(figsize=(12, 8))
for df, name in zip(dfs, dfs_names):
    rolling_reward = df["reward"].rolling(window=100).mean()
    plt.plot(df["episode"], rolling_reward, label=name)
plt.legend()
plt.title("Running Average of Reward per Episode (window=100)")
plt.xlabel("Episode")
plt.ylabel("Accumulated Reward")
plt.savefig("plots/reward_plot.png", dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# Second plot - Steps
plt.figure(figsize=(12, 8))
for df, name in zip(dfs, dfs_names):
    rolling_steps = df["steps"].rolling(window=100).mean()
    plt.plot(df["episode"], rolling_steps, label=name)
plt.legend()
plt.title("Running Average of Steps per Episode (window=100)")
plt.xlabel("Episode")
plt.ylabel("Number of Steps")
plt.savefig("plots/steps_plot.png", dpi=300, bbox_inches='tight')
plt.show()
plt.close()
