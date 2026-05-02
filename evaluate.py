import gymnasium as gym
import LaneKeeping
import numpy as np
from stable_baselines3 import PPO
import time

env = gym.make("LaneKeeping/steerRL-v0")
# input in the model you want to evaluate
model = PPO.load("lane_keeping_ppo3", env=env)

episodes_to_test = 10
print(f"Runing CARLA Metric Evaluation over {episodes_to_test} episodes....")

all_max_lat_errors = []
all_mean_lat_errors = []
all_survival_steps = []
success_count = 0

# run episode testing
for ep in range(episodes_to_test):
    obs, info = env.reset()
    done = False

    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        # uncomment time for visual
        # time.sleep(0.05)
        done = terminated or truncated
        if done:
            all_max_lat_errors.append(info["max_lateral_error_meters"])
            all_mean_lat_errors.append(info["mean_lateral_error_meters"])
            all_survival_steps.append(info["steps_survived"])
            if info["reached_destination"]:
                success_count+=1
            print(f"Episode {ep+1} Finished | Survived: {info['steps_survived']} steps | Max Error: {info['max_lateral_error_meters']:.2f}")

print("\n" + "="*40)
print("Final Driving Metric")
print("="*40) 
print(f"Success Rate (Reached End): {(success_count/episodes_to_test)*100:.1f}%")
print(f"Avg Survival Time         : {np.mean(all_survival_steps):.1f} steps")
print(f"Avg Mean Lateral Time     : {np.mean(all_mean_lat_errors):.3f} meters")
print(f"Avg Max Lateral Time      : {np.mean(all_max_lat_errors):.3f} meters")
print("="*40) 

env.close()