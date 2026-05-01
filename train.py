import gymnasium as gym
import LaneKeeping
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

env = gym.make("LaneKeeping/steerRL-v0")
env = Monitor(env)

# model = PPO("MultiInputPolicy", env, n_steps=128, verbose=1)
# # 1000 = 3 min, # 100000 = 5 hr
# # Use small before big train
# model.learn(total_timesteps=1280)
# model.save("lane_keeping_ppo")

model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log="./ppo_carlalog")
# 1000 = 3 min, # 100000 = 5 hr
# Use small before big train
model.learn(total_timesteps=100000)
model.save("lane_keeping_ppo4")
#lane_keeping_ppo1 basic (PP0_1.0)
#lane_keeping_ppo2 throttle = 0.4 (PPO_4)
#lane_keeping_ppo3 add smoothness penalty (PP0_5)
#lane_keeping_ppo4 fix lateral error calculation, d= 0.5 (PPO_6)