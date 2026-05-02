import gymnasium as gym
import LaneKeeping
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

env = gym.make("LaneKeeping/steerRL-v0")
env = Monitor(env)

# safe training process to tensorboard log for better viewing
model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log="./ppo_carlalog")
model.learn(total_timesteps=100000)
model.save("lane_keeping_ppo4")

#lane_keeping_ppo basic (PP0_1)
#lane_keeping_ppo4 best model (PPO_4)