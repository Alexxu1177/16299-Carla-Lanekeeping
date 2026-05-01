import numpy as np
from LaneKeeping.steer_RL import LaneEnv

env = LaneEnv()

obs, _ = env.reset()

MODE = "controller"  
# options: "straight", "random", "controller"

for step in range(500):

    if MODE == "straight":
        action = np.array([0.0])

    elif MODE == "random":
        action = env.action_space.sample()

    elif MODE == "controller":
        k_lat = -1.5
        k_steer = -0.8
        action_val = (
            k_lat * obs["lateral_error"][0] + k_steer * obs["steer_error"][0]
        )
        action = np.clip([action_val], -1.0, 1.0)

    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, _ = env.reset()