from gymnasium.envs.registration import register

register(
    id = "LaneKeeping/steerRL-v0",
    entry_point = "LaneKeeping.steer_RL:laneEnv"
)