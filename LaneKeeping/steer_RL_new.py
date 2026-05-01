import carla
import random
import math
import gymnasium as gym
from gymnasium import spaces
import numpy as np


def if_passed(car, waypoint):
    v = car.get_transform().location - waypoint.transform.location
    v_xy = carla.Vector3D(v.x, v.y, 0)
    vw = waypoint.transform.get_forward_vector()
    vw_xy = carla.Vector3D(vw.x, vw.y, 0)
    angle = math.degrees(v_xy.get_vector_angle(vw_xy))
    return angle <= 90


class LaneEnv(gym.Env):
    def __init__(self):
        super().__init__()

        client = carla.Client('localhost', 2000)
        client.set_timeout(20)

        client.load_world('Town01')
        self._world = client.get_world()

        settings = self._world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        self._world.apply_settings(settings)

        self._spectator = self._world.get_spectator()

        self._ego_blueprint = self._world.get_blueprint_library().find('vehicle.lincoln.mkz_2020')
        self._spawn_points = self._world.get_map().get_spawn_points()

        self.observation_space = spaces.Dict({
            "lateral_error": spaces.Box(-10.0, 10.0, (1,), np.float32),
            "steer_error": spaces.Box(-1.0, 1.0, (1,), np.float32)
        })

        self.action_space = spaces.Box(-1.0, 1.0, (1,), np.float32)

        self._ego_car = None
        self._waypoint_list = None
        self._cur_idx = 0
        self._reach_end = False

        self._step_count = 0

        # Episode stats
        self._episode_lateral_error = 0.0
        self._episode_steps = 0
        self._max_lateral_error = 0.0

    def safe_angle(self, v1, v2):
        a = np.array([v1.x, v1.y])
        b = np.array([v2.x, v2.y])

        if np.linalg.norm(a) < 1e-6 or np.linalg.norm(b) < 1e-6:
            return 0.0

        cos_theta = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        return math.acos(cos_theta)

    def _get_obs(self):
        wp = self._waypoint_list[self._cur_idx]

        if if_passed(self._ego_car, wp) and self._cur_idx + 1 < len(self._waypoint_list):
            self._cur_idx += 1
            wp = self._waypoint_list[self._cur_idx]
        else:
            if self._cur_idx + 1 >= len(self._waypoint_list):
                self._reach_end = True

        car_loc = self._ego_car.get_transform().location
        wp_loc = wp.transform.location

        car_xy = carla.Vector3D(car_loc.x, car_loc.y, 0)
        wp_xy = carla.Vector3D(wp_loc.x, wp_loc.y, 0)

        wp_forward = wp.transform.get_forward_vector()
        wp_forward_xy = carla.Vector3D(wp_forward.x, wp_forward.y, 0)

        v_xy = car_xy - wp_xy
        dist = np.linalg.norm([v_xy.x, v_xy.y])

        cross = v_xy.cross(wp_forward_xy)
        direction = -1 if cross.z < 0 else 1

        theta = self.safe_angle(v_xy, wp_forward_xy)
        lateral_error = direction * dist * math.sin(theta)

        car_forward = self._ego_car.get_transform().get_forward_vector()
        car_forward_xy = carla.Vector3D(car_forward.x, car_forward.y, 0)

        steer_angle = self.safe_angle(car_forward_xy, wp_forward_xy)
        cross2 = car_forward_xy.cross(wp_forward_xy)
        direction2 = -1 if cross2.z < 0 else 1

        steer_error = direction2 * math.degrees(steer_angle) / 180.0

        lateral_error = np.clip(lateral_error, -10.0, 10.0)
        steer_error = np.clip(steer_error, -1.0, 1.0)

        return {
            "lateral_error": np.array([lateral_error], dtype=np.float32),
            "steer_error": np.array([steer_error], dtype=np.float32)
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self._ego_car:
            self._ego_car.destroy()

        self._ego_car = self._world.spawn_actor(
            self._ego_blueprint,
            random.choice(self._spawn_points)
        )

        self._world.tick()

        wp = self._world.get_map().get_waypoint(self._ego_car.get_location())
        self._waypoint_list = wp.next_until_lane_end(4)
        self._cur_idx = 0
        self._reach_end = False
        self._step_count = 0

        # Reset stats
        self._episode_lateral_error = 0.0
        self._episode_steps = 0
        self._max_lateral_error = 0.0

        return self._get_obs(), {}

    def step(self, action):
        self._ego_car.apply_control(
            carla.VehicleControl(throttle=0.5, steer=float(action[0]))
        )

        self._world.tick()

        obs = self._get_obs()

        velocity = self._ego_car.get_velocity()
        speed = np.linalg.norm([velocity.x, velocity.y])

        lat = abs(obs["lateral_error"][0])
        steer = abs(obs["steer_error"][0])

        # Reward breakdown
        lat_pen = 0.15 * lat
        steer_pen = 0.3 * steer
        
        # Don't reward when car is not moving
        if speed < 0.1:
            reward = 0.0
        else:
            reward = 1.0 - lat_pen - steer_pen
        
        # Track stats
        self._episode_lateral_error += lat
        self._max_lateral_error = max(self._max_lateral_error, lat)
        self._episode_steps += 1

        # Debug print
        print(
            f"step={self._step_count} | "
            f"lat={lat:.3f} (pen={lat_pen:.3f}) | "
            f"steer={steer:.3f} (pen={steer_pen:.3f}) | "
            f"reward={reward:.3f} | speed={speed:.2f}"
        )

        self._step_count += 1

        wp = self._waypoint_list[self._cur_idx]
        width = wp.lane_width

        terminated = lat > width / 2.0 or (self._step_count > 20 and speed < 0.1)
        truncated = self._reach_end

        info = {}

        if terminated or truncated:
            avg_lat = self._episode_lateral_error / max(1, self._episode_steps)
            score = max(0, 100 * (1 - avg_lat / (width / 2.0)))

            print(
                f"\nEPISODE SUMMARY:\n"
                f"steps={self._episode_steps} | "
                f"avg_lat={avg_lat:.3f} | "
                f"max_lat={self._max_lateral_error:.3f} | "
                f"score={score:.1f}\n"
            )

            info["episode"] = {
                "avg_lat": avg_lat,
                "max_lat": self._max_lateral_error,
                "steps": self._episode_steps,
                "score": score
            }

        return obs, reward, terminated, truncated, info