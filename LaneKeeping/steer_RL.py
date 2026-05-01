import carla
import random
import math
import csv
import gymnasium as gym
from gymnasium import spaces
import numpy as np

def if_passed(car, waypoint):
    v = car.get_transform().location - waypoint.transform.location
    v_xy = carla.Vector3D(v.x, v.y, 0)
    vw = waypoint.transform.get_forward_vector()
    vw_xy = carla.Vector3D(vw.x, vw.y, 0)
    angle = math.degrees(v_xy.get_vector_angle(vw_xy))
    if (angle <= 90):
        return True
    return False

class laneEnv(gym.Env):
    def __init__(self, render_mode=None):

        # connect to the carla server
        client = carla.Client('localhost', 2000)
        client.set_timeout(20)

        # world setup
        client.load_world('Town03')
        self._world = client.get_world()

        # tick settings
        settings = self._world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        settings.no_rendering_mode = True
        self._world.apply_settings(settings)

        # spectator setup
        self._spectator = self._world.get_spectator()

        # get car blueprint and spawn points
        self._ego_blueprint = self._world.get_blueprint_library().find('vehicle.lincoln.mkz_2020')
        self._ego_blueprint.set_attribute('role_name', 'hero')
        self._spawn_points = self._world.get_map().get_spawn_points()
        
        # observe the lateral and heading value
        self.observation_space = spaces.Dict(
            {
                "lateral_error": spaces.Box(low=-10.0, high=10.0, shape=(1, ), dtype=np.float32),
                "steer_error": spaces.Box(low=-1.0, high=1.0, shape=(1, ), dtype=np.float32)
            }
        )

        # the action space is just the steer angle, which ranges from [-1.0, 1.0]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1, ), dtype=np.float32)

        self._location = None
        # self._steer_angle = 0 
        self._ego_car = None
        self._waypoint_list = None
        self._cur_idx = 0
        self._reach_end = False

        self._action = 0
        self._last_action = 0
        self._reward = 0
        self._step_count = 0

        self._episode_lateral_error = 0.0
        self._episode_steps = 0
        self._max_lateral_error = 0.0

        self.render_mode = render_mode
    
    # return the car's current location and steer angle
    def _get_info(self):
        mean_lat_error = 0.0
        if self._episode_steps > 0:
            mean_lat_error = self._episode_lateral_error / self._episode_steps
        return {
            "max_lateral_error_meters": self._max_lateral_error,
            "mean_lateral_error_meters": mean_lat_error,
            "steps_survived": self._step_count,
            "reached_destination": self._reach_end
        }

    def safe_angle(self, v1, v2):
        a = np.array([v1.x, v1.y])
        b = np.array([v2.x, v2.y])

        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a < 1e-6 or norm_b < 1e-6:
            return 0.0  # avoid NaN

        cos_theta = np.dot(a, b) / (norm_a * norm_b)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)

        return math.acos(cos_theta)
    
    # it should calculate the lateral error and the steer_error
    # def _get_obs(self):
    #     # cur_waypoint = self._waypoint_list[self._cur_idx]
    #     # passed = if_passed(self._ego_car, cur_waypoint)

    #     base_waypoint = self._waypoint_list[self._cur_idx]
    #     passed = if_passed(self._ego_car, base_waypoint)

    #     # boundary check and check if the car has passed the previous waypoint
    #     if (self._cur_idx+1 < len(self._waypoint_list)):
    #         if (passed):
    #             self._cur_idx += 1
    #             cur_waypoint = self._waypoint_list[self._cur_idx]
    #     else:
    #         self._reach_end = True

    #     look_ahead = 5
    #     target_idx = min(self._cur_idx+ look_ahead, len(self._waypoint_list)-1)
    #     cur_waypoint = self._waypoint_list[target_idx]

    #     cur_waypoint_forward = cur_waypoint.transform.get_forward_vector()
    #     w_forward_xy = carla.Vector3D(cur_waypoint_forward.x, cur_waypoint_forward.y, 0)

    #     # calulate the lateral error
    #     self._location = self._ego_car.get_transform().location
    #     x = self._location.x
    #     y = self._location.y
    #     car_xy = carla.Vector3D(x, y, 0)

    #     cur_waypoint_location = cur_waypoint.transform.location
    #     w_x = cur_waypoint_location.x
    #     w_y = cur_waypoint_location.y
    #     w_xy = carla.Vector3D(w_x, w_y, 0)

    #     v_xy = car_xy - w_xy
    #     norm = np.linalg.norm(np.array([v_xy.x, v_xy.y]))

    #     # compute the cross product to get the direction 
    #     cross1 = v_xy.cross(w_forward_xy)
    #     direction1 = 0
    #     if (cross1.z < 0):
    #         direction1 = -1
    #     else:
    #         direction1 = 1
    #     theta = self.safe_angle(v_xy, w_forward_xy)
    #     lateral_error = direction1 * norm * math.sin(theta)

    #     # calculate the steer error
    #     car_forward = self._ego_car.get_transform().get_forward_vector()
    #     car_forward_xy = carla.Vector3D(car_forward.x, car_forward.y, 0)
    #     steer_radian = self.safe_angle(car_forward_xy, w_forward_xy)
    #     cross2 = car_forward_xy.cross(w_forward_xy)
    #     direction2 = 0
    #     if (cross2.z < 0):
    #         direction2 = -1
    #     else:
    #         direction2 = 1
    #     steer_angle = math.degrees(steer_radian)
    #     cur_steer_error = (direction2 * steer_angle) / 180.0 # normalize to [-1, 1]
    #     if not np.isfinite(lateral_error):
    #         lateral_error = 0.0

    #     if not np.isfinite(cur_steer_error):
    #         cur_steer_error = 0.0
        
    #     lateral_error = np.clip(lateral_error, -10.0, 10.0)
    #     cur_steer_error = np.clip(cur_steer_error, -1.0, 1.0)
    #     return {"lateral_error": np.array([lateral_error], dtype = np.float32), "steer_error": np.array([cur_steer_error], dtype = np.float32)}
    
    def _get_obs(self):
        # 1. BASE WAYPOINT (Right under the car)
        base_waypoint = self._waypoint_list[self._cur_idx]
        passed = if_passed(self._ego_car, base_waypoint)

        if (self._cur_idx+1 < len(self._waypoint_list)):
            if (passed):
                self._cur_idx += 1
                base_waypoint = self._waypoint_list[self._cur_idx]
        else:
            self._reach_end = True

        # 2. FUTURE WAYPOINT (Look-ahead)
        look_ahead = 20
        target_idx = min(self._cur_idx + look_ahead, len(self._waypoint_list)-1)
        future_waypoint = self._waypoint_list[target_idx]

        # --- CALCULATE LATERAL ERROR (Using BASE Waypoint) ---
        base_forward = base_waypoint.transform.get_forward_vector()
        b_forward_xy = carla.Vector3D(base_forward.x, base_forward.y, 0)

        self._location = self._ego_car.get_transform().location
        car_xy = carla.Vector3D(self._location.x, self._location.y, 0)

        b_loc = base_waypoint.transform.location
        b_xy = carla.Vector3D(b_loc.x, b_loc.y, 0)

        v_xy = car_xy - b_xy
        norm = np.linalg.norm(np.array([v_xy.x, v_xy.y]))

        cross1 = v_xy.cross(b_forward_xy)
        direction1 = -1 if cross1.z < 0 else 1
        theta = self.safe_angle(v_xy, b_forward_xy)
        lateral_error = direction1 * norm * math.sin(theta)

        # --- CALCULATE STEER ERROR (Using FUTURE Waypoint) ---
        future_forward = future_waypoint.transform.get_forward_vector()
        f_forward_xy = carla.Vector3D(future_forward.x, future_forward.y, 0)

        car_forward = self._ego_car.get_transform().get_forward_vector()
        car_forward_xy = carla.Vector3D(car_forward.x, car_forward.y, 0)
        
        steer_radian = self.safe_angle(car_forward_xy, f_forward_xy)
        cross2 = car_forward_xy.cross(f_forward_xy)
        direction2 = -1 if cross2.z < 0 else 1
        
        steer_angle = math.degrees(steer_radian)
        cur_steer_error = (direction2 * steer_angle) / 180.0 

        # --- NORMALIZE AND RETURN ---
        if not np.isfinite(lateral_error): lateral_error = 0.0
        if not np.isfinite(cur_steer_error): cur_steer_error = 0.0
        
        lateral_error = np.clip(lateral_error, -10.0, 10.0)
        cur_steer_error = np.clip(cur_steer_error, -1.0, 1.0)
        
        return {"lateral_error": np.array([lateral_error], dtype=np.float32), 
                "steer_error": np.array([cur_steer_error], dtype=np.float32)}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # destroy the previous car
        if(self._ego_car is not None):
            self._ego_car.destroy()

        self._reach_end = False
        self._step_count = 0
        self._episode_lateral_error = 0.0
        self._episode_steps = 0
        self._max_lateral_error = 0.0
        self._last_action = 0.0

        # spawn_point = self._spawn_points[0]
        # self._ego_car = self._world.spawn_actor(self._ego_blueprint, spawn_point)
        
        # spawn the ego vihecle at a random spawn point on the map
        self._ego_car = self._world.spawn_actor(self._ego_blueprint, random.choice(self._spawn_points))

        # update the world
        self._world.tick()

        # get all the way points of the current lane every d meters
        wp = self._world.get_map().get_waypoint(self._ego_car.get_location())
        # Change here (bigger d = less waypoints)
        d = 0.5
        self._cur_idx = 0
        self._waypoint_list = wp.next_until_lane_end(d)
        
        observation = self._get_obs()
        info = self._get_info()

        return observation, info
    
    # update the location and steer of the ego car, calculate the reward
    def step(self, action):
        terminated = False
        truncated = False

        # we only control the steer angle (action is an np.arrary with shape(1, ), so action[0])
        self._action = float(action[0])

        # we FIRST apply the control to the ego car
        # Change throttle (maybe 0.35 or 0.4) (0.5 too fast)
        self._ego_car.apply_control(carla.VehicleControl(throttle = 0.4, steer = self._action))

        # update the world
        self._world.tick()

        # we THEN get the observation
        # Change here
        observation = self._get_obs()
        velocity = self._ego_car.get_velocity()
        speed = np.linalg.norm(np.array([velocity.x, velocity.y]))
        speed_reward = speed * 0.1
        
        lat_penalty = -0.5 *abs(observation["lateral_error"][0])
        steer_penalty = - 1.0 * abs(observation["steer_error"][0])
        steer_delta= abs(self._action - self._last_action)
        smoothness_penalty = -0.2 * steer_delta
        self._reward = speed_reward + lat_penalty + steer_penalty + smoothness_penalty
        self._last_action = self._action
        # print(f"Total Reward: {self._reward:.3f} | Lat Penalty: {lat_penalty:.3f} | Steer Penalty: {steer_penalty:.3f} | Action: {self._action:.2f}")

        #spectator pos update
        ego_transform = self._ego_car.get_transform()
        forward = self._ego_car.get_transform().get_forward_vector()
        ego_transform.location.x -= 8 * forward.x
        ego_transform.location.y -= 8 * forward.y
        ego_transform.location.z += 3
        ego_transform.rotation.pitch = -20
        self._spectator.set_transform(ego_transform)

        # get the lane width
        cur_waypoint = self._waypoint_list[self._cur_idx]
        width = cur_waypoint.lane_width

        # print(f"lateral: {observation['lateral_error'][0]}, width/2: {width/2.0}")

        # if the car gets stuck, terminate (skip first 20 steps)rint(f"lateral: {observation['lateral_error'][0]}, width/2: {width/2.0}")
        self._step_count += 1
    
        if (self._step_count > 100 and speed < 0.1):
            terminated = True
        # if the car goes out of the lane, terminate
        if (abs(observation["lateral_error"][0]) > width/2.0):
            terminated = True
            self._reward -= 50 # Terminal crash penalty
        # if reach the end of waypoints
        if (self._reach_end):
            truncated = True
        
        lat = abs(observation["lateral_error"][0])

        self._episode_lateral_error += lat
        self._max_lateral_error = max(self._max_lateral_error, lat)
        self._episode_steps += 1

        # get info
        info = self._get_info()

        return observation, self._reward, terminated, truncated, info
    
    def render(self):
        print(self._reward)





