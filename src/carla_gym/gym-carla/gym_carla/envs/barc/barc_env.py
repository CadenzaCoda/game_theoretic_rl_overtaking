import copy
import time
from typing import Tuple, Optional, Dict, Any, Union, List

from gymnasium.core import ActType, ObsType

from mpclab_common.pytypes import VehicleState, VehicleActuation, VehiclePrediction, Position, ParametricPose, \
    BodyLinearVelocity, OrientationEuler, BodyAngularVelocity
# from utils import data_utils
import numpy as np
# import utils.data_fitting as fit

import gymnasium as gym
from gymnasium import spaces
# from utils.data_utils import DynamicsDataset as DD

# from barc_gym.mpclab_simulation.mpclab_python_simulation.utils.renderer import LMPCVisualizer
from mpclab_common.track import get_track
from mpclab_common.models.dynamics_models import CasadiDynamicBicycle, CasadiDynamicCLBicycle, DynamicBicycleConfig

from loguru import logger

from gym_carla.envs.utils.renderer import LMPCVisualizer, MultiVehicleVisualizer
from mpclab_simulation.dynamics_simulator import DynamicsSimulator


class BarcEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, track_name, t0=0., dt=0.1, dt_sim=0.01, max_n_laps=100,
                 do_render=False, enable_camera=True, host='localhost', port=2000):
        self.track_obj = get_track(track_name)
        # self.track_obj.slack = 1
        self.t0 = t0  # Constant
        self.dt = dt
        self.dt_sim = dt_sim
        self.max_n_laps = max_n_laps
        self.do_render = do_render
        self.enable_camera = enable_camera
        self.track_name = track_name
        self.host = host
        self.port = port

        L = self.track_obj.track_length
        H = self.track_obj.half_width
        VL = 0.37
        VW = 0.195
        sim_dynamics_config = DynamicBicycleConfig(dt=dt_sim,
                                                   model_name='dynamic_bicycle',
                                                   noise=False,
                                                   discretization_method='rk4',
                                                   simple_slip=False,
                                                   tire_model='pacejka',
                                                   mass=2.2187,
                                                   yaw_inertia=0.02723,
                                                   wheel_friction=0.9,
                                                   pacejka_b_front=5.0,
                                                   pacejka_b_rear=5.0,
                                                   pacejka_c_front=2.28,
                                                   pacejka_c_rear=2.28)
        # dynamics_simulator = DynamicsSimulator(t, sim_dynamics_config, delay=[0.1, 0.1], track=track_obj)
        self.dynamics_simulator = DynamicsSimulator(t0, sim_dynamics_config, delay=None, track=self.track_obj)
        if enable_camera:
            from gym_carla.envs.barc.cameras.carla_bridge import CarlaConnector
            self.camera_bridge = CarlaConnector(self.track_name, host=self.host, port=self.port)
        else:
            self.camera_bridge = None

        self.visualizer = LMPCVisualizer(track_obj=self.track_obj, VL=VL, VW=VW)

        self.sim_state: VehicleState = None
        self.last_state: VehicleState = None

        # Additional information fields
        self.lap_speed = []
        self.lap_no = 0

        observation_space = dict(
            gps=spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
            velocity=spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
            state=spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32),
        )
        if self.enable_camera:
            from gym_carla.envs.barc.cameras.carla_bridge import CarlaConnector
            # All images are channel-first.
            observation_space.update(dict(
                camera=spaces.Box(low=0, high=255, shape=(self.camera_bridge.height, self.camera_bridge.width, 3),
                                  dtype=np.uint8),
                # depth=spaces.Box(low=0, high=255, shape=(3, H, W), dtype=np.uint8),
                # imu=spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float64),
            ))

        self.observation_space = spaces.Dict(observation_space)
        # self.action_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float64)
        self._action_bounds = np.array([2, 0.45])
        self.action_space = spaces.Box(low=-self._action_bounds,
                                       high=self._action_bounds, dtype=np.float64)

        self.t = None
        self.max_lap_speed = self.min_lap_speed = self._sum_lap_speed = self.eps_len = 0

    def get_track(self):
        return self.track_obj

    def bind_controller(self, controller):
        self.visualizer.bind_controller(controller)

    def clip_action(self, action):
        return np.clip(action, -self._action_bounds, self._action_bounds)

    def _is_new_lap(self):
        def orientation(p, q, r):
            val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
            if val == 0:
                return 0  # Collinear
            return 1 if val > 0 else 2  # Clockwise or Counterclockwise

        def on_segment(p, q, r):
            return (min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and
                    min(p[1], r[1]) <= q[1] <= max(p[1], r[1]))

        def do_intersect(p1, q1, p2, q2):
            o1 = orientation(p1, q1, p2)
            o2 = orientation(p1, q1, q2)
            o3 = orientation(p2, q2, p1)
            o4 = orientation(p2, q2, q1)

            # General case
            if o1 != o2 and o3 != o4:
                return True

            # Special cases
            if o1 == 0 and on_segment(p1, p2, q1):
                return True
            if o2 == 0 and on_segment(p1, q2, q1):
                return True
            if o3 == 0 and on_segment(p2, p1, q2):
                return True
            if o4 == 0 and on_segment(p2, q1, q2):
                return True

            return False

        return do_intersect(np.array([self.sim_state.x.x, self.sim_state.x.y]),
                            np.array([self.last_state.x.x, self.last_state.x.y]),
                            np.array([0, self.track_obj.half_width]),
                            np.array([0, -self.track_obj.half_width]))

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
    ) -> Tuple[ObsType, dict]:
        if seed is not None:
            np.random.seed(seed)
        if (options is not None and options.get('render')) or self.do_render:
            self.visualizer.reset()
        # elif self.visualizer is not None:
        else:
            self.visualizer.close()
        if options.get('spawning') == 'fixed':
            logger.debug("Respawning at fixed location.")
            self.sim_state = VehicleState(t=0.0,
                                          p=ParametricPose(s=0.1, x_tran=0),
                                          e=OrientationEuler(psi=0),
                                          v=BodyLinearVelocity(v_long=0.5, v_tran=0),
                                          w=BodyAngularVelocity(w_psi=0))
        else:
            self.sim_state = VehicleState(t=0.0,
                                          p=ParametricPose(s=np.random.uniform(0.1, self.track_obj.track_length - 2),
                                                           x_tran=np.random.uniform(
                                                               -self.track_obj.half_width - self.track_obj.slack,
                                                               self.track_obj.half_width + self.track_obj.slack),
                                                           e_psi=np.random.uniform(-np.pi / 6, np.pi / 6), ),
                                          # e=OrientationEuler(psi=0),
                                          v=BodyLinearVelocity(v_long=np.random.uniform(0.5, 2), v_tran=0),
                                          w=BodyAngularVelocity(w_psi=0))
        self.track_obj.local_to_global_typed(self.sim_state)
        self.last_state = copy.deepcopy(self.sim_state)

        self.t = self.t0
        self.lap_start = self.t
        self.lap_speed = []
        self.lap_no = 0

        self._reset_speed_stats()
        self.eps_len = 1

        return self._get_obs(), self._get_info()

    def _update_speed_stats(self):
        v = np.linalg.norm([self.sim_state.v.v_long, self.sim_state.v.v_tran])
        self.max_lap_speed = max(self.max_lap_speed, v)
        self.min_lap_speed = min(self.min_lap_speed, v)
        self._sum_lap_speed += v
        self.eps_len += 1

    def _reset_speed_stats(self):
        v = np.linalg.norm([self.sim_state.v.v_long, self.sim_state.v.v_tran])
        self.max_lap_speed = self.min_lap_speed = self._sum_lap_speed = v

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        action = np.clip(action, -self._action_bounds, self._action_bounds)
        self.sim_state.u.u_a, self.sim_state.u.u_steer = action
        self.last_state = copy.deepcopy(self.sim_state)
        # self.sim_state.copy_control(action)
        self.render()

        truncated = False
        try:
            self.dynamics_simulator.step(self.sim_state, T=self.dt)
            # _slack, self.track_obj.slack = self.track_obj.slack, 0.5
            self.track_obj.global_to_local_typed(self.sim_state)
            # self.track_obj.slack = _slack
            # TODO: Future - replace the dynamics_simulator with other dynamics functions. (e.g. data-driven models)
        except ValueError as e:
            truncated = True  # The control action may drive the vehicle out of the track during the internal steps. Process the exception here immediately.

        self.t += self.dt
        self._update_speed_stats()

        obs = self._get_obs()
        rew = self._get_reward()
        terminated = self._get_terminal()
        truncated = truncated or self._get_truncated()
        info = self._get_info()

        if terminated:
            logger.info(
                f"Lap {self.lap_no} finished in {info['lap_time']:.1f} s. "
                f"avg_v = {info['avg_lap_speed']:.4f}, max_v = {info['max_lap_speed']:.4f}, "
                f"min_v = {info['min_lap_speed']:.4f}")
            self.lap_no += 1
            self.lap_start = self.t
            self._reset_speed_stats()
            self.eps_len = 1

        return obs, rew, terminated, truncated, info

    def render(self):
        if not self.do_render:
            return
        self.visualizer.step(self.sim_state)

    def _get_obs(self) -> Dict[str, np.ndarray]:
        ob = {
            'gps': np.array([self.sim_state.x.x, self.sim_state.x.y, self.sim_state.e.psi], dtype=np.float32),
            'velocity': np.array([self.sim_state.v.v_long, self.sim_state.v.v_tran, self.sim_state.w.w_psi],
                                 dtype=np.float32),
            'state': np.array([self.sim_state.v.v_long, self.sim_state.v.v_tran, self.sim_state.w.w_psi,
                               self.sim_state.p.s, self.sim_state.p.x_tran, self.sim_state.p.e_psi], dtype=np.float32),
            # self.sim_state.x.x, self.sim_state.x.y, self.sim_state.e.psi], dtype=np.float32),
            # For backward compatibility.
        }
        if self.enable_camera:
            while True:
                try:
                    camera = self.camera_bridge.query_rgb(self.sim_state)
                    break
                except RuntimeError as e:
                    logger.error(e)
                from gym_carla.envs.barc.cameras.carla_bridge import CarlaConnector
                while True:
                    time.sleep(10)
                    try:
                        self.camera_bridge = CarlaConnector(self.track_name, self.host, self.port)
                        break
                    except RuntimeError as e:
                        logger.error(e)
            ob.update({
                'camera': camera,
                # 'depth': None,
                # 'imu': None,
            })
        return ob

    def _get_reward(self) -> float:
        # Initialize reward
        reward = 0.0
        
        # Check for collisions with track boundary
        if np.abs(self.sim_state.p.x_tran) > self.track_obj.half_width:
            return -100.0
        
        # Check for slow vehicles or wrong direction
        if self.sim_state.v.v_long < 0.25 or np.abs(self.sim_state.p.e_psi) > np.pi / 2:
            return -100.0
        
        # Calculate distance between vehicles
        ego_pos = np.array([self.sim_state.x.x, self.sim_state.x.y])
        opp_pos = np.array([self.last_state.x.x, self.last_state.x.y])
        distance = np.linalg.norm(ego_pos - opp_pos)
        
        # Check for collisions with opponent (simple distance-based check)
        if distance < 0.3:
            return -100.0
        
        # Reward for successful overtaking
        if self._is_new_lap():
            return 100.0
            
        # Reward components
        k_progress = 0.5  # Increased from 0.1
        k_relative = 0.3  # Increased from 0.1
        k_speed = 0.2     # New component for speed
        k_lateral = 0.1   # New component for lateral position
        
        # 1. Progress reward - encourage forward movement
        progress = self.sim_state.p.s - self.last_state.p.s
        reward_progress = k_progress * progress
        
        # 2. Relative distance reward - encourage getting closer to opponent
        safe_distance_min = 0.5
        distance_threshold = -0.5
        epsilon = 1e-1
        
        if distance < safe_distance_min:
            # Penalize for being too close (exponential penalty)
            relative_reward = -k_relative * np.exp(safe_distance_min - distance)
        else:
            # Reward for closing the gap (with diminishing returns)
            relative_reward = k_relative * (1.0 / (distance + epsilon))
        
        # 3. Speed reward - encourage maintaining good speed
        speed = self.sim_state.v.v_long
        target_speed = 5.0  # Target speed in m/s
        speed_reward = k_speed * (1.0 - np.abs(speed - target_speed) / target_speed)
        
        # 4. Lateral position reward - encourage staying in the middle of the track
        lateral_error = np.abs(self.sim_state.p.x_tran)
        max_lateral_error = self.track_obj.half_width * 0.8  # 80% of track width
        lateral_reward = k_lateral * (1.0 - lateral_error / max_lateral_error)
        
        # Combine all rewards
        reward = reward_progress + relative_reward + speed_reward + lateral_reward
        
        # Add small negative reward to encourage efficiency
        reward -= 0.01
        
        return reward

    def _get_terminal(self) -> bool:
        """
        Note: Now `terminated` means the beginning of a new lap.
        To collect long trajectories, this should NOT be used to reset the rollout.
        Instead, use this as a trigger to e.g. update the expert.
        """
        return self._is_new_lap() and self.sim_state.p.s < self.last_state.p.s

    def _get_truncated(self) -> bool:
        """
        Note: Now `truncated` means constraint violation or maximum lap reached.
        This should be used for truncating the rollout (resetting the simulation).
        """
        conditions = [
            np.abs(self.sim_state.p.x_tran) > self.track_obj.half_width,  # Out of track.
            self.lap_no >= self.max_n_laps,  # Maximum lap number reached.
            self.sim_state.v.v_long < 0.25,
            np.abs(self.sim_state.p.e_psi) > np.pi / 2,
        ]
        return any(conditions)

    def _get_info(self) -> Dict[str, Union[VehicleState, int, float]]:
        return {
            'vehicle_state': copy.deepcopy(self.sim_state),  # Ground truth vehicle state.
            'lap_no': self.lap_no,  # Lap number
            'terminated': self._is_new_lap(),
            'avg_lap_speed': self._sum_lap_speed / self.eps_len,  # Mean velocity of the current lap.
            'max_lap_speed': self.max_lap_speed,  # Max velocity of the current lap.
            'min_lap_speed': self.min_lap_speed,  # Min velocity of the current lap.
            'lap_time': self.eps_len * self.dt,  # Time elapsed so far in the current lap.
        }


class MultiBarcEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, track_name, t0=0., dt=0.1, dt_sim=0.01, max_n_laps=5, max_steps=300,
                 do_render=False, enable_camera=True, host='localhost', port=2000):
        self.track_obj = get_track(track_name)
        # Fixed to 2 vehicles for racing
        self.n_vehicles = 2
        # self.track_obj.slack = 1
        self.t0 = t0  # Constant
        self.dt = dt
        self.dt_sim = dt_sim
        self.max_n_laps = max_n_laps
        self.max_steps = max_steps
        self.do_render = do_render
        self.enable_camera = enable_camera
        self.track_name = track_name
        self.host = host
        self.port = port

        L = self.track_obj.track_length
        H = self.track_obj.half_width
        VL = 0.37
        VW = 0.195

        sim_dynamics_config = DynamicBicycleConfig(dt=dt_sim,
                                                   model_name='dynamic_bicycle',
                                                   noise=False,
                                                   discretization_method='rk4',
                                                   simple_slip=False,
                                                   tire_model='pacejka',
                                                   mass=2.2187,
                                                   yaw_inertia=0.02723,
                                                   wheel_friction=0.9,
                                                   pacejka_b_front=5.0,
                                                   pacejka_b_rear=5.0,
                                                   pacejka_c_front=2.28,
                                                   pacejka_c_rear=2.28)

        # Create exactly 2 dynamics simulators
        self.dynamics_simulator = [
            DynamicsSimulator(t0, sim_dynamics_config, delay=None, track=self.track_obj) for _ in range(2)
        ]

        if enable_camera:
            from gym_carla.envs.barc.cameras.carla_bridge import CarlaConnector
            self.camera_bridge = CarlaConnector(self.track_name, host=self.host, port=self.port)
        else:
            self.camera_bridge = None

        # self.visualizer = LMPCVisualizer(track_obj=self.track_obj, VL=VL, VW=VW)
        self.visualizer = MultiVehicleVisualizer(track_obj=self.track_obj, VL=VL, VW=VW)

        self.sim_state: Optional[List[VehicleState]] = None
        self.last_state: Optional[List[VehicleState]] = None

        # Additional information fields
        self.lap_speed = []
        # self.lap_no = 0
        
        # Track relative distance between vehicles for overtaking detection
        self.rel_dist = 0.0
        self.last_rel_dist = 0.0
        self.lap_no = [0, 0]  # Track laps completed by each vehicle

        observation_space = dict(
            gps=spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
            velocity=spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
            state=spaces.Box(low=-np.inf, high=np.inf, shape=(2 * 9,), dtype=np.float32),  # Fixed to 2 vehicles
        )
        if self.enable_camera:
            from gym_carla.envs.barc.cameras.carla_bridge import CarlaConnector
            # All images are channel-first.
            observation_space.update(dict(
                camera=spaces.Box(low=0, high=255, shape=(self.camera_bridge.height, self.camera_bridge.width, 3),
                                  dtype=np.uint8),
                # depth=spaces.Box(low=0, high=255, shape=(3, H, W), dtype=np.uint8),
                # imu=spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float64),
            ))

        self.observation_space = spaces.Dict(observation_space)
        # Fixed action space for 2 vehicles
        self._action_bounds = np.tile(np.array([2, 0.45]), [2, 1])
        self.action_space = spaces.Box(low=-self._action_bounds,
                                       high=self._action_bounds,
                                       dtype=np.float64)

        self.t = None
        self.max_lap_speed = self.min_lap_speed = self._sum_lap_speed = self.eps_len = 0
        
        self.collision_threshold = 0.3
        self.low_speed_threshold = 0.25
        self.wrong_direction_threshold = np.pi / 2
        self.overtake_margin = -0.5
    def get_track(self):
        return self.track_obj

    def bind_controller(self, controller):
        self.visualizer.bind_controller(controller)

    def clip_action(self, action):
        return np.clip(action, -self._action_bounds, self._action_bounds)

    def _is_new_lap(self) -> List[bool]:
        def orientation(p, q, r):
            val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
            if val == 0:
                return 0  # Collinear
            return 1 if val > 0 else 2  # Clockwise or Counterclockwise

        def on_segment(p, q, r):
            return (min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and
                    min(p[1], r[1]) <= q[1] <= max(p[1], r[1]))

        def do_intersect(p1, q1, p2, q2):
            o1 = orientation(p1, q1, p2)
            o2 = orientation(p1, q1, q2)
            o3 = orientation(p2, q2, p1)
            o4 = orientation(p2, q2, q1)

            # General case
            if o1 != o2 and o3 != o4:
                return True

            # Special cases
            if o1 == 0 and on_segment(p1, p2, q1):
                return True
            if o2 == 0 and on_segment(p1, q2, q1):
                return True
            if o3 == 0 and on_segment(p2, p1, q2):
                return True
            if o4 == 0 and on_segment(p2, q1, q2):
                return True

            return False

        return [do_intersect(np.array([_state.x.x, _state.x.y]),
                            np.array([_prev.x.x, _prev.x.y]),
                            np.array([0, self.track_obj.half_width]),
                            np.array([0, -self.track_obj.half_width])) for _state, _prev in zip(self.sim_state, self.last_state)]

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
    ) -> Tuple[ObsType, dict]:
        if seed is not None:
            np.random.seed(seed)
        if (options is not None and options.get('render')) or self.do_render:
            self.visualizer.reset()
        elif self.visualizer is not None:
            self.visualizer.close()
        if options is not None and options.get('spawning') == 'fixed':
            logger.debug("Respawning at fixed location.")

            # Initialize exactly 2 vehicles with fixed spacing
            self.sim_state = [VehicleState(t=0.0,
                                          p=ParametricPose(s=0.1 + 2 * i, x_tran=0),
                                          e=OrientationEuler(psi=0),
                                          v=BodyLinearVelocity(v_long=0.5, v_tran=0),
                                          w=BodyAngularVelocity(w_psi=0)) for i in range(2)]
        else:
            # Initialize exactly 2 vehicles with random positions
            _s = np.random.uniform(0.1, self.track_obj.track_length - 4)
            self.sim_state = [VehicleState(t=0.0,
                                          p=ParametricPose(s=_s + 2 * i,
                                                           x_tran=np.random.uniform(
                                                               -self.track_obj.half_width / 2,
                                                               self.track_obj.half_width / 2),
                                                           e_psi=np.random.uniform(-np.pi / 6, np.pi / 6), ),
                                          v=BodyLinearVelocity(v_long=np.random.uniform(0.5, 2), v_tran=0),
                                          w=BodyAngularVelocity(w_psi=0)) for i in range(2)]
        for _state in self.sim_state:
            self.track_obj.local_to_global_typed(_state)
        self.last_state = copy.deepcopy(self.sim_state)

        self.t = self.t0
        self.lap_start = self.t
        self.lap_speed = []
        # self.lap_no = 0
        
        # Initialize relative distance and lap counters
        self.rel_dist = self.sim_state[1].p.s - self.sim_state[0].p.s
        self.last_rel_dist = self.rel_dist
        self.lap_no = [0, 0]

        self._reset_speed_stats()
        self.eps_len = 1

        return self._get_obs(), self._get_info()

    def _update_speed_stats(self):
        v = np.linalg.norm([self.sim_state[0].v.v_long, self.sim_state[0].v.v_tran])
        self.max_lap_speed = max(self.max_lap_speed, v)
        self.min_lap_speed = min(self.min_lap_speed, v)
        self._sum_lap_speed += v
        self.eps_len += 1

    def _reset_speed_stats(self):
        v = np.linalg.norm([self.sim_state[0].v.v_long, self.sim_state[0].v.v_tran])
        self.max_lap_speed = self.min_lap_speed = self._sum_lap_speed = v

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        action = np.clip(action, -self._action_bounds, self._action_bounds)
        
        # Apply actions to each vehicle
        for i, _state in enumerate(self.sim_state):
            _state.u.u_a, _state.u.u_steer = action[i]
            
        self.last_state = copy.deepcopy(self.sim_state)
        self.last_rel_dist = self.rel_dist  # Store the previous relative distance
        self.render()

        truncated = False
        try:
            # Step each vehicle's dynamics
            for i, _state in enumerate(self.sim_state):
                self.dynamics_simulator[i].step(_state, T=self.dt)
                self.track_obj.global_to_local_typed(_state)
        except ValueError as e:
            truncated = True  # The control action may drive the vehicle out of the track during the internal steps.

        # Update relative distance and lap counters
        self._update_relative_distance()
        
        self.t += self.dt
        self._update_speed_stats()

        obs = self._get_obs()
        rew = self._get_reward()
        terminated = self._get_terminal()
        truncated = truncated or self._get_truncated()
        info = self._get_info()

        if terminated:
            logger.info(
                f"Overtaking successful in {info['lap_time']:.1f} s. "
                f"avg_v = {info['avg_lap_speed']:.4f}, max_v = {info['max_lap_speed']:.4f}, "
                f"min_v = {info['min_lap_speed']:.4f}")
            # self.lap_no += 1
            # self.lap_start = self.t
            # self._reset_speed_stats()
            # self.eps_len = 1

        return obs, rew, terminated, truncated, info

    def render(self):
        # if not self.do_render:
            # return
        self.visualizer.step(self.sim_state)

    def _get_obs(self) -> Dict[str, np.ndarray]:
        # For backward compatibility, use the first vehicle's state for gps and velocity
        ob = {
            'gps': np.array([self.sim_state[0].x.x, self.sim_state[0].x.y, self.sim_state[0].e.psi], dtype=np.float32),
            'velocity': np.array([self.sim_state[0].v.v_long, self.sim_state[0].v.v_tran, self.sim_state[0].w.w_psi],
                                 dtype=np.float32),
            'state': np.array([[state.v.v_long, state.v.v_tran, state.w.w_psi,
                               state.p.s, state.p.x_tran, state.p.e_psi,
                               state.x.x, state.x.y, state.e.psi] for state in self.sim_state], dtype=np.float32).reshape(-1),
        }
        if self.enable_camera:
            while True:
                try:
                    camera = self.camera_bridge.query_rgb(self.sim_state[0])
                    break
                except RuntimeError as e:
                    logger.error(e)
                from gym_carla.envs.barc.cameras.carla_bridge import CarlaConnector
                while True:
                    time.sleep(10)
                    try:
                        self.camera_bridge = CarlaConnector(self.track_name, self.host, self.port)
                        break
                    except RuntimeError as e:
                        logger.error(e)
            ob.update({
                'camera': camera,
                # 'depth': None,
                # 'imu': None,
            })
        return ob
    
    def _get_reward(self) -> float:
        # Check for collisions with track boundary
        if np.abs(self.sim_state[0].p.x_tran) > self.track_obj.half_width:
            return -100.0
        
        # Check for slow vehicles or wrong direction
        if self.sim_state[0].v.v_long < self.low_speed_threshold or np.abs(self.sim_state[0].p.e_psi) > self.wrong_direction_threshold:
            return -100.0
        
        # Check for collisions with opponent (simple distance-based check)
        if np.linalg.norm(np.array([self.sim_state[0].x.x, self.sim_state[0].x.y]) - np.array([self.sim_state[1].x.x, self.sim_state[1].x.y])) < 0.3:
            return -100.0
        
        # Check for being very behind the opponent
        if self.rel_dist > self.track_obj.track_length * 0.8:
            return -100.0
        
        # Reward for successful overtaking
        if self._get_terminal():
            return 100.0
            
        k_progress = 1
        k_relative = 0.2  # 20 / (1 - 0.1) / 20
        k_catching_up = 10
        k_boundary = 0.2  # 20 / (1 - 0.9) / 20
        k_speed = 0.2  # 20 / (1 - 0.5) / 20
        safe_distance_min = 0.5
        reward_progress_decay = 0.95

        reward_progress = k_progress * max(0, self.sim_state[0].p.s - self.last_state[0].p.s) * reward_progress_decay ** self.eps_len

        physical_distance = np.linalg.norm([
            self.sim_state[0].x.x - self.sim_state[1].x.x,
            self.sim_state[0].x.y - self.sim_state[1].x.y
        ])

        # Penalty for being too close to the boundary
        boundary_penalty = -k_boundary * max(0, np.abs(self.sim_state[0].p.x_tran) / self.track_obj.half_width - 0.9)

        # Penalty for being too slow
        speed_penalty = -k_speed * max(0, 0.5 - self.sim_state[0].v.v_long)

        catching_up_reward = k_catching_up * (self.last_rel_dist - self.rel_dist)  # Reward for catching up
        if physical_distance < safe_distance_min:
            proximity_penalty = -k_relative * (safe_distance_min - physical_distance)  # Penalty for being too close to the opponent
        else:
            proximity_penalty = 0
        return reward_progress + catching_up_reward + proximity_penalty + boundary_penalty + speed_penalty - 0.1

    # def _get_reward(self) -> float:
    #     # Base components
    #     progress = self.sim_state[0].p.s - self.last_state[0].p.s
    #     if self._is_new_lap()[0]:
    #         progress += self.track_obj.track_length

    #     velocity = self.sim_state[0].v.v_long
    #     heading = np.abs(self.sim_state[0].p.e_psi)
        
    #     # Dynamic safety components
    #     physical_distance = np.linalg.norm([
    #         self.sim_state[0].x.x - self.sim_state[1].x.x,
    #         self.sim_state[0].x.y - self.sim_state[1].x.y
    #     ])
    #     track_offset = np.abs(self.sim_state[0].p.x_tran)
    #     safety_margin = 0.4  # meters
    #     max_speed = 3.
        
    #     # Progressive penalties (smooth instead of binary)
    #     safety_penalty = np.clip(1 - (physical_distance / safety_margin), 0, 1) * -10
    #     track_penalty = np.clip(track_offset / (self.track_obj.half_width * 0.9), 0, 1) * -5
        
    #     # Velocity incentives (encourage controlled speed)
    #     speed_bonus = np.clip(velocity / max_speed, 0, 1) * 2
    #     heading_penalty = np.clip(heading / (np.pi / 4), 0, 1) * -3
        
    #     # Overtaking incentives
    #     overtaking_bonus = np.clip(-self.rel_dist/2, -1, 1)  # Scales with lead
        
    #     # Progress reward with diminishing returns
    #     progress_reward = 0.2 * np.log(1 + progress)
        
    #     # Composite reward
    #     reward = (
    #         progress_reward +
    #         speed_bonus +
    #         overtaking_bonus +
    #         safety_penalty +
    #         track_penalty +
    #         heading_penalty
    #     )
        
    #     # Terminal conditions
    #     if self._get_terminal():
    #         reward += 50 + 20 * -self.rel_dist  # Scale with overtaking margin
    #     elif any([
    #         track_offset > self.track_obj.half_width,
    #         velocity < self.low_speed_threshold,
    #         self.rel_dist < self.collision_threshold
    #     ]):
    #         reward -= 20  # Reduced from -100
        
    #     return float(reward)


    def _get_terminal(self) -> bool:
        """
        Episode terminates when the agent successfully overtakes the opponent
        """
        # Check if agent has overtaken the opponent using relative distance
        was_behind = self.last_rel_dist >= self.overtake_margin
        is_ahead = self.rel_dist < self.overtake_margin
        
        return was_behind and is_ahead

    def _get_truncated(self) -> bool:
        """
        Episode is truncated if:
        1) Successful overtaking (we only learn the overtaking maneuver)
        2) Out of track
        3) Maximum time steps reached
        4) Any vehicle going too slow (< 0.25)
        5) Any vehicle going in the wrong way (e.psi > pi/2)
        """
        # Check for successful overtaking (already handled by termination)
        if self._get_terminal():
            return True
            
        # Check for out of track
        for i, state in enumerate(self.sim_state):
            if np.abs(state.p.x_tran) > self.track_obj.half_width:
                # logger.debug(f"Out of track: {np.abs(state.p.x_tran)} by vehicle {i}")
                return True
                
        # Check for maximum time steps (assuming max_steps is defined in __init__)
        # if hasattr(self, 'max_steps') and self.eps_len >= self.max_steps:
        #     return True
        # if any(lap_no >= self.max_n_laps for lap_no in self.lap_no):
            # logger.debug(f"Max laps reached: {self.lap_no}")
            # return True
        if self.eps_len > self.max_steps:
            return True
            
        # Check for slow vehicles or wrong direction
        for i, state in enumerate(self.sim_state):
            if state.v.v_long < self.low_speed_threshold or np.abs(state.p.e_psi) > self.wrong_direction_threshold:
                # logger.debug(f"Slow vehicle: {state.v.v_long} or wrong direction: {state.p.e_psi} by vehicle {i}")
                return True
            
        # Check for collision with opponent
        if np.linalg.norm(np.array([self.sim_state[0].x.x, self.sim_state[0].x.y]) - np.array([self.sim_state[1].x.x, self.sim_state[1].x.y])) < self.collision_threshold:
            # logger.debug(f"Collision: {np.linalg.norm(np.array([self.sim_state[0].x.x, self.sim_state[0].x.y]) - np.array([self.sim_state[1].x.x, self.sim_state[1].x.y]))}")
            return True

        # Stop if the ego is very far behind the opponent
        if self.rel_dist > self.track_obj.track_length * 0.8:
            # logger.debug(f"Ego is very far behind the opponent: {self.rel_dist}")
            return True
        
        return False

    def _get_info(self) -> Dict[str, Union[List[VehicleState], int, float]]:
        return {
            'vehicle_state': copy.deepcopy(self.sim_state),  # Ground truth vehicle state.
            # 'lap_no': self.lap_no,  # Lap number
            'terminated': self._is_new_lap(),
            'avg_lap_speed': self._sum_lap_speed / self.eps_len,  # Mean velocity of the current lap.
            'max_lap_speed': self.max_lap_speed,  # Max velocity of the current lap.
            'min_lap_speed': self.min_lap_speed,  # Min velocity of the current lap.
            'lap_time': self.eps_len * self.dt,  # Time elapsed so far in the current lap.
            'relative_distance': self.rel_dist,  # Relative distance between vehicles
            'lap_no': self.lap_no,  # Laps completed by each vehicle
        }
    
    def _update_relative_distance(self):
        """Update the relative distance between vehicles, accounting for lap transitions"""
        # Check for lap transitions for both vehicles
        for i in range(2):
            if self._is_new_lap()[i] and self.sim_state[i].p.s < self.last_state[i].p.s:
                self.lap_no[i] += 1
                logger.info(f"Lap {self.lap_no[i]} completed by vehicle {i}.")
                
        # Calculate relative distance in s-coordinate
        s_diff = self.sim_state[1].p.s - self.sim_state[0].p.s
        
        # Adjust for lap differences
        lap_diff = self.lap_no[1] - self.lap_no[0]
        
        # Update relative distance
        self.rel_dist = s_diff + lap_diff * self.track_obj.track_length
        
        # Handle the case where the relative distance jumps due to lap transitions
        # If the jump is too large, it's likely due to a lap transition
        if abs(self.rel_dist - self.last_rel_dist) > self.track_obj.track_length / 2:
            # Adjust the relative distance to be consistent with the previous step
            if self.rel_dist > self.last_rel_dist:
                self.rel_dist -= self.track_obj.track_length
            else:
                self.rel_dist += self.track_obj.track_length
