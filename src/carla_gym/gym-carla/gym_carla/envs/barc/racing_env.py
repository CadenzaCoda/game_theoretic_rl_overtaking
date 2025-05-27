import copy
import time
from typing import Optional, List, Tuple, Dict, Union

import numpy as np
from gymnasium import spaces
from gymnasium.core import ObsType, ActType
from loguru import logger

from gym_carla.envs.utils.renderer import MultiVehicleVisualizer
from mpclab_common.models.model_types import DynamicBicycleConfig
from mpclab_common.pytypes import VehicleState, ParametricPose, OrientationEuler, BodyLinearVelocity, \
    BodyAngularVelocity
from mpclab_common.track import get_track
from mpclab_simulation.dynamics_simulator import DynamicsSimulator
from ray.rllib.env.multi_agent_env import MultiAgentEnv


class RacingEnv(MultiAgentEnv):
    metadata = {'render.modes': ['human']}

    def __init__(self, track_name, t0=0., dt=0.1, dt_sim=0.01, target_laps=3, max_steps=300,
                 do_render=False, enable_camera=False, host='localhost', port=2000,
                 discrete_action: bool = False):
        super().__init__()
        self.possible_agents = ["ego", "oppo"]
        self.agents = ["ego", "oppo"]
        self.track_obj = get_track(track_name)
        self.discrete = discrete_action
        # Fixed to 2 vehicles for racing
        self.n_vehicles = 2
        self.target_laps = target_laps  # Number of laps required to win
        self.t0 = t0
        self.dt = dt
        self.dt_sim = dt_sim
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

        self.sim_dynamics_config = DynamicBicycleConfig(dt=dt_sim,
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
            DynamicsSimulator(t0, self.sim_dynamics_config, delay=None, track=self.track_obj) for _ in range(2)
        ]

        if enable_camera:
            from gym_carla.envs.barc.cameras.carla_bridge import CarlaConnector
            self.camera_bridge = CarlaConnector(self.track_name, host=self.host, port=self.port)
        else:
            self.camera_bridge = None

        self.visualizer = MultiVehicleVisualizer(track_obj=self.track_obj, VL=VL, VW=VW)

        self.sim_state: Optional[List[VehicleState]] = None
        self.last_state: Optional[List[VehicleState]] = None

        # Track laps completed by each vehicle
        self.lap_no = [0, 0]
        self.last_lap_no = [0, 0]

        observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2 * 10,), dtype=np.float32)

        self.observation_space = spaces.Dict({
            'ego': observation_space,
            'oppo': observation_space,
        })

        # Fixed action space for 2 vehicles
        self._action_bounds = np.tile(np.array([2, 0.45]), [2, 1])
        if self.discrete:
            self.u_a_space = np.linspace(-2, 2, 32, endpoint=True, dtype=np.float32)
            self.u_steer_space = np.linspace(-0.45, 0.45, 32, endpoint=True, dtype=np.float32)
            action_space = spaces.MultiDiscrete([len(self.u_a_space), len(self.u_steer_space)])
        else:
            self.u_a_space = None
            self.u_steer_space = None
            action_space = spaces.Box(low=-self._action_bounds, high=self._action_bounds, dtype=np.float32)

        self.action_space = spaces.Dict({
            "ego": action_space,
            "oppo": action_space,
        })

        self.t = None
        self.max_eps_speed, self.min_eps_speed, self._sum_eps_speed = [[0, 0] for _ in range(3)]
        self.eps_len = 0

        self.collision_threshold = 0.3
        self.low_speed_threshold = 0.25
        self.wrong_direction_threshold = np.pi / 2

        # Cache for step results
        self._cached_results = {
            'success': [False, False],
            'failure': [False, False],
            'draw': False,
            'collision': False
        }

    def get_track(self):
        return self.track_obj

    def decode_action(self, action):
        if not self.discrete:
            raise ValueError
        return np.array([(self.u_a_space[_action[0]], self.u_steer_space[_action[1]]) for _action in action])

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

            if o1 != o2 and o3 != o4:
                return True

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
                           np.array([0, -self.track_obj.half_width])) for _state, _prev in
                zip(self.sim_state, self.last_state)]

    def _check_collision(self) -> bool:
        """Check if there is a collision between vehicles"""
        if self.sim_state is None or len(self.sim_state) < 2:
            return False
            
        pos1 = np.array([self.sim_state[0].x.x, self.sim_state[0].x.y])
        pos2 = np.array([self.sim_state[1].x.x, self.sim_state[1].x.y])
        distance = np.linalg.norm(pos1 - pos2)
        return distance < self.collision_threshold

    def _update_cached_results(self, simulator_failures: Optional[List[bool]] = None):
        """Update cached results for the current step"""
        if simulator_failures is None:
            simulator_failures = [False, False]
            
        # Update success and failure states
        for i in range(2):
            self._cached_results['success'][i] = self._is_success(i)
            # Failure includes both regular failure conditions and simulator failures
            self._cached_results['failure'][i] = self._is_failure(i) or simulator_failures[i]
        
        # Check for collision
        self._cached_results['collision'] = self._check_collision()
        
        # Update draw condition
        self._cached_results['draw'] = (
            (self._cached_results['success'][0] and self._cached_results['success'][1]) or
            (self._cached_results['failure'][0] and self._cached_results['failure'][1]) or
            self._cached_results['collision']
        )

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
                                         v=BodyLinearVelocity(v_long=1.5 - i, v_tran=0),
                                         w=BodyAngularVelocity(w_psi=0)) for i in range(2)]
        for _state in self.sim_state:
            self.track_obj.local_to_global_typed(_state)
        self.last_state = copy.deepcopy(self.sim_state)

        self.t = self.t0
        lap_no = np.random.randint(0, self.target_laps)
        self.lap_no = [lap_no, lap_no]
        self.last_lap_no = [lap_no, lap_no]

        self._reset_speed_stats()
        self.eps_len = 1

        # Reset cached results
        self._cached_results = {
            'success': [False, False],
            'failure': [False, False],
            'draw': False,
            'collision': False
        }

        return self._get_obs(), self._get_info()

    def _update_speed_stats(self):
        for i, _state in enumerate(self.sim_state):
            v = np.linalg.norm([_state.v.v_long, _state.v.v_tran])
            self.max_eps_speed[i] = max(self.max_eps_speed[i], v)
            self.min_eps_speed[i] = min(self.min_eps_speed[i], v)
            self._sum_eps_speed[i] += v
        self.eps_len += 1

    def _reset_speed_stats(self):
        v = [np.linalg.norm([_state.v.v_long, _state.v.v_tran]) for _state in self.sim_state]
        self.max_eps_speed, self.min_eps_speed, self._sum_eps_speed = [v for _ in range(3)]

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        action = np.array([action['ego'], action['oppo']])
        if self.discrete:
            action = self.decode_action(action)
        action = np.clip(action, -self._action_bounds, self._action_bounds)

        # Apply actions to each vehicle
        for i, _state in enumerate(self.sim_state):
            _state.u.u_a, _state.u.u_steer = action[i]

        self.last_state = copy.deepcopy(self.sim_state)
        self.last_lap_no = copy.deepcopy(self.lap_no)
        self.render()

        # Track simulator failures
        simulator_failures = [False, False]
        for i, _state in enumerate(self.sim_state):
            try:
                # Step each vehicle's dynamics
                self.dynamics_simulator[i].step(_state, T=self.dt)
                self.track_obj.global_to_local_typed(_state)
            except ValueError as e:
                # Simulator failure (usually due to out of bounds) is a failure condition
                simulator_failures[i] = True
                logger.debug(f"Vehicle {i} failed due to simulator error: {str(e)}")

        # Update lap counters
        new_laps = self._is_new_lap()
        for i in range(2):
            if new_laps[i] and self.sim_state[i].p.s < self.last_state[i].p.s:
                self.lap_no[i] += 1

        self.t += self.dt
        self._update_speed_stats()
        
        # Update cached results for this step, including simulator failures
        self._update_cached_results(simulator_failures)

        obs = self._get_obs()
        rew = self._get_reward()
        terminated = self._get_terminal()
        truncated = self._get_truncated()
        info = self._get_info()

        return obs, rew, {"ego": terminated, "oppo": terminated, "__all__": terminated}, {"ego": truncated, "oppo": truncated, "__all__": truncated}, info

    def render(self):
        self.visualizer.step(self.sim_state)

    def _get_obs(self) -> Dict[str, np.ndarray]:
        ob = np.array([[state.v.v_long, state.v.v_tran, state.w.w_psi,
                        state.p.s, state.p.x_tran, state.p.e_psi,
                        state.x.x, state.x.y, state.e.psi, (state.p.s / self.track_obj.track_length + lap_no) / self.target_laps] for lap_no, state in zip(self.lap_no, self.sim_state)],
                      dtype=np.float32)
        ob = {
            "ego": ob.flatten(),
            "oppo": ob[::-1].flatten()
        }
        return ob

    def _is_success(self, vehicle_idx: int) -> bool:
        """Check if a vehicle has won by completing target laps"""
        return self.lap_no[vehicle_idx] >= self.target_laps

    def _is_failure(self, vehicle_idx: int) -> bool:
        """Check if a vehicle has failed"""
        state = self.sim_state[vehicle_idx]
        
        # Check for boundary violation
        if np.abs(state.p.x_tran) > self.track_obj.half_width:
            return True
        
        # Check for low speed or wrong direction
        if state.v.v_long < self.low_speed_threshold or np.abs(state.p.e_psi) > self.wrong_direction_threshold:
            return True
            
        return False

    def _is_draw(self) -> bool:
        """Check if the game is a draw"""
        return self._cached_results['draw']

    def _get_reward(self) -> dict:
        # Check for draw condition first (including collision)
        if self._is_draw():
            # logger.info(f"Draw! Ego average speed: {self._sum_eps_speed[0] / self.eps_len}, Oppo average speed: {self._sum_eps_speed[1] / self.eps_len}")
            return {"ego": 0.0, "oppo": 0.0}

        # Use cached results for success/failure
        ego_success = self._cached_results['success'][0]
        oppo_success = self._cached_results['success'][1]
        ego_failure = self._cached_results['failure'][0]
        oppo_failure = self._cached_results['failure'][1]

        # Terminal rewards for success/failure
        if ego_success or oppo_failure:
            # logger.info(f"Ego vehicle won! Ego average speed: {self._sum_eps_speed[0] / self.eps_len}, Oppo average speed: {self._sum_eps_speed[1] / self.eps_len}")
            return {"ego": 100.0, "oppo": -100.0}  # Ego wins
        elif oppo_success or ego_failure:
            # logger.info(f"Oppo vehicle won! Ego average speed: {self._sum_eps_speed[0] / self.eps_len}, Oppo average speed: {self._sum_eps_speed[1] / self.eps_len}")
            return {"ego": -100.0, "oppo": 100.0}  # Oppo wins

        # Progress-based rewards (zero-sum)
        k_progress = 1.0
        k_boundary = 0.2
        k_speed = 0.2

        # Calculate progress for each vehicle
        is_new_lap = self._is_new_lap()
        progress_ego = self.sim_state[0].p.s - self.last_state[0].p.s + self.track_obj.track_length if is_new_lap[0] else 0
        progress_oppo = self.sim_state[1].p.s - self.last_state[1].p.s + self.track_obj.track_length if is_new_lap[1] else 0

        # Boundary penalties
        boundary_penalty_ego = -k_boundary * max(0, np.abs(self.sim_state[0].p.x_tran) / self.track_obj.half_width - 0.9)
        boundary_penalty_oppo = -k_boundary * max(0, np.abs(self.sim_state[1].p.x_tran) / self.track_obj.half_width - 0.9)

        # Speed penalties
        speed_penalty_ego = -k_speed * max(0.0, 1.0 - self.sim_state[0].v.v_long)
        speed_penalty_oppo = -k_speed * max(0.0, 1.0 - self.sim_state[1].v.v_long)

        # Calculate relative progress (zero-sum)  Note: Not used for now. 
        relative_progress = progress_ego - progress_oppo

        return {
            # "ego": k_progress * progress_ego + boundary_penalty_ego + speed_penalty_ego,
            # "oppo": k_progress * progress_oppo + boundary_penalty_oppo + speed_penalty_oppo
            "ego": k_progress * relative_progress + boundary_penalty_ego + speed_penalty_ego,
            "oppo": -k_progress * relative_progress + boundary_penalty_oppo + speed_penalty_oppo
        }

    def _get_terminal(self) -> bool:
        """Episode is terminated if any vehicle has succeeded or failed"""
        # Check for draw condition
        if self._is_draw():
            return True
            
        # Check for individual success/failure using cached results
        if any(self._cached_results['success'][i] or self._cached_results['failure'][i] for i in range(2)):
            return True
            
        return False

    def _get_truncated(self) -> bool:
        """Episode is truncated if maximum time steps reached"""
        return self.eps_len > self.max_steps

    def _get_info(self) -> Dict[str, Dict[str, Union[List[VehicleState], int, float, bool, List[bool]]]]:
        return {
            'ego': {
                'vehicle_state': self.sim_state[0],
                'lap_no': self.lap_no[0],
                'avg_eps_speed': self._sum_eps_speed[0] / self.eps_len,
                'max_eps_speed': self.max_eps_speed[0],
                'min_eps_speed': self.min_eps_speed[0],
                'lap_time': self.eps_len * self.dt,
                'success': self._cached_results['success'][0],
                'failure': self._cached_results['failure'][0],
                'collision': self._cached_results['collision'],
            },
            'oppo': {
                'vehicle_state': self.sim_state[1],
                'lap_no': self.lap_no[1],
                'avg_eps_speed': self._sum_eps_speed[1] / self.eps_len,
                'max_eps_speed': self.max_eps_speed[1],
                'min_eps_speed': self.min_eps_speed[1],
                'lap_time': self.eps_len * self.dt,
                'success': self._cached_results['success'][1],
                'failure': self._cached_results['failure'][1],
                'collision': self._cached_results['collision'],
            },
        } 