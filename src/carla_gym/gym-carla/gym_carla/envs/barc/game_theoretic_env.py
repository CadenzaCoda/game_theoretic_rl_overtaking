#!/usr/bin/env python3
"""
game_theoretic_env.py

A MultiBarcEnv subclass where `step` returns a tuple of (ego_reward, opp_reward),
and the underlying `_get_reward` is overridden to support dual rewards.
"""
import copy
from typing import Tuple, Optional

import numpy as np
from gymnasium.core import ObsType
from networkx.algorithms.clique import enumerate_all_cliques
from gymnasium import spaces

from mpclab_simulation.dynamics_simulator import DynamicsSimulator
# import gym_carla
# from gym_carla.envs.barc.multibarc_env import MultiBarcEnv
from .multibarc_env import MultiBarcEnv
from mpclab_common.models.dynamics_models import CasadiDynamicBicycle
from gym_carla.controllers.barc_pid import PIDWrapper

from loguru import logger
import pdb

from mpclab_common.models.dynamics_models import get_dynamics_model
from gym_carla.controllers.barc_pid_ref_tracking import PIDRacelineFollowerWrapper


class GameTheoreticEnv(MultiBarcEnv):
    def __init__(
            self,
            track_name: str,
            opponent=None,
            opp_reward_fn=None,
            sample_k: int = 16,
            **kwargs
    ):
        super().__init__(track_name=track_name, **kwargs)
        logger.debug("GameTheoreticEnv init")
        # self.opp_grid    = self.get_action_grids(num_bins_per_dim=10)
        dynamics_config_approx = copy.deepcopy(self.sim_dynamics_config)
        dynamics_config_approx.dt = self.dt
        dynamics_config_approx.model_name = 'dynamic_bicycle_cl'
        self.dynamics = get_dynamics_model(t_start=self.t0, model_config=dynamics_config_approx, track=self.track_obj)
        self.opponent = PIDRacelineFollowerWrapper(t0=0., dt=0.1, track_obj=self.track_obj)
        self.K = sample_k

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
    ) -> Tuple[ObsType, dict]:
        obs, info = super().reset(seed=seed, options=options)
        self.opponent.reset(options={'vehicle_state': self.sim_state[1]})
        if (options is not None and options.get('render')) or self.do_render:
            self.opponent.raceline.plot_raceline(self.visualizer.ax_xy)
        return obs, info

    def _get_reward(self, last_obs=None, ego_action=None, opp_action=None) -> Tuple[float, float]:
        """
        Returns a tuple (ego_reward, opp_reward) by calling the base reward logic
        on vehicle 0 and vehicle 1 respectively using state swapping.
        """
        # Ego reward via original logic
        ego_r = super()._get_reward()

        # Save internal state to restore later
        saved = (
            copy.deepcopy(self.sim_state),
            copy.deepcopy(self.last_state),
            self.rel_dist,
            self.last_rel_dist
        )

        # Swap vehicle 0 and 1 states and flip relative distance
        self.sim_state[0], self.sim_state[1] = self.sim_state[1], self.sim_state[0]
        self.last_state[0], self.last_state[1] = self.last_state[1], self.last_state[0]
        self.rel_dist = -self.rel_dist
        self.last_rel_dist = -self.last_rel_dist

        # Opponent reward via same base logic on swapped state
        opp_r = super()._get_reward()

        # Restore original state
        (self.sim_state, self.last_state,
         self.rel_dist, self.last_rel_dist) = saved

        return ego_r, opp_r

    def step(self, action):
        # support raw action or (action, probs) tuple
        # if isinstance(action_dist, tuple):
        #     ego_action, ego_probs = action_dist
        # else:
        ego_action, ego_dist = action
        ego_action_samples = ego_dist.sample([self.K]).squeeze(1).cpu().numpy()
        if self.discrete:
            ego_action_samples = [self.decode_action(action) for action in ego_action_samples]
        # print(ego_action_samples)
        ego_next_states = self.dynamics.predict(copy.deepcopy(self.sim_state[0]),
                                                ego_action_samples)  # v_long, v_tran, w_psi, e_psi, s, x_tran
        ego_x_tran_median = np.median(ego_next_states[:, -1])
        ego_v_long_median = np.median(ego_next_states[:, 0])
        reaction_strength = np.exp(-abs(self.rel_dist - self.collision_threshold * 2))
        # logger.debug(f"ego_x_tran_median: {ego_x_tran_median}, ego_v_long_median: {ego_v_long_median}, reaction_strength: {reaction_strength}")
        # saved = (
        #     copy.deepcopy(self.sim_state),
        #     copy.deepcopy(self.last_state),
        #     self.rel_dist,
        #     self.last_rel_dist,
        #     self.eps_len
        # )
        a_opp, _ = self.opponent.step(self.sim_state[1], lap_no=self.lap_no[1],
                                      terminated=self._is_new_lap()[1],
                                      reference_modifier=(ego_v_long_median, ego_x_tran_median, reaction_strength))
        #
        # a_opp, best_opp_rew = None, -np.inf
        # # Try nominal opponent action.

        # u_a_eps_scale, u_steer_eps_scale = 0.1, 0.1
        # eps_u_a, eps_u_steer = (np.linspace(-self._action_bounds[0], self._action_bounds[0], 3, endpoint=True) * u_a_eps_scale,
        #                         np.linspace(-self._action_bounds[1], self._action_bounds[1], 3, endpoint=True) * u_steer_eps_scale)
        # eps_grid = np.array(np.meshgrid(eps_u_a, eps_u_steer, indexing='ij')).reshape(2, -1).T
        #
        # for a_e in ego_action_samples:
        #     # a_e = self.ego_grid[idx]
        #     # apply candidate joint action
        #     for i, s in enumerate(self.sim_state):
        #         s.u.u_a, s.u.u_steer = (a_e[0] if i == 0 else opponent_action)
        #     # simulate one step with error handling
        #     physical_distance = np.linalg.norm(
        #         np.array(self.sim_state[0].x.x, self.sim_state[0].x.y) - \
        #         np.array(self.sim_state[1].x.x, self.sim_state[1].x.y)
        #     )
        #     if physical_distance > self.collision_threshold * 2:
        #         # If the resulting distance is above a threshold, apply the nominal action directly.
        #         a_opp = opponent_action
        #     else:
        #         # Otherwise, search locally for an optimal opponent action.
        #         for eps_u_a, eps_u_steer in eps_grid:
        #             self.sim_state[1] = copy.deepcopy(saved[0][1])
        #             _a_opp = np.clip(np.array([eps_u_a, eps_u_steer]) + opponent_action, -self._action_bounds,
        #                              self._action_bounds)
        #             self.sim_state[1].u.u_a, self.sim_state[1].u.u_steer = _a_opp
        #             self.dynamics_simulator[1].step(self.sim_state[1], T=self.dt)
        #             opp_rew = self._get_reward()[1]
        #             if opp_rew > best_opp_rew:
        #                 best_opp_rew, a_opp = opp_rew, _a_opp

        # (self.sim_state, self.last_state,
        #  self.rel_dist, self.last_rel_dist,
        #  self.eps_len) = saved

        action = np.vstack([ego_action, a_opp])
        action = np.clip(action, -self._action_bounds, self._action_bounds)

        # Apply actions to each vehicle
        for i, _state in enumerate(self.sim_state):
            _state.u.u_a, _state.u.u_steer = action[i]

        self.last_state = copy.deepcopy(self.sim_state)
        self.last_rel_dist = self.rel_dist  # Store the previous relative distance
        self.render()

        terminated = False
        for i, _state in enumerate(self.sim_state):
            try:
                self.dynamics_simulator[i].step(_state, T=self.dt)
                self.track_obj.global_to_local_typed(_state)
            except ValueError as e:
                terminated = True

        # truncated = False
        # try:
        #     # Step each vehicle's dynamics
        #     for i, _state in enumerate(self.sim_state[1:]):
        #         self.dynamics_simulator[i].step(_state, T=self.dt)
        #         self.track_obj.global_to_local_typed(_state)
        # except ValueError as e:
        #     truncated = True  # The control action may drive the vehicle out of the track during the internal steps.

        # Update relative distance and lap counters
        self._update_relative_distance()

        self.t += self.dt
        self._update_speed_stats()

        obs = self._get_obs()
        rew = self._get_reward()
        terminated = terminated or self._get_terminal()
        truncated = self._get_truncated()
        info = self._get_info()

        if self._is_successful():
            logger.debug(
                f"Overtaking successful in {info['lap_time']:.1f} s. "
                f"avg_v = {info['avg_eps_speed']:.4f}, max_v = {info['max_eps_speed']:.4f}, "
                f"min_v = {info['min_eps_speed']:.4f}")

        # return obs, rew, terminated, truncated, info
        ego_r2, opp_r = rew
        info['opp_reward'] = opp_r
        return obs, (ego_r2, opp_r), terminated, truncated, info
