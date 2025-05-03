#!/usr/bin/env python3
"""
game_theoretic_env.py

A MultiBarcEnv subclass where `step` returns a tuple of (ego_reward, opp_reward),
and the underlying `_get_reward` is overridden to support dual rewards.
"""
import copy
import numpy as np
from gym_carla.envs.barc.multibarc_env import MultiBarcEnv


class GameTheoreticEnv(MultiBarcEnv):
    def __init__(
        self,
        track_name: str,
        ego_action_grid: list,
        opp_action_grid: list,
        opp_reward_fn=None,
        sample_k: int = 16,
        **kwargs
    ):
        super().__init__(track_name=track_name, **kwargs)
        self.ego_grid    = ego_action_grid
        self.opp_grid    = opp_action_grid
        self.K           = sample_k

    def _get_reward(self, last_obs=None, ego_action=None, opp_action=None):
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

    def step(self, action_input):
        # support raw action or (action, probs) tuple
        if isinstance(action_input, tuple):
            ego_action, ego_probs = action_input
        else:
            ego_action = action_input
            ego_probs = np.ones(len(self.ego_grid)) / len(self.ego_grid)
        idxs = np.random.choice(len(self.ego_grid), size=self.K, p=ego_probs)
        saved = (
            copy.deepcopy(self.sim_state),
            copy.deepcopy(self.last_state),
            self.rel_dist,
            self.last_rel_dist,
            self.eps_len
        )

        best_a_opp, best_val = None, -np.inf
        for a_opp in self.opp_grid:
            total = 0.0
            for idx in idxs:
                a_e = self.ego_grid[idx]
                # apply candidate joint action
                for i, s in enumerate(self.sim_state):
                    s.u.u_a, s.u.u_steer = (a_e if i == 0 else a_opp)
                # simulate one step with error handling
                try:
                    for sim, st in zip(self.dynamics_simulator, self.sim_state):
                        sim.step(st, T=self.dt)
                        self.track_obj.global_to_local_typed(st)
                    self._update_relative_distance()
                    self.eps_len += 1

                    # compute opponent reward
                    # After roll-out, swap vehicles for opponent reward
                    self.sim_state[0], self.sim_state[1] = self.sim_state[1], self.sim_state[0]
                    self.last_state[0], self.last_state[1] = self.last_state[1], self.last_state[0]
                    self.rel_dist = -self.rel_dist
                    self.last_rel_dist = -self.last_rel_dist
                    opp_i = super()._get_reward()
                    # Swap back after computing reward
                    self.sim_state[0], self.sim_state[1] = self.sim_state[1], self.sim_state[0]
                    self.last_state[0], self.last_state[1] = self.last_state[1], self.last_state[0]
                    self.rel_dist = -self.rel_dist
                    self.last_rel_dist = -self.last_rel_dist
                except ValueError:
                    # Off-track or invalid state → heavy penalty
                    print(f"Invalid state: {self.sim_state}")
                    opp_i = -100.0

                total += opp_i
                # restore saved state
                (self.sim_state, self.last_state,
                 self.rel_dist, self.last_rel_dist,
                 self.eps_len) = saved
            avg = total / self.K
            if avg > best_val:
                best_val, best_a_opp = avg, a_opp

        (self.sim_state, self.last_state,
         self.rel_dist, self.last_rel_dist,
         self.eps_len) = saved

        joint = np.vstack([ego_action, best_a_opp])
        obs, ego_r, term, trunc, info = super().step(joint)
        ego_r2, opp_r = self._get_reward(last_obs=obs,
                                         ego_action=ego_action,
                                         opp_action=best_a_opp)
        info['opp_reward'] = opp_r
        return obs, (ego_r2, opp_r), term, trunc, info