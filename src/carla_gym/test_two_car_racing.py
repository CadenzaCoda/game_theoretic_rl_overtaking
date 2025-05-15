import gymnasium as gym
import numpy as np
import time

import torch

from gym_carla.controllers.barc_lmpc import LMPCWrapper
from gym_carla.controllers.barc_mpcc_conv import MPCCConvWrapper
from mpclab_common.track import get_track
from loguru import logger

from gym_carla.controllers.barc_pid import PIDWrapper
from gym_carla.controllers.barc_pid_ref_tracking import PIDRacelineFollowerWrapper
from torch.distributions import Normal


def main(seed=0):
    """
    Test script for the two-car racing environment.
    Uses LMPCWrapper for the ego vehicle and PIDWrapper for the opponent.
    """
    dt = 0.1
    dt_sim = 0.01
    t0 = 0
    rng = np.random.default_rng()
    track_name = 'L_track_barc'

    # Create the two-car racing environment
    controller_type = [PIDRacelineFollowerWrapper, PIDRacelineFollowerWrapper]
    ego_controller = controller_type[0](dt=dt, t0=t0, track_obj=get_track(track_name))
    opponent_controller = controller_type[1](dt=dt, t0=t0, track_obj=get_track(track_name))

    env = gym.make('barc-v1',
                   # opponent=opponent_controller,
                   track_name=track_name,
                   t0=t0, dt=dt, dt_sim=dt_sim,
                   do_render=True,
                   enable_camera=False,
                   discrete_action=False)

    # Bind the controllers to the environment
    env.unwrapped.bind_controller(ego_controller)

    # Reset the environment
    ob, info = env.reset(seed=seed, options={'spawning': 'fixed'})

    # Reset the controllers
    ego_controller.reset(seed=seed, options={'vehicle_state': info['ego']['vehicle_state']})
    opponent_controller.reset(seed=seed, options={'vehicle_state': info['ego']['vehicle_state']})

    # Initialize variables
    rew, terminated, truncated = None, False, False
    episode_count = 0
    success_count = 0

    # Main simulation loop
    while True:
        # Get actions from both controllers
        # Note: Your step function can take anything that the environment outputs, including the entire info dictionary and the observation vector.
        # See the details in multibarc_env.py.
        ego_action, _ = ego_controller.step(vehicle_state=info['ego']['vehicle_state'], terminated=info['ego']['terminated'],
                                            lap_no=info['ego']['lap_no'])
        oppo_action, _ = opponent_controller.step(vehicle_state=info['oppo']['vehicle_state'], terminated=info['oppo']['terminated'],
                                             lap_no=info['oppo']['lap_no'])
        # Step the environment
        ob, rew, terminated, truncated, info = env.step({'ego': ego_action, 'oppo': oppo_action})

        # Log episode results
        if terminated['__all__'] or truncated['__all__']:
            episode_count += 1
            ob, info = env.reset()
            ego_controller.reset(seed=seed, options={'vehicle_state': info['ego']['vehicle_state']})
            opponent_controller.reset(seed=seed, options={'vehicle_state': info['oppo']['vehicle_state']})

            # Add a small delay between episodes
            time.sleep(1)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    params = vars(parser.parse_args())

    main(**params)
