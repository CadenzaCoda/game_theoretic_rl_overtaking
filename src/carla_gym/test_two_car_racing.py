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
    controller_type = [PIDRacelineFollowerWrapper, MPCCConvWrapper][::-1]
    ego_controller = controller_type[0](dt=dt, t0=t0, track_obj=get_track(track_name))
    opponent_controller = controller_type[1](dt=dt, t0=t0, track_obj=get_track(track_name))

    env = gym.make('barc-v2',
                   # opponent=opponent_controller,
                   track_name=track_name,
                   t0=t0, dt=dt, dt_sim=dt_sim,
                   do_render=True,
                   enable_camera=False)
    
    # Create controllers for both vehicles
    # ego_controller = LMPCWrapper(dt=dt, t0=t0, track_obj=env.unwrapped.get_track())
    # controller_type = rng.choice([PIDWrapper, MPCCConvWrapper, LMPCWrapper], size=2, replace=False)

    # Bind the controllers to the environment
    env.unwrapped.bind_controller(ego_controller)
    
    # Reset the environment
    ob, info = env.reset(seed=seed, options={'spawning': 'fixed'})
    
    # Reset the controllers
    ego_controller.reset(seed=seed, options={'vehicle_state': info['vehicle_state'][0]})
    opponent_controller.reset(seed=seed, options={'vehicle_state': info['vehicle_state'][1]})
    
    # Initialize variables
    rew, terminated, truncated = None, False, False
    episode_count = 0
    success_count = 0
    
    # Main simulation loop
    while True:
        # Get actions from both controllers
        ego_action, _ = ego_controller.step(index=0, **info)
        ego_dist = Normal(loc=torch.from_numpy(ego_action), scale=torch.ones(2) * 0.1)
        
        # For the opponent, we need to create a modified observation and info
        # that only contains the opponent's state
        # opponent_ob = {
        #     'gps': np.array([info['vehicle_state'][1].x.x,
        #                      info['vehicle_state'][1].x.y,
        #                      info['vehicle_state'][1].e.psi], dtype=np.float32),
        #     'velocity': np.array([info['vehicle_state'][1].v.v_long,
        #                          info['vehicle_state'][1].v.v_tran,
        #                          info['vehicle_state'][1].w.w_psi], dtype=np.float32),
        #     'state': np.array([info['vehicle_state'][1].v.v_long,
        #                       info['vehicle_state'][1].v.v_tran,
        #                       info['vehicle_state'][1].w.w_psi,
        #                       info['vehicle_state'][1].p.s,
        #                       info['vehicle_state'][1].p.x_tran,
        #                       info['vehicle_state'][1].p.e_psi,
        #                       info['vehicle_state'][1].x.x,
        #                       info['vehicle_state'][1].x.y,
        #                       info['vehicle_state'][1].e.psi], dtype=np.float32),
        # }
        #
        # opponent_info = {
        #     'vehicle_state': info['vehicle_state'][1],
        #     'lap_no': info['lap_no'][1],
        #     'terminated': info['terminated'][1],
        #     'avg_lap_speed': info['avg_eps_speed'],
        #     'max_lap_speed': info['max_eps_speed'],
        #     'min_lap_speed': info['min_eps_speed'],
        #     'lap_time': info['lap_time'],
        # }
        #
        # opponent_action, _ = opponent_controller.step(**opponent_ob, **opponent_info)
        #
        # # Combine actions for both vehicles
        # combined_action = np.vstack([ego_action, opponent_action])
        
        # Step the environment
        ob, rew, terminated, truncated, info = env.step((ego_action, ego_dist))
        
        # Log episode results
        if info['success']:
            success_count += 1
            logger.info(f"Successful overtaking! Episode {episode_count}")
            logger.info(f"Success rate: {success_count}/{episode_count+1}")
            
        elif terminated or truncated:
            episode_count += 1
            logger.info(f"Episode {episode_count} truncated.")
            logger.info(f"Success rate: {success_count}/{episode_count}")
            
            # Reset the environment and controllers
            # ob, info = env.reset(seed=seed, options={'spawning': 'fixed'})
            ob, info = env.reset()
            # controller_type = [MPCCConvWrapper, LMPCWrapper]
            # controller_type = rng.choice([PIDWrapper, MPCCConvWrapper, LMPCWrapper], size=2, replace=False)
            # ego_controller = controller_type[0](dt=dt, t0=t0, track_obj=env.unwrapped.get_track())
            # env.unwrapped.bind_controller(ego_controller)
            # opponent_controller = controller_type[1](dt=dt, t0=t0, track_obj=env.unwrapped.get_track())

            ego_controller.reset(seed=seed, options={'vehicle_state': info['vehicle_state'][0]})
            opponent_controller.reset(seed=seed, options={'vehicle_state': info['vehicle_state'][1]})
            
            # Add a small delay between episodes
            time.sleep(1)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    params = vars(parser.parse_args())

    main(**params) 
