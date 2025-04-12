import gymnasium as gym
import numpy as np
import time

import gym_carla

from controllers.barc_lmpc import LMPCWrapper
from controllers.barc_pid import PIDWrapper
from controllers.barc_mpcc_conv import MPCCConvWrapper
from loguru import logger


def main(seed=0):
    """
    Test script for the two-car racing environment.
    Uses LMPCWrapper for the ego vehicle and PIDWrapper for the opponent.
    """
    dt = 0.1
    dt_sim = 0.01
    t0 = 0
    rng = np.random.default_rng()
    
    # Create the two-car racing environment
    env = gym.make('barc-v1', track_name='L_track_barc',
                   t0=t0, dt=dt, dt_sim=dt_sim,
                   do_render=True,
                   max_n_laps=10,
                   enable_camera=False)
    
    # Create controllers for both vehicles
    # ego_controller = LMPCWrapper(dt=dt, t0=t0, track_obj=env.unwrapped.get_track())
    controller_type = rng.choice([PIDWrapper, MPCCConvWrapper, LMPCWrapper], size=2, replace=False)
    ego_controller = controller_type[0](dt=dt, t0=t0, track_obj=env.unwrapped.get_track())
    opponent_controller = controller_type[1](dt=dt, t0=t0, track_obj=env.unwrapped.get_track())
    
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
        ego_action, _ = ego_controller.step(index=0, **ob, **info)
        
        # For the opponent, we need to create a modified observation and info
        # that only contains the opponent's state
        opponent_ob = {
            'gps': np.array([info['vehicle_state'][1].x.x, 
                             info['vehicle_state'][1].x.y, 
                             info['vehicle_state'][1].e.psi], dtype=np.float32),
            'velocity': np.array([info['vehicle_state'][1].v.v_long, 
                                 info['vehicle_state'][1].v.v_tran, 
                                 info['vehicle_state'][1].w.w_psi], dtype=np.float32),
            'state': np.array([info['vehicle_state'][1].v.v_long, 
                              info['vehicle_state'][1].v.v_tran, 
                              info['vehicle_state'][1].w.w_psi,
                              info['vehicle_state'][1].p.s, 
                              info['vehicle_state'][1].p.x_tran, 
                              info['vehicle_state'][1].p.e_psi,
                              info['vehicle_state'][1].x.x, 
                              info['vehicle_state'][1].x.y, 
                              info['vehicle_state'][1].e.psi], dtype=np.float32),
        }
        
        opponent_info = {
            'vehicle_state': info['vehicle_state'][1],
            'lap_no': info['lap_no'][1],
            'terminated': info['terminated'][1],
            'avg_lap_speed': info['avg_lap_speed'],
            'max_lap_speed': info['max_lap_speed'],
            'min_lap_speed': info['min_lap_speed'],
            'lap_time': info['lap_time'],
        }
        
        opponent_action, _ = opponent_controller.step(**opponent_ob, **opponent_info)
        
        # Combine actions for both vehicles
        combined_action = np.vstack([ego_action, opponent_action])
        
        # Step the environment
        ob, rew, terminated, truncated, info = env.step(combined_action)
        
        # Log episode results
        if terminated:
            success_count += 1
            logger.info(f"Successful overtaking! Episode {episode_count}")
            logger.info(f"Success rate: {success_count}/{episode_count+1}")
            
        if truncated:
            episode_count += 1
            logger.info(f"Episode {episode_count} truncated.")
            logger.info(f"Success rate: {success_count}/{episode_count}")
            
            # Reset the environment and controllers
            ob, info = env.reset(seed=seed, options={'spawning': 'fixed'})
            controller_type = rng.choice([PIDWrapper, MPCCConvWrapper, LMPCWrapper], size=2, replace=False)
            ego_controller = controller_type[0](dt=dt, t0=t0, track_obj=env.unwrapped.get_track())
            opponent_controller = controller_type[1](dt=dt, t0=t0, track_obj=env.unwrapped.get_track())

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
