from gymnasium.envs.registration import register

register(
    id='barc-v0',
    entry_point='gym_carla.envs.barc.barc_env:BarcEnv',
    # max_episode_steps=100000,
)
register(
    id='barc-v1',
    entry_point='gym_carla.envs.barc.multibarc_env:MultiBarcEnv',
    # max_episode_steps=100000,
)
register(
    id='barc-v1-race',
    entry_point='gym_carla.envs.barc.barc_race_env:BarcEnvRace',
)
register(
    id='barc-v2',
    entry_point='gym_carla.envs.barc.game_theoretic_env:GameTheoreticEnv',
)
register(
    id='barc-laps-v1',
    entry_point='gym_carla.envs.barc.racing_env:RacingEnv',
)
