import numpy as np
import pytest
import gymnasium as gym
import gym_carla
from mpclab_common.track import get_track
from gym_carla.controllers.barc_pid import PIDWrapper


@pytest.fixture
def barc_env_with_discrete_action():
    track_name = "L_track_barc"
    opponent = PIDWrapper(dt=0.1, t0=0., track_obj=get_track(track_name))
    return gym.make('barc-v1-race',
                    opponent=opponent,
                    track_name=track_name,
                    discrete_action=True
                    )


@pytest.fixture
def barc_env_with_continuous_action():
    track_name = "L_track_barc"
    opponent = PIDWrapper(dt=0.1, t0=0., track_obj=get_track(track_name))
    return gym.make('barc-v1-race',
                    opponent=opponent,
                    track_name=track_name,
                    discrete_action=False
                    )


def test_discrete_action_space_dimension(barc_env_with_discrete_action):
    env = barc_env_with_discrete_action
    env.reset()
    assert (env.action_space.nvec == np.array([32, 32])).all(), "Discrete action space dimension is incorrect"


def test_decode_action_discrete(barc_env_with_discrete_action):
    env = barc_env_with_discrete_action
    env.reset()
    # env.unwrapped.u_a_space = [0.0, 0.5, 1.0]
    # env.unwrapped.u_steer_space = [-1.0, 0.0, 1.0]
    u_a_space = np.linspace(-2, 2, 32, endpoint=True, dtype=np.float32)
    u_steer_space = np.linspace(-0.45, 0.45, 32, endpoint=True, dtype=np.float32)
    action = [3, 9]
    result = env.unwrapped.decode_action(action)
    expected = np.array([u_a_space[action[0]], u_steer_space[action[1]]])
    assert np.array_equal(result, expected), "decode_action failed for discrete action"


def test_decode_action_continuous(barc_env_with_continuous_action):
    env = barc_env_with_continuous_action
    env.reset()
    with pytest.raises(ValueError):
        env.unwrapped.decode_action([2, 1])
