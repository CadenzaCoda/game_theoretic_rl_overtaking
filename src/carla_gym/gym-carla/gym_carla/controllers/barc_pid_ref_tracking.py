#!/usr/bin/env python3
from os import PathLike
from pathlib import Path

from loguru import logger
from mpclab_controllers.PID import AbstractController, PID
from mpclab_controllers.utils.controllerTypes import PIDParams

from mpclab_common.models.dynamics_models import CasadiDynamicCLBicycle
from mpclab_common.models.model_types import DynamicBicycleConfig
from mpclab_common.pytypes import VehicleState, VehicleActuation, VehiclePrediction, Position, ParametricPose, \
    BodyLinearVelocity, OrientationEuler, BodyAngularVelocity
from mpclab_common.track import get_track

import pdb
import os
import numpy as np


class Raceline:
    def __init__(self, raceline_file: PathLike = 'raceline.npz'):
        while not os.path.exists(raceline_file):
            raise FileNotFoundError(f"Raceline data file {raceline_file} is not found. Please upload it here. ")
        data = np.load(raceline_file)
        self.s, self.ey, self.epsi, self.vx, self.vy, self.epsi_dot, self.t, self.u0, self.u1 = map(np.array,
                                                                                                    (data['s'],
                                                                                                     data['e_y'],
                                                                                                     data['e_psi'],
                                                                                                     data['v_long'],
                                                                                                     data['v_tran'],
                                                                                                     data['psidot'],
                                                                                                     data['t'],
                                                                                                     data['u_a'],
                                                                                                     data['u_s']))

    def get_reference(self, _s, speed_scaling=1.):
        # interp = lambda yp: np.interp(_s, self.s, yp)
        # ey_ref, epsi_ref, vx_ref, vy_ref, epsi_dot_ref, t_ref, u0_ref, u1_ref = map(interp, (self.ey, self.epsi, self.vx, self.vy, self.epsi_dot, self.t, self.u0, self.u1))

        ey_ref = np.interp(_s, self.s, self.ey)
        epsi_ref = np.interp(_s, self.s, self.epsi)
        vx_ref = np.interp(_s, self.s,
                           self.vx) * speed_scaling  # TODO: This is just a trick to scale down the speed to make the raceline easier to track. PROPERLY IMPLEMENT THIS!
        vy_ref = np.interp(_s, self.s, self.vy)
        epsi_dot_ref = np.interp(_s, self.s, self.epsi_dot)
        t_ref = np.interp(_s, self.s, self.t)
        u0_ref = np.interp(_s, self.s, self.u0)
        u1_ref = np.interp(_s, self.s, self.u1)
        return {
            's': _s,
            'x0': vx_ref,
            'x1': vy_ref,
            'x2': epsi_dot_ref,
            'x3': epsi_ref,
            'x4': t_ref,
            'x5': ey_ref,
            'u0': u0_ref,
            'u1': u1_ref,
        }

    def plot_raceline(self, ax):
        svec = self.s
        Psi = self.epsi[0]
        X, Y = [0], [self.ey[0]]
        for j in range(1, len(svec)):
            sj = svec[j]
            deltaT = self.t[j] - self.t[j - 1]
            Psi = Psi + deltaT * self.epsi_dot[j]
            X.append(X[j - 1] + deltaT * (self.vx[j] * np.cos(Psi) - self.vy[j] * np.sin(Psi)))
            Y.append(Y[j - 1] + deltaT * (self.vx[j] * np.sin(Psi) + self.vy[j] * np.cos(Psi)))
        ax.plot(X, Y, c='r', ls='--', label='raceline')


class PIDRacelineFollower(AbstractController):
    '''
    Class for PID throttle and steering control of a vehicle
    Incorporates separate PID controllers for maintaining a constant speed and a constant lane offset

    target speed: v_ref
    target lane offset_ x_ref
    '''

    def __init__(self, dt: float,
                 steer_pid_params: PIDParams = None,
                 speed_pid_params: PIDParams = None,
                 speed_scaling: float = 1.,
                 raceline: 'Raceline' = None,
                 ):
        if steer_pid_params is None:
            steer_pid_params = PIDParams()
            steer_pid_params.dt = dt
            steer_pid_params.default_steer_params()
        if speed_pid_params is None:
            speed_pid_params = PIDParams()
            speed_pid_params.dt = dt
            speed_pid_params.default_speed_params()  # these may use dt so it is updated first

        self.dt = dt
        steer_pid_params.dt = dt
        speed_pid_params.dt = dt

        self.steer_pid_params = steer_pid_params
        self.speed_pid_params = speed_pid_params

        self.steer_pid_params.x_ref = 0
        self.speed_pid_params.x_ref = 0

        self.steer_pid = PID(self.steer_pid_params)
        self.speed_pid = PID(self.speed_pid_params)

        self.lat_ref = steer_pid_params.x_ref
        self.raceline = raceline or Raceline(raceline_file=Path(__file__).resolve().parent / 'data' / 'raceline.npz')
        self.speed_scaling = speed_scaling
        self.requires_env_state = False
        return

    def reset(self):
        # Reinstantiate the two PID controllers.
        self.steer_pid = PID(self.steer_pid_params)
        self.speed_pid = PID(self.speed_pid_params)

    def initialize(self, **args):
        return

    def solve(self, *args, **kwargs):
        raise NotImplementedError

    def step(self, vehicle_state: VehicleState, reference_modifier=(0., 0., 0.)):
        reference = self.raceline.get_reference(vehicle_state.p.s, speed_scaling=self.speed_scaling)
        _v_ref, _x_tran_ref, psi_ref = reference['x0'], reference['x5'], reference['x3']

        v_ext, x_tran_ext, strength = reference_modifier
        strength *= np.exp(-np.abs(x_tran_ext - _x_tran_ref))
        v_ref = strength * v_ext + (1 - strength) * _v_ref
        x_tran_ref = strength * x_tran_ext + (1 - strength) * _x_tran_ref

        # logger.debug(f"strength: {strength}")
        # logger.debug(f"v_ref: {_v_ref}, v_ref_modified: {v_ref}, x_tran_ref: {_x_tran_ref}, x_tran_ref_modified: {x_tran_ref}, psi_ref: {psi_ref} ")

        v = vehicle_state.v.v_long
        vehicle_state.u.u_a, _ = self.speed_pid.solve(v - v_ref)

        # Weighting factor: alpha*x_trans + beta*psi_diff
        alpha = 5.0
        beta = 1.0
        vehicle_state.u.u_steer, _ = self.steer_pid.solve(
            alpha * (vehicle_state.p.x_tran - x_tran_ref) + beta * (vehicle_state.p.e_psi - psi_ref))
        return np.array([vehicle_state.u.u_a, vehicle_state.u.u_steer])

    def get_prediction(self):
        return None

    def get_safe_set(self):
        return None


class PIDRacelineFollowerWrapper:
    def __init__(self, dt=0.1, t0=0., track_obj=None, noise=False, VL=0.37, VW=0.195):
        # Input type: ndarray, local frame
        self.pid_controller = None
        self.t = None

        raceline_file = Path(__file__).resolve().parent / 'data' / 'raceline.npz'
        self.raceline = Raceline(raceline_file=raceline_file)

        self.dt = dt
        self.noise = noise
        self.track_obj = track_obj
        self.t0 = t0
        dynamics_config = DynamicBicycleConfig(dt=dt,
                                               model_name='dynamic_bicycle_cl',
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
                                               pacejka_c_rear=2.28,
                                               code_gen=False,
                                               jit=True,
                                               opt_flag='O3')
        self.dyn_model = CasadiDynamicCLBicycle(t0, dynamics_config, track=track_obj)
        self.state_input_ub = VehicleState(
            p=ParametricPose(s=2 * self.track_obj.track_length, x_tran=self.track_obj.half_width - VW / 2, e_psi=100),
            v=BodyLinearVelocity(v_long=10, v_tran=10),
            w=BodyAngularVelocity(w_psi=10),
            u=VehicleActuation(u_a=2.0, u_steer=0.45))
        self.state_input_lb = VehicleState(
            p=ParametricPose(s=-2 * self.track_obj.track_length, x_tran=-(self.track_obj.half_width - VW / 2),
                             e_psi=-100),
            v=BodyLinearVelocity(v_long=-10, v_tran=-10),
            w=BodyAngularVelocity(w_psi=-10),
            u=VehicleActuation(u_a=-2.0, u_steer=-0.45))
        self.input_rate_ub = VehicleState(u=VehicleActuation(u_a=20.0, u_steer=4.5))
        self.input_rate_lb = VehicleState(u=VehicleActuation(u_a=-20.0, u_steer=-4.5))
        self.lmpc_params = dict(
            N=15,
            n_ss_pts=48,
            n_ss_its=4,
        )
        self.prediction = VehiclePrediction()
        self.safe_set = VehiclePrediction()

    def setup_pid_controller(self):
        pid_steer_params = PIDParams(dt=self.dt,
                                     Kp=0.25,
                                     Ki=0.1,
                                     Kd=0.05,
                                     u_max=self.state_input_ub.u.u_steer,
                                     u_min=self.state_input_lb.u.u_steer,
                                     du_max=self.input_rate_ub.u.u_steer,
                                     du_min=self.input_rate_lb.u.u_steer,
                                     x_ref=0.0,
                                     noise=False,
                                     noise_max=0.2,
                                     noise_min=-0.2)
        pid_speed_params = PIDParams(dt=self.dt,
                                     Kp=0.5,
                                     Ki=0.05,
                                     Kd=0.05,
                                     u_max=self.state_input_ub.u.u_a,
                                     u_min=self.state_input_lb.u.u_a,
                                     du_max=self.input_rate_ub.u.u_a,
                                     du_min=self.input_rate_lb.u.u_a,
                                     x_ref=1.0,
                                     noise=False,
                                     noise_max=0.9,
                                     noise_min=-0.9)
        self.pid_controller = PIDRacelineFollower(self.dt, pid_steer_params, pid_speed_params,
                                                  raceline=self.raceline,
                                                  speed_scaling=0.7)

    def _step_pid(self, _state: VehicleState, reference_modifier=None):
        self.pid_controller.step(_state, reference_modifier=reference_modifier)
        self.track_obj.global_to_local_typed(_state)  # Recall that PID uses the global frame.
        _state.p.s = np.mod(_state.p.s, self.track_obj.track_length)
        return {'success': True, 'status': 0}  # PID controller never fails.

    def reset(self, *, seed=None, options=None):
        self.setup_pid_controller()
        self.t = self.t0

    def step(self, vehicle_state, terminated, lap_no, index=None, reference_modifier=None, **kwargs):
        """
        Use VehicleState to step directly. Closer to how the simulation script works.
        """
        if index is not None:
            vehicle_state = vehicle_state[index]
            lap_no = lap_no[index]
            terminated = terminated[index]
        info = self._step_pid(vehicle_state, reference_modifier)
        return np.array([vehicle_state.u.u_a, vehicle_state.u.u_steer]), info

    def get_prediction(self):
        return None

    def get_safe_set(self):
        return None
