#!/usr/bin/env python3

from mpclab_controllers.CA_LTV_MPC import CA_LTV_MPC
from mpclab_controllers.utils.controllerTypes import CALTVMPCParams

from mpclab_simulation.dynamics_simulator import DynamicsSimulator

from mpclab_common.pytypes import VehicleState, VehicleActuation, VehiclePrediction, Position, ParametricPose, BodyLinearVelocity, OrientationEuler, BodyAngularVelocity
from mpclab_common.models.dynamics_models import CasadiDynamicCLBicycle, CasadiKinematicUnicycle
from mpclab_common.models.model_types import DynamicBicycleConfig
from mpclab_common.track import get_track

import pdb

import numpy as np
import casadi as ca

import copy
import time

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.transforms import Affine2D

save_fig = False
spline_approximation = False

# Initial time
t = 0

# Import scenario
# track_obj = get_track('L_track_barc')
# if spline_approximation:
#     from mpclab_common.tracks.casadi_bspline_track import CasadiBSplineTrack
#     # Compute spline approximation of track
#     S = np.linspace(0, track_obj.track_length-1e-3, 100)
#     X, Y = [], []
#     for s in S:
#         # Centerline
#         x, y, _ = track_obj.local_to_global((s, 0, 0))
#         X.append(x)
#         Y.append(y)
#     xy = np.array([X, Y]).T
#     track_obj = CasadiBSplineTrack(xy, track_obj.track_width, track_obj.slack, s_waypoints=S)

# sim_state = VehicleState(t=0.0, 
#                         p=ParametricPose(s=0.1, x_tran=0.0, e_psi=0), 
#                         v=BodyLinearVelocity(v_long=0.5))
# track_obj.local_to_global_typed(sim_state)

# L = track_obj.track_length
# H = track_obj.half_width

# =============================================
# Set up model
# =============================================
class LTVMPCWrapper:
    def __init__(self, dt, t0, track_obj):
        self.dt = dt
        self.t0 = t0
        self.track_obj = track_obj
        L = self.L = self.track_obj.track_length
        H = self.H = self.track_obj.half_width

        discretization_method = 'rk4'
        dynamics_config = DynamicBicycleConfig(dt=dt,
                                                model_name='kinematic_unicycle',
                                                discretization_method=discretization_method,
                                                mass=2.2187,
                                                )
        if spline_approximation:
            dynamics_config.use_mx = True
        # self.dyn_model = CasadiDynamicCLBicycle(t, dynamics_config, track=track_obj)
        self.dyn_model = CasadiKinematicUnicycle(t, dynamics_config, track=track_obj)

        state_input_ub = VehicleState(p=ParametricPose(s=2*L, x_tran=H, e_psi=100),
                                    v=BodyLinearVelocity(v_long=10, v_tran=10),
                                    w=BodyAngularVelocity(w_psi=10),
                                    u=VehicleActuation(u_a=2.0, u_steer=0.45))
        state_input_lb = VehicleState(p=ParametricPose(s=-2*L, x_tran=-H, e_psi=-100),
                                    v=BodyLinearVelocity(v_long=-10, v_tran=-10),
                                    w=BodyAngularVelocity(w_psi=-10),
                                    u=VehicleActuation(u_a=-2.0, u_steer=-0.45))
        input_rate_ub = VehicleState(u=VehicleActuation(u_a=20.0, u_steer=4.5))
        input_rate_lb = VehicleState(u=VehicleActuation(u_a=-20.0, u_steer=-4.5))

        D = (5+1)*self.L
        self.v_ref = 2.0
        self.s_ref = np.linspace(0, D, int(D/(self.v_ref*self.dt)))
        # ey_ref = 0.25*np.sin(2*np.pi*6*s_ref/(L-1))
        self.ey_ref = np.zeros((self.s_ref.shape[0], ))

        x_ref, y_ref = [], []
        for s, ey in zip(self.s_ref, self.ey_ref):
            x, y, _ = track_obj.local_to_global((s, ey, 0))
            x_ref.append(x)
            y_ref.append(y)

        # =============================================
        # MPC controller setup
        # =============================================
        N = self.N = 15
        mpc_params = CALTVMPCParams(N=N,
                                    state_scaling=[2, 7, 2*L, 1.5*H],
                                    input_scaling=[2, 0.45],
                                    # state_scaling=None,
                                    # input_scaling=None,
                                    # soft_state_bound_idxs=[3],
                                    soft_state_bound_quad=[5],
                                    soft_state_bound_lin=[25],
                                    wrapped_state_idxs=[2],
                                    wrapped_state_periods=[L],
                                    damping=0.75,
                                    qp_iters=2,
                                    # delay=[1, 1],
                                    delay=None,
                                    verbose=True,
                                    qp_interface='casadi')

        # Symbolic placeholder variables
        sym_q = ca.MX.sym('q', self.dyn_model.n_q)
        sym_u = ca.MX.sym('u', self.dyn_model.n_u)
        sym_du = ca.MX.sym('du', self.dyn_model.n_u)

        sym_q_ref = ca.MX.sym('q_ref', self.dyn_model.n_q)
        s_idx = 2
        ey_idx = 3

        ua_idx = 0
        us_idx = 1

        Q = np.diag([1, 0, 10, 10])  # (v_long, e_psi, s, x_tran)
        sym_state_stage = 0.5*ca.bilin(Q, sym_q-sym_q_ref, sym_q-sym_q_ref)
        sym_state_term = 0.5*ca.bilin(Q, sym_q-sym_q_ref, sym_q-sym_q_ref)

        sym_input_stage = 0.5*(1e-4*(sym_u[ua_idx])**2 + 1e-4*(sym_u[us_idx])**2)
        sym_input_term = 0.5*(1e-4*(sym_u[ua_idx])**2 + 1e-4*(sym_u[us_idx])**2)

        sym_rate_stage = 0.5*(0.01*(sym_du[ua_idx])**2 + 0.01*(sym_du[us_idx])**2)

        sym_costs = {'state': [None for _ in range(N+1)], 'input': [None for _ in range(N+1)], 'rate': [None for _ in range(N)]}
        for k in range(N):
            sym_costs['state'][k] = ca.Function(f'state_stage_{k}', [sym_q, sym_q_ref], [sym_state_stage])
            sym_costs['input'][k] = ca.Function(f'input_stage_{k}', [sym_u], [sym_input_stage])
            sym_costs['rate'][k] = ca.Function(f'rate_stage_{k}', [sym_du], [sym_rate_stage])
        sym_costs['state'][N] = ca.Function('state_term', [sym_q, sym_q_ref], [sym_state_term])
        sym_costs['input'][N] = ca.Function('input_term', [sym_u], [sym_input_term])

        # a_max = dynamics_config.gravity*dynamics_config.wheel_friction
        # sym_ax, sym_ay, _ = dyn_model.f_a(sym_q, sym_u)
        # friction_circle_constraint = ca.Function('friction_circle', [sym_q, sym_u], [sym_ax**2 + sym_ay**2 - a_max**2])

        sym_constrs = {
            # 'state_input': [friction_circle_constraint for _ in range(N+1)], 
            'state_input': [None for _ in range(N + 1)],
            'rate': [None for _ in range(N)]
            }

        self.mpc_controller = CA_LTV_MPC(self.dyn_model, 
                                        sym_costs, 
                                        sym_constrs, 
                                        {'qu_ub': state_input_ub, 'qu_lb': state_input_lb, 'du_ub': input_rate_ub, 'du_lb': input_rate_lb},
                                        mpc_params)
        
    def reset(self, seed=None, options=None):
        # Define state reference
        if options.get('vehicle_state'):
            sim_state = options['vehicle_state']
        else:
            sim_state = VehicleState()

        ref_start = np.argmin(np.abs(self.s_ref - sim_state.p.s))
        q_ref = []
        for i in range(ref_start, ref_start + self.N+1):
            q_ref.append(np.array([self.v_ref, 0, self.s_ref[i], self.ey_ref[i]]))

        u_ws = np.zeros((self.N+1, self.dyn_model.n_u))
        du_ws = np.zeros((self.N, self.dyn_model.n_u))
        self.mpc_controller.set_warm_start(u_ws, du_ws, state=sim_state, parameters=np.concatenate(q_ref))

    def step(self, vehicle_state: VehicleState, **kwargs):
        # Get reference
        ref_start = np.argmin(np.abs(self.s_ref - vehicle_state.p.s))
        q_ref = []
        for i in range(ref_start, ref_start + self.N + 1):
            q_ref.append(np.array([self.v_ref, 0, self.s_ref[i], self.ey_ref[i]]))

        # Solve for car 1 control
        # st = time.time()
        self.mpc_controller.step(vehicle_state, parameters=np.concatenate(q_ref))
        return np.array([vehicle_state.u.u_a, vehicle_state.u.u_steer]), {}
    
    def get_prediction(self):
        return self.mpc_controller.get_prediction()
    
    def get_safe_set(self):
        return None

    

# # =============================================
# # Create visualizer
# # =============================================
# # Set up plotter
# vehicle_draw_L=0.37
# vehicle_draw_W=0.195

# plot = True
# if plot:
#     plt.ion()
#     fig = plt.figure()
#     ax_xy = fig.add_subplot(1,2,1)
#     ax_a = fig.add_subplot(2,2,2)
#     ax_d = fig.add_subplot(2,2,4)
#     track_obj.plot_map(ax_xy)
#     ax_xy.plot(x_ref, y_ref, 'r')
#     ax_xy.set_aspect('equal')
#     l_pred = ax_xy.plot([], [], 'b-o', markersize=4)[0]
#     l_ref = ax_xy.plot([], [], 'ko', markersize=4)[0]
#     l_a = ax_a.plot([], [], '-bo')[0]
#     l_d = ax_d.plot([], [], '-bo')[0]
#     VL = vehicle_draw_L
#     VW = vehicle_draw_W
#     rect = patches.Rectangle((-0.5*VL, -0.5*VW), VL, VW, linestyle='solid', color='b', alpha=0.5)
#     ax_xy.add_patch(rect)
#     fig.canvas.draw()
#     fig.canvas.flush_events()

# # =============================================
# # Run race
# # =============================================
# # Initialize inputs
# t = 0.0
# sim_state.u.u_a, sim_state.u.u_steer = 0.0, 0.0
# pred = VehiclePrediction()
# control = VehicleActuation(t=t, u_a=0, u_steer=0)

# while True:
#     state = copy.deepcopy(sim_state)

#     # Get reference
#     ref_start = np.argmin(np.abs(s_ref - state.p.s))
#     q_ref = []
#     for i in range(ref_start, ref_start+N+1):
#         q_ref.append(np.array([v_ref, 0, 0, 0, s_ref[i], ey_ref[i]]))

#     # Solve for car 1 control
#     st = time.time()
#     mpc_controller.step(state, parameters=np.concatenate(q_ref))
#     print('Controller solve time: ' + str(time.time()-st))
#     state.copy_control(control)
#     pred = mpc_controller.get_prediction()
#     pred.t = t

#     # Apply control action and advance simulation
#     # last_state = copy.deepcopy(sim_state)

#     if plot:
#         x, y, psi = track_obj.local_to_global((state.p.s, state.p.x_tran, state.p.e_psi))
#         b_left = x - VL/2
#         b_bot  = y - VW/2
#         r = Affine2D().rotate_around(x, y, psi) + ax_xy.transData
#         rect.set_xy((b_left,b_bot))
#         rect.set_transform(r)
#         pred_x, pred_y, ref_x, ref_y = [], [], [], []
#         for i in range(len(pred.s)):
#             x, y, psi = track_obj.local_to_global((pred.s[i], pred.x_tran[i], pred.e_psi[i]))
#             pred_x.append(x)
#             pred_y.append(y)
#         for i in range(len(q_ref)):
#             x, y, _ = track_obj.local_to_global((q_ref[i][4], q_ref[i][5], 0))
#             ref_x.append(x)
#             ref_y.append(y)
#         l_pred.set_data(pred_x, pred_y)
#         l_ref.set_data(ref_x, ref_y)
#         l_a.set_data(np.arange(N), pred.u_a)
#         l_d.set_data(np.arange(N), pred.u_steer)
#         ax_a.relim()
#         ax_a.autoscale_view()
#         ax_d.relim()
#         ax_d.autoscale_view()
#         fig.canvas.draw()
#         fig.canvas.flush_events()

#     t += dt
#     sim_state = copy.deepcopy(state)
#     # sim_state.p.s = np.mod(sim_state.p.s, L)
#     # dyn_model.step(sim_state)
#     dynamics_simulator.step(sim_state, T=dt)

#     # pdb.set_trace()