from collections import deque

import numpy as np
from matplotlib import pyplot as plt, patches
from matplotlib.transforms import Affine2D


class LMPCVisualizer:
    def __init__(self, track_obj, VL=0.37, VW=0.195, BUFFER_MAXLEN=50):
        self.VL, self.VW = VL, VW
        self.BUFFER_MAXLEN = BUFFER_MAXLEN
        self.track_obj = track_obj
        self.enabled = False
        self.controller = None

    def initialize(self):
        plt.ion()
        self.fig = plt.figure(figsize=(15, 10))
        self.ax_xy = self.fig.add_subplot(1, 2, 1)
        self.ax_v = self.fig.add_subplot(3, 2, 2)
        self.ax_a = self.fig.add_subplot(3, 2, 4)
        self.ax_d = self.fig.add_subplot(3, 2, 6)
        self.track_obj.plot_map(self.ax_xy)
        self.ax_xy.set_aspect('equal')
        self.l_pred = self.ax_xy.plot([], [], 'b-o', markersize=4)[0]
        self.l_ss = self.ax_xy.plot([], [], 'rs', markersize=4, markerfacecolor='None')[0]
        self.l_pa = self.ax_a.plot([], [], '-go')[0]
        self.l_pd = self.ax_d.plot([], [], '-go')[0]
        self.l_v = self.ax_v.plot([], [], '-bo')[0]
        self.l_a = self.ax_a.plot([], [], '-bo')[0]
        self.l_d = self.ax_d.plot([], [], '-bo')[0]
        self.rect = patches.Rectangle((-0.5 * self.VL, -0.5 * self.VW), self.VL, self.VW, linestyle='solid', color='b', alpha=0.5)
        self.ax_xy.add_patch(self.rect)
        self.ax_a.set_ylabel('long in')
        self.ax_d.set_ylabel('lat in')
        self.ax_v.set_ylabel('long vel')
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        self.v_data = deque([], maxlen=self.BUFFER_MAXLEN)
        self.a_data = deque([], maxlen=self.BUFFER_MAXLEN)
        self.d_data = deque([], maxlen=self.BUFFER_MAXLEN)

    def bind_controller(self, controller):
        self.controller = controller

    def step(self, state=None, *, q=None, u=None, pred=None, ss=None):
        if state is not None:
            x, y, psi = self.track_obj.local_to_global((state.p.s, state.p.x_tran, state.p.e_psi))
            v_long, v_tran, w_psi = state.v.v_long, state.v.v_tran, state.w.w_psi
            u_a, u_steer = state.u.u_a, state.u.u_steer
        elif q is not None:
            v_long, v_tran, w_psi, x, y, psi = q
            u_a, u_steer = u
        else:
            raise ValueError("Must provide either state or q")

        if self.controller is not None:
            pred = self.controller.get_prediction()
            ss = self.controller.get_safe_set()

        b_left = x - self.VL / 2
        b_bot = y - self.VW / 2
        r = Affine2D().rotate_around(x, y, psi) + self.ax_xy.transData
        self.rect.set_xy((b_left, b_bot))
        self.rect.set_transform(r)
        pred_x, pred_y, ss_x, ss_y = [], [], [], []

        # Plot the predictions
        if pred is not None:
            if pred.x is not None:
                pred_x, pred_y = pred.x, pred.y
            else:
                for i in range(len(pred.s)):
                    x, y, psi = self.track_obj.local_to_global((pred.s[i], pred.x_tran[i], pred.e_psi[i]))
                    pred_x.append(x)
                    pred_y.append(y)
            self.l_pred.set_data(pred_x, pred_y)
            self.l_pa.set_data(np.arange(len(self.a_data), len(self.a_data) + len(pred.u_a)), pred.u_a)
            self.l_pd.set_data(np.arange(len(self.d_data), len(self.d_data) + len(pred.u_steer)), pred.u_steer)

        # Plot the safety set (red squares)
        if ss is not None:
            # for i in range(len(ss.s)):
            #     x, y, psi = self.track_obj.local_to_global((ss.s[i], ss.x_tran[i], ss.e_psi[i]))
            for s_i, x_tran_i, e_psi_i in zip(ss.s, ss.x_tran, ss.e_psi):
                x, y, psi = self.track_obj.local_to_global((s_i, x_tran_i, e_psi_i))
                ss_x.append(x)
                ss_y.append(y)
            self.l_ss.set_data(ss_x, ss_y)

        self.v_data.append(v_long)
        self.a_data.append(u_a)
        self.d_data.append(u_steer)
        self.l_v.set_data(np.arange(len(self.v_data)), self.v_data)
        self.l_a.set_data(np.arange(len(self.a_data)), self.a_data)
        self.l_d.set_data(np.arange(len(self.d_data)), self.d_data)
        self.ax_v.relim()
        self.ax_v.autoscale_view()
        self.ax_a.relim()
        self.ax_a.autoscale_view()
        self.ax_d.relim()
        self.ax_d.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def reset(self):
        if not self.enabled:
            self.enabled = True
            self.initialize()
        self.l_pred.set_data([], [])
        self.l_ss.set_data([], [])
        self.l_pa.set_data([], [])
        self.l_pd.set_data([], [])
        self.l_v.set_data([], [])
        self.l_a.set_data([], [])
        self.l_d.set_data([], [])
        self.v_data.clear()
        self.a_data.clear()
        self.d_data.clear()

    def close(self):
        if not self.enabled:
            return
        self.enabled = False
        plt.ioff()
        plt.close(self.fig)


class MultiVehicleVisualizer:
    def __init__(self, track_obj, VL=0.37, VW=0.195, BUFFER_MAXLEN=50, n_vehicles=2):
        self.VL, self.VW = VL, VW
        self.BUFFER_MAXLEN = BUFFER_MAXLEN
        self.track_obj = track_obj
        self.enabled = False
        self.controller = None
        self.n_vehicles = n_vehicles
        
        # Define colors for each vehicle
        self.colors = ['b', 'r', 'g', 'c', 'm', 'y']
        if n_vehicles > len(self.colors):
            # If we need more colors than we have, repeat the colors
            self.colors = self.colors * (n_vehicles // len(self.colors) + 1)
            self.colors = self.colors[:n_vehicles]

    def initialize(self):
        plt.ion()
        self.fig = plt.figure(figsize=(15, 10))
        self.ax_xy = self.fig.add_subplot(1, 2, 1)
        self.ax_v = self.fig.add_subplot(3, 2, 2)
        self.ax_a = self.fig.add_subplot(3, 2, 4)
        self.ax_d = self.fig.add_subplot(3, 2, 6)
        self.track_obj.plot_map(self.ax_xy)
        self.ax_xy.set_aspect('equal')
        
        # Create prediction and safety set lines for each vehicle
        self.l_pred = []
        self.l_ss = []
        for i in range(self.n_vehicles):
            color = self.colors[i]
            self.l_pred.append(self.ax_xy.plot([], [], f'{color}-o', markersize=4)[0])
            self.l_ss.append(self.ax_xy.plot([], [], f'{color}s', markersize=4, markerfacecolor='None')[0])
        
        # Create control and velocity lines for each vehicle
        self.l_pa = []
        self.l_pd = []
        self.l_v = []
        self.l_a = []
        self.l_d = []
        for i in range(self.n_vehicles):
            color = self.colors[i]
            self.l_pa.append(self.ax_a.plot([], [], f'-{color}o')[0])
            self.l_pd.append(self.ax_d.plot([], [], f'-{color}o')[0])
            self.l_v.append(self.ax_v.plot([], [], f'-{color}o')[0])
            self.l_a.append(self.ax_a.plot([], [], f'-{color}o')[0])
            self.l_d.append(self.ax_d.plot([], [], f'-{color}o')[0])
        
        # Create vehicle rectangles for each vehicle
        self.rects = []
        for i in range(self.n_vehicles):
            color = self.colors[i]
            rect = patches.Rectangle((-0.5 * self.VL, -0.5 * self.VW), self.VL, self.VW, 
                                    linestyle='solid', color=color, alpha=0.5)
            self.ax_xy.add_patch(rect)
            self.rects.append(rect)
        
        self.ax_a.set_ylabel('long in')
        self.ax_d.set_ylabel('lat in')
        self.ax_v.set_ylabel('long vel')
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        # Create data buffers for each vehicle
        self.v_data = [deque([], maxlen=self.BUFFER_MAXLEN) for _ in range(self.n_vehicles)]
        self.a_data = [deque([], maxlen=self.BUFFER_MAXLEN) for _ in range(self.n_vehicles)]
        self.d_data = [deque([], maxlen=self.BUFFER_MAXLEN) for _ in range(self.n_vehicles)]

    def bind_controller(self, controller):
        self.controller = controller

    def step(self, state=None, *, q=None, u=None, pred=None, ss=None):
        if state is not None:
            # Handle list of vehicle states
            if isinstance(state, list):
                # Process each vehicle state
                for i, vehicle_state in enumerate(state):
                    self._process_vehicle_state(i, vehicle_state)
            else:
                # Backward compatibility for single vehicle state
                self._process_vehicle_state(0, state)
        elif q is not None:
            # Handle list of vehicle states in q format
            if isinstance(q, list):
                for i, vehicle_q in enumerate(q):
                    self._process_vehicle_q(i, vehicle_q, u[i] if isinstance(u, list) else u)
            else:
                # Backward compatibility for single vehicle state in q format
                self._process_vehicle_q(0, q, u)
        else:
            raise ValueError("Must provide either state or q")

        # Update the plot
        self.ax_v.relim()
        self.ax_v.autoscale_view()
        self.ax_a.relim()
        self.ax_a.autoscale_view()
        self.ax_d.relim()
        self.ax_d.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def _process_vehicle_state(self, index, state):
        """Process a single vehicle state and update the visualization"""
        # Extract vehicle state
        x, y, psi = self.track_obj.local_to_global((state.p.s, state.p.x_tran, state.p.e_psi))
        v_long, v_tran, w_psi = state.v.v_long, state.v.v_tran, state.w.w_psi
        u_a, u_steer = state.u.u_a, state.u.u_steer
        
        # Get prediction and safety set if controller is available
        pred = None
        ss = None
        if self.controller is not None and index == 0:  # Only use controller for the first vehicle
            pred = self.controller.get_prediction()
            ss = self.controller.get_safe_set()
        
        # Update vehicle rectangle
        b_left = x - self.VL / 2
        b_bot = y - self.VW / 2
        r = Affine2D().rotate_around(x, y, psi) + self.ax_xy.transData
        self.rects[index].set_xy((b_left, b_bot))
        self.rects[index].set_transform(r)
        
        # Plot predictions
        if pred is not None:
            pred_x, pred_y = [], []
            if pred.x is not None:
                pred_x, pred_y = pred.x, pred.y
            else:
                for i in range(len(pred.s)):
                    x, y, psi = self.track_obj.local_to_global((pred.s[i], pred.x_tran[i], pred.e_psi[i]))
                    pred_x.append(x)
                    pred_y.append(y)
            self.l_pred[index].set_data(pred_x, pred_y)
            self.l_pa[index].set_data(np.arange(len(self.a_data[index]), len(self.a_data[index]) + len(pred.u_a)), pred.u_a)
            self.l_pd[index].set_data(np.arange(len(self.d_data[index]), len(self.d_data[index]) + len(pred.u_steer)), pred.u_steer)
        
        # Plot safety set
        if ss is not None:
            ss_x, ss_y = [], []
            for s_i, x_tran_i, e_psi_i in zip(ss.s, ss.x_tran, ss.e_psi):
                x, y, psi = self.track_obj.local_to_global((s_i, x_tran_i, e_psi_i))
                ss_x.append(x)
                ss_y.append(y)
            self.l_ss[index].set_data(ss_x, ss_y)
        
        # Update data buffers
        self.v_data[index].append(v_long)
        self.a_data[index].append(u_a)
        self.d_data[index].append(u_steer)
        
        # Update plots
        self.l_v[index].set_data(np.arange(len(self.v_data[index])), self.v_data[index])
        self.l_a[index].set_data(np.arange(len(self.a_data[index])), self.a_data[index])
        self.l_d[index].set_data(np.arange(len(self.d_data[index])), self.d_data[index])

    def _process_vehicle_q(self, index, q, u):
        """Process a single vehicle state in q format and update the visualization"""
        # Extract vehicle state
        v_long, v_tran, w_psi, x, y, psi = q
        u_a, u_steer = u
        
        # Update vehicle rectangle
        b_left = x - self.VL / 2
        b_bot = y - self.VW / 2
        r = Affine2D().rotate_around(x, y, psi) + self.ax_xy.transData
        self.rects[index].set_xy((b_left, b_bot))
        self.rects[index].set_transform(r)
        
        # Update data buffers
        self.v_data[index].append(v_long)
        self.a_data[index].append(u_a)
        self.d_data[index].append(u_steer)
        
        # Update plots
        self.l_v[index].set_data(np.arange(len(self.v_data[index])), self.v_data[index])
        self.l_a[index].set_data(np.arange(len(self.a_data[index])), self.a_data[index])
        self.l_d[index].set_data(np.arange(len(self.d_data[index])), self.d_data[index])

    def reset(self):
        if not self.enabled:
            self.enabled = True
            self.initialize()
        
        # Clear all plots
        for i in range(self.n_vehicles):
            self.l_pred[i].set_data([], [])
            self.l_ss[i].set_data([], [])
            self.l_pa[i].set_data([], [])
            self.l_pd[i].set_data([], [])
            self.l_v[i].set_data([], [])
            self.l_a[i].set_data([], [])
            self.l_d[i].set_data([], [])
            self.v_data[i].clear()
            self.a_data[i].clear()
            self.d_data[i].clear()

    def close(self):
        if not self.enabled:
            return
        self.enabled = False
        plt.ioff()
        plt.close(self.fig)
