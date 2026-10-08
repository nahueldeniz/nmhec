"""The redundant state vector of the paper, and the filter that gets into trouble with it.

Section 2.1 carries ``q = (beta_i, theta_i, x_i, y_i)``: every segment's pose is a state of
its own, even though the chain determines it from the tractor's pose and the joint angles.
The estimator is supposed to hold those coordinates together -- in the NMHE the kinematic
relations are equality constraints of the optimisation, so every iterate is a configuration
the vehicle could actually be in.

A Kalman filter has no such mechanism. Its correction is a linear pull of the whole state
towards the measurement, and nothing in it knows that ``(x_N, y_N, theta_N)`` must remain
reachable from ``(x_0, y_0, theta_0)`` through ``beta_1 .. beta_N``. Under slippage the
prediction and the measurement disagree persistently, and the filter settles on a state whose
last-trailer pose no chain configuration admits: the estimate is not merely inaccurate, it
is infeasible. That is what :func:`feasibility_error` measures, and it is the mechanism the
paper reports when the EKF "provides non-feasible estimates".
"""
from __future__ import annotations

import casadi as ca
import numpy as np


class RedundantModel:
    """Kinematics written on ``q``: [x_0..x_N, y_0..y_N, theta_0..theta_N, beta_1..beta_N]."""

    def __init__(self, geometry, ts: float = 0.1):
        self.geo = geometry
        self.ts = float(ts)
        self.n = geometry.n_trailers
        self.ns = self.n + 1                      # segments, tractor included
        self.nx = 3 * self.ns + self.n
        self.nu = 2
        q = ca.SX.sym("q", self.nx)
        u = ca.SX.sym("u", self.nu)
        self.step = ca.Function("step", [q, u], [self._rk4(q, u)])

    # ------------------------------------------------------------------ index
    def ix(self, i): return i
    def iy(self, i): return self.ns + i
    def ith(self, i): return 2 * self.ns + i
    def ib(self, i): return 3 * self.ns + i       # i = 0 .. n-1 for beta_1 .. beta_N

    # --------------------------------------------------------------- dynamics
    def _ode(self, q, u):
        ns, n = self.ns, self.n
        v, w = [u[1]], [u[0]]
        for i in range(n):
            b = q[self.ib(i)]
            lh, ll = self.geo.hitch[i], self.geo.link[i]
            v.append(v[i] * ca.cos(b) + lh * w[i] * ca.sin(b))
            w.append((v[i] * ca.sin(b) - lh * w[i] * ca.cos(b)) / ll)
        dx = [v[i] * ca.cos(q[self.ith(i)]) for i in range(ns)]
        dy = [v[i] * ca.sin(q[self.ith(i)]) for i in range(ns)]
        dth = [w[i] for i in range(ns)]
        db = [w[i] - w[i + 1] for i in range(n)]
        return ca.vertcat(*dx, *dy, *dth, *db)

    def _rk4(self, q, u):
        h = self.ts
        k1 = self._ode(q, u)
        k2 = self._ode(q + h / 2 * k1, u)
        k3 = self._ode(q + h / 2 * k2, u)
        k4 = self._ode(q + h * k3, u)
        return q + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    # ---------------------------------------------------------------- mapping
    def from_minimal(self, x) -> np.ndarray:
        """Build a consistent q from the minimal state (used to initialise the filter)."""
        x = np.asarray(x, float)
        q = np.zeros(self.nx)
        th, xi, yi = x[2], x[0], x[1]
        q[self.ix(0)], q[self.iy(0)], q[self.ith(0)] = xi, yi, th
        for i in range(self.n):
            b = x[3 + i]
            lh, ll = self.geo.hitch[i], self.geo.link[i]
            th_next = th - b
            xi = xi - lh * np.cos(th) - ll * np.cos(th_next)
            yi = yi - lh * np.sin(th) - ll * np.sin(th_next)
            th = th_next
            q[self.ix(i + 1)], q[self.iy(i + 1)], q[self.ith(i + 1)] = xi, yi, th
            q[self.ib(i)] = b
        return q

    def to_minimal(self, q) -> np.ndarray:
        q = np.asarray(q, float)
        return np.concatenate([[q[self.ix(0)], q[self.iy(0)], q[self.ith(0)]],
                               [q[self.ib(i)] for i in range(self.n)]])

    def pose(self, q, segment: int) -> np.ndarray:
        """The filter's own estimate of a segment's pose -- feasible or not."""
        q = np.asarray(q, float)
        return np.array([q[self.ix(segment)], q[self.iy(segment)], q[self.ith(segment)]])

    def measurement_matrix(self, sensors) -> np.ndarray:
        """h(q) is a selection here: every measured quantity is a state."""
        cfg = sensors.cfg
        rows = []
        for i in range(self.n):
            r = np.zeros(self.nx); r[self.ib(i)] = 1.0; rows.append(r)
        for s in cfg.pose_segments:
            if cfg.measure_attitude:
                r = np.zeros(self.nx); r[self.ith(s)] = 1.0; rows.append(r)
            if cfg.measure_position:
                r = np.zeros(self.nx); r[self.ix(s)] = 1.0; rows.append(r)
                r = np.zeros(self.nx); r[self.iy(s)] = 1.0; rows.append(r)
        return np.array(rows)


def feasibility_error(model: RedundantModel, q) -> float:
    """Metres between the filter's last-trailer pose and the one the chain actually implies.

    Zero for any configuration the vehicle could hold. Anything else is an estimate that
    describes no physical vehicle -- the trailers have been pulled off their hitches.
    """
    q = np.asarray(q, float)
    consistent = model.from_minimal(model.to_minimal(q))
    n = model.n
    return float(np.hypot(q[model.ix(n)] - consistent[model.ix(n)],
                          q[model.iy(n)] - consistent[model.iy(n)]))


class RedundantEKF:
    """EKF on q, as the paper's comparison uses. Nothing enforces the chain kinematics."""

    def __init__(self, model: RedundantModel, sensors, q=1e-3, p0=1e2, r_scale=1.0):
        self.model, self.sensors = model, sensors
        self.nx, self.ny = model.nx, sensors.ny
        qs = ca.SX.sym("q", self.nx)
        us = ca.SX.sym("u", model.nu)
        self.F = ca.Function("F", [qs, us], [ca.jacobian(model.step(qs, us), qs)])
        self.H = model.measurement_matrix(sensors)
        self.Q = q * np.eye(self.nx)
        self.R = np.diag(np.maximum(sensors.sigma, 1e-9) ** 2 / r_scale)
        self.p0 = p0
        self.angle_mask = np.zeros(self.ny, dtype=bool)
        self.angle_mask[: model.n] = True
        idx = model.n
        for _ in sensors.cfg.pose_segments:
            if sensors.cfg.measure_attitude:
                self.angle_mask[idx] = True
                idx += 1
            if sensors.cfg.measure_position:
                idx += 2
        self.reset()

    def reset(self, x0=None):
        self.q_hat = (np.zeros(self.nx) if x0 is None
                      else self.model.from_minimal(np.asarray(x0, float)))
        self.P = self.p0 * np.eye(self.nx)
        self.last_solve = 0.0

    @property
    def x_hat(self) -> np.ndarray:
        return self.model.to_minimal(self.q_hat)

    def pose(self, segment: int) -> np.ndarray:
        return self.model.pose(self.q_hat, segment)

    @property
    def feasibility_error(self) -> float:
        return feasibility_error(self.model, self.q_hat)

    def __call__(self, y, u_prev) -> np.ndarray:
        import time
        t0 = time.perf_counter()
        if u_prev is not None:
            F = np.array(self.F(self.q_hat, u_prev))
            self.q_hat = np.array(self.model.step(self.q_hat, u_prev)).ravel()
            self.P = F @ self.P @ F.T + self.Q
        v = np.asarray(y, float) - self.H @ self.q_hat
        v[self.angle_mask] = np.arctan2(np.sin(v[self.angle_mask]), np.cos(v[self.angle_mask]))
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.q_hat = self.q_hat + K @ v
        self.P = (np.eye(self.nx) - K @ self.H) @ self.P
        self.last_solve = time.perf_counter() - t0
        return self.x_hat
