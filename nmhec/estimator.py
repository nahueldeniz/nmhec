"""State estimation for a generalised N-trailer vehicle from partial, noisy measurements.

``NMHE``  the nonlinear moving horizon estimator of the paper: a window of the most recent
          measurements, the whole trajectory inside it re-optimised against the kinematic
          model, everything older compressed into an adaptive arrival cost, and the joint
          angles constrained. Slippage is absorbed as a bounded disturbance on the input
          channel -- nothing is assumed about its distribution.

``EKF``   the baseline it is measured against. Cheap, and correct only while the
          disturbances are small and roughly Gaussian.

Both estimate the minimal state ``x = [x0, y0, theta0, beta_1 .. beta_N]`` from
``y = h(x) + v`` (see :mod:`nmhec.sensing`) and from the inputs that were commanded --
not the ones the ground allowed.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import casadi as ca
import numpy as np

from .arrival_cost import AdaptiveArrivalCost


def wrap(a):
    """Map an angle difference to (-pi, pi], symbolically or numerically."""
    if isinstance(a, (ca.SX, ca.MX)):
        return ca.atan2(ca.sin(a), ca.cos(a))
    return np.arctan2(np.sin(a), np.cos(a))


@dataclass
class EstimatorConfig:
    n_estimation: int = 15            # N_e, the estimation window [samples]
    w_bounds: tuple = (-0.6, 0.6)     # bound on the input-channel disturbance (rad/s, m/s)
    beta_max: float = np.pi / 2 - 10 * np.pi / 180   # Section 3.2.1: +/- (pi/2 - delta_beta),
    """delta_beta = 10 pi / 180, so the bound is +/- 1.396 rad. Do not read 1.745 off the axis
    of Fig. 15 -- that is a tick label, not the constraint, and a looser bound lets the chain
    jackknife."""
    q_w: tuple = (1.0, 1.0)           # weight on the disturbance; low = trust the measurements
    r_scale: float = 1.0              # scales the measurement weight (1 / sigma^2)
    arrival_sigma: float = 1.0
    p0_scale: float = 1e2             # initial arrival-cost covariance
    max_iter: int = 80
    warm_start: bool = True
    ipopt: dict = field(default_factory=lambda: {
        "ipopt.print_level": 0, "print_time": 0, "ipopt.sb": "yes",
        "ipopt.tol": 1e-6, "ipopt.acceptable_tol": 1e-4})


class NMHE:
    """Nonlinear moving horizon estimator. The NLP is built once and solved each step."""

    def __init__(self, model, sensors, cfg: EstimatorConfig = EstimatorConfig()):
        self.model, self.sensors, self.cfg = model, sensors, cfg
        self.nx, self.nu, self.ny = model.nx, model.nu, sensors.ny
        self.nw = model.nu                      # slippage acts through the input
        self.N = int(cfg.n_estimation)
        self.n_trailers = model.geo.n_trailers

        # which measurement channels are angles, and so need wrapping in the residual
        self.angle_mask = np.zeros(self.ny, dtype=bool)
        self.angle_mask[: self.n_trailers] = True
        idx = self.n_trailers
        for _ in sensors.cfg.pose_segments:
            if sensors.cfg.measure_attitude:
                self.angle_mask[idx] = True
                idx += 1
            if sensors.cfg.measure_position:
                idx += 2

        self._build()
        self.reset()

    # ------------------------------------------------------------------- NLP
    def _build(self):
        N, nx, nu, nw, ny = self.N, self.nx, self.nu, self.nw, self.ny
        X = ca.SX.sym("X", nx, N + 1)
        W = ca.SX.sym("W", nw, N)
        U = ca.SX.sym("U", nu, N)                  # commanded inputs over the window
        Y = ca.SX.sym("Y", ny, N + 1)              # measurements over the window
        xbar = ca.SX.sym("xbar", nx)
        Pinv = ca.SX.sym("Pinv", nx, nx)

        Qw = ca.diag(ca.DM(self.cfg.q_w))
        r = self.cfg.r_scale / np.maximum(self.sensors.sigma, 1e-9) ** 2
        Rv = ca.diag(ca.DM(r))

        d0 = X[:, 0] - xbar
        d0 = ca.vertcat(d0[:2], wrap(d0[2]), *[wrap(d0[3 + i]) for i in range(self.n_trailers)])
        J = ca.mtimes([d0.T, Pinv, d0])

        g = []
        for j in range(N + 1):
            v = Y[:, j] - self.sensors.h(X[:, j])
            v = ca.vertcat(*[wrap(v[i]) if self.angle_mask[i] else v[i] for i in range(ny)])
            J += ca.mtimes([v.T, Rv, v])
            if j < N:
                J += ca.mtimes([W[:, j].T, Qw, W[:, j]])
                g.append(X[:, j + 1] - self.model.step(X[:, j], U[:, j] + W[:, j]))

        z = ca.vertcat(ca.vec(X), ca.vec(W))
        p = ca.vertcat(ca.vec(U), ca.vec(Y), xbar, ca.vec(Pinv))
        self.solver = ca.nlpsol("nmhe", "ipopt",
                                {"x": z, "f": J, "g": ca.vertcat(*g), "p": p},
                                {**self.cfg.ipopt, "ipopt.max_iter": self.cfg.max_iter})

        big, bmax = 1e6, self.cfg.beta_max
        lbx = np.concatenate([np.tile([-big, -big, -big] + [-bmax] * self.n_trailers, N + 1),
                              np.tile(self.cfg.w_bounds[0], nw * N)])
        ubx = np.concatenate([np.tile([big, big, big] + [bmax] * self.n_trailers, N + 1),
                              np.tile(self.cfg.w_bounds[1], nw * N)])
        self.lbx, self.ubx = lbx, ubx
        self.ng = nx * N

    # ----------------------------------------------------------------- state
    def reset(self, x0=None):
        self.buf_u, self.buf_y = [], []
        self.x_hat = np.zeros(self.nx) if x0 is None else np.asarray(x0, float).copy()
        self.arrival = AdaptiveArrivalCost(self.x_hat, self.cfg.p0_scale * np.eye(self.nx),
                                           sigma=self.cfg.arrival_sigma)
        self._z0 = None
        self.last_solve = 0.0

    def __call__(self, y, u_prev) -> np.ndarray:
        """Fold in one measurement (and the input that preceded it); return the estimate."""
        import time
        self.buf_y.append(np.asarray(y, float))
        if u_prev is not None:
            self.buf_u.append(np.asarray(u_prev, float))
        if len(self.buf_y) < 2:                       # nothing to propagate through yet
            self.x_hat = self._seed_from(self.buf_y[-1])
            self.arrival.xbar = self.x_hat.copy()
            return self.x_hat

        N = self.N
        self.buf_y = self.buf_y[-(N + 1):]
        self.buf_u = self.buf_u[-N:]
        n = len(self.buf_y) - 1                       # usable window, may be shorter at the start
        Y = np.zeros((self.ny, N + 1))
        U = np.zeros((self.nu, N))
        Y[:, N - n:] = np.array(self.buf_y[-(n + 1):]).T
        Y[:, : N - n] = Y[:, [N - n]]
        U[:, N - n:] = np.array(self.buf_u[-n:]).T

        t0 = time.perf_counter()
        z0 = self._z0 if (self._z0 is not None and self.cfg.warm_start) else self._cold(Y)
        p = np.concatenate([U.ravel(order="F"), Y.ravel(order="F"),
                            self.arrival.xbar, self.arrival.weight.ravel(order="F")])
        sol = self.solver(x0=z0, p=p, lbx=self.lbx, ubx=self.ubx,
                          lbg=np.zeros(self.ng), ubg=np.zeros(self.ng))
        self.last_solve = time.perf_counter() - t0

        z = np.array(sol["x"]).ravel()
        X = z[: self.nx * (N + 1)].reshape(self.nx, N + 1, order="F")
        self.x_hat = X[:, -1].copy()
        self._z0 = z

        x_start = X[:, 1]                             # window start after the shift
        self.arrival.update(x_start, self._residual(self.buf_y[-n], x_start))
        return self.x_hat

    # --------------------------------------------------------------- helpers
    def _residual(self, y, x):
        v = np.asarray(y, float) - self.sensors.h(np.asarray(x, float))
        v[self.angle_mask] = wrap(v[self.angle_mask])
        return v

    def _seed_from(self, y) -> np.ndarray:
        """A first guess of the state from one measurement, for the very first step."""
        x = np.zeros(self.nx)
        x[3:] = y[: self.n_trailers]
        cfg = self.sensors.cfg
        idx = self.n_trailers
        theta = pos = None
        for s in cfg.pose_segments:
            if cfg.measure_attitude:
                if s == 0:
                    theta = y[idx]
                idx += 1
            if cfg.measure_position:
                if s == 0:
                    pos = y[idx:idx + 2]
                idx += 2
        if theta is not None:
            x[2] = theta
        if pos is not None:
            x[:2] = pos
        elif cfg.measure_position:                    # only a trailer is measured: walk forward
            x[:2] = y[idx - 2:idx]
        return x

    def _cold(self, Y) -> np.ndarray:
        X0 = np.tile(self._seed_from(Y[:, -1]), (self.N + 1, 1)).T
        return np.concatenate([X0.ravel(order="F"), np.zeros(self.nw * self.N)])


class EKF:
    """Extended Kalman filter on the same model and measurements, for comparison."""

    def __init__(self, model, sensors, q=1e-3, p0=1e2, r_scale=1.0):
        self.model, self.sensors = model, sensors
        self.nx, self.ny = model.nx, sensors.ny
        x = ca.SX.sym("x", self.nx)
        u = ca.SX.sym("u", model.nu)
        self.F = ca.Function("F", [x, u], [ca.jacobian(model.step(x, u), x)])
        self.H = ca.Function("H", [x], [ca.jacobian(sensors.h(x), x)])
        self.Q = q * np.eye(self.nx)
        self.R = np.diag(np.maximum(sensors.sigma, 1e-9) ** 2 / r_scale)
        self.p0 = p0
        self.angle_mask = np.zeros(self.ny, dtype=bool)
        self.angle_mask[: model.geo.n_trailers] = True
        idx = model.geo.n_trailers
        for _ in sensors.cfg.pose_segments:
            if sensors.cfg.measure_attitude:
                self.angle_mask[idx] = True
                idx += 1
            if sensors.cfg.measure_position:
                idx += 2
        self.reset()

    def reset(self, x0=None):
        self.x_hat = np.zeros(self.nx) if x0 is None else np.asarray(x0, float).copy()
        self.P = self.p0 * np.eye(self.nx)
        self.last_solve = 0.0

    def __call__(self, y, u_prev) -> np.ndarray:
        import time
        t0 = time.perf_counter()
        if u_prev is not None:
            F = np.array(self.F(self.x_hat, u_prev))
            self.x_hat = self.model.simulate(self.x_hat, u_prev)
            self.P = F @ self.P @ F.T + self.Q
        H = np.array(self.H(self.x_hat))
        v = np.asarray(y, float) - self.sensors.h(self.x_hat)
        v[self.angle_mask] = wrap(v[self.angle_mask])
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x_hat = self.x_hat + K @ v
        self.P = (np.eye(self.nx) - K @ H) @ self.P
        self.last_solve = time.perf_counter() - t0
        return self.x_hat
