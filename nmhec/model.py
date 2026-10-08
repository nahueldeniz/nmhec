# Shared with the nmppfc package (paper 02, IEEE T-ASE 2025): this file is a copy,
# kept byte-identical below the header so fixes can be ported between the two repos.
# Source: github.com/nahueldeniz/nmppfc
"""Kinematic model of a generalised N-trailer (GNT) vehicle, built with CasADi.

An active segment (the tractor, a unicycle) pulls ``n_trailers`` passive segments.
Segment ``i`` is hitched to segment ``i-1`` at a distance ``hitch[i]`` behind the axle of
segment ``i-1`` (off-axle hitching; ``hitch[i] = 0`` is the standard, on-axle case) and its
own axle is ``link[i]`` behind that hitch point.

State (minimal):  x = [x0, y0, theta0, beta_1, ..., beta_N]
Control:          u = [omega_0, v_0]        (tractor angular and linear velocity)

The pose of every segment is recovered algebraically from this state; the paper carries the
redundant vector q = (beta_i, theta_0, theta_i, x_i, y_i), which describes the same
configuration (see NOTES.md).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import casadi as ca
import numpy as np


@dataclass
class VehicleGeometry:
    """Geometry of the chain. ``hitch`` and ``link`` have one entry per trailer."""

    n_trailers: int = 2
    hitch: tuple = (0.342, 0.0)     # L_h,i [m]
    link: tuple = (1.08, 0.78)      # L_i [m]
    body_length: tuple = (0.99, 1.0, 1.0)   # per segment, tractor first [m]
    body_width: tuple = (0.67, 0.8, 0.8)    # per segment, tractor first [m]

    def __post_init__(self):
        if len(self.hitch) != self.n_trailers or len(self.link) != self.n_trailers:
            raise ValueError("hitch and link must have one entry per trailer")
        if any(l <= 0 for l in self.link):
            raise ValueError("link lengths must be positive")
        if len(self.body_length) < self.n_trailers + 1:
            self.body_length = tuple(self.body_length) + (1.0,) * (
                self.n_trailers + 1 - len(self.body_length))
            self.body_width = tuple(self.body_width) + (0.8,) * (
                self.n_trailers + 1 - len(self.body_width))

    @property
    def n_segments(self) -> int:
        return self.n_trailers + 1

    def half_diagonal(self, i: int) -> float:
        """Radius of the disc that covers segment ``i`` (used for clearance)."""
        return 0.5 * float(np.hypot(self.body_length[i], self.body_width[i]))


class NTrailer:
    """CasADi model: continuous dynamics, RK4 discretisation and segment poses."""

    def __init__(self, geometry: VehicleGeometry = VehicleGeometry(), ts: float = 0.05):
        self.geo = geometry
        self.ts = float(ts)
        self.nx = 3 + geometry.n_trailers
        self.nu = 2

        x = ca.SX.sym("x", self.nx)
        u = ca.SX.sym("u", self.nu)
        self.f = ca.Function("f", [x, u], [self._ode(x, u)], ["x", "u"], ["xdot"])
        self.step = ca.Function("step", [x, u], [self._rk4(x, u)], ["x", "u"], ["x_next"])
        self.poses = ca.Function("poses", [x], [self._poses(x)], ["x"], ["p"])

    # ----------------------------------------------------------------- dynamics
    def _ode(self, x, u):
        th0, w0, v0 = x[2], u[0], u[1]
        xdot = [v0 * ca.cos(th0), v0 * ca.sin(th0), w0]
        v_prev, w_prev = v0, w0
        for i in range(self.geo.n_trailers):
            b = x[3 + i]
            lh, ll = self.geo.hitch[i], self.geo.link[i]
            v_i = v_prev * ca.cos(b) + lh * w_prev * ca.sin(b)
            w_i = (v_prev * ca.sin(b) - lh * w_prev * ca.cos(b)) / ll
            xdot.append(w_prev - w_i)          # beta_i_dot = omega_{i-1} - omega_i
            v_prev, w_prev = v_i, w_i
        return ca.vertcat(*xdot)

    def _rk4(self, x, u):
        h = self.ts
        k1 = self._ode(x, u)
        k2 = self._ode(x + h / 2 * k1, u)
        k3 = self._ode(x + h / 2 * k2, u)
        k4 = self._ode(x + h * k3, u)
        return x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    # -------------------------------------------------------------------- poses
    def _poses(self, x):
        """(3 x n_segments) matrix of segment poses [x; y; theta], tractor first."""
        cols = [ca.vertcat(x[0], x[1], x[2])]
        th_prev, xp, yp = x[2], x[0], x[1]
        for i in range(self.geo.n_trailers):
            th_i = th_prev - x[3 + i]
            lh, ll = self.geo.hitch[i], self.geo.link[i]
            xi = xp - lh * ca.cos(th_prev) - ll * ca.cos(th_i)
            yi = yp - lh * ca.sin(th_prev) - ll * ca.sin(th_i)
            cols.append(ca.vertcat(xi, yi, th_i))
            th_prev, xp, yp = th_i, xi, yi
        return ca.horzcat(*cols)

    # ------------------------------------------------------------------ helpers
    def pose_array(self, x) -> np.ndarray:
        """Numeric (3 x n_segments) poses for a numeric state."""
        return np.array(self.poses(ca.DM(x))).reshape(3, self.geo.n_segments)

    def state_from_pose(self, x0=0.0, y0=0.0, theta0=0.0, betas=None) -> np.ndarray:
        betas = np.zeros(self.geo.n_trailers) if betas is None else np.asarray(betas, float)
        return np.concatenate([[x0, y0, theta0], betas])

    def simulate(self, x, u, n_steps: int = 1) -> np.ndarray:
        x = ca.DM(x)
        for _ in range(n_steps):
            x = self.step(x, ca.DM(u))
        return np.array(x).ravel()
