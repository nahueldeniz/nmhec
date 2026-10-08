"""What the vehicle can actually measure, and what the ground does to it.

Two separate things live here, and keeping them apart matters:

* :class:`Sensors` -- the measurement equation, Eq. (3) of the paper:
  ``r_t = h(q_t) + v_t``, the joint angles plus the pose of *at least one* segment.
  Which segment is measured is a parameter. The simulation study of the paper measures
  the tractor; the G2T field rig carries the RTK GPS on the axle of the last trailer and
  the IMU on the tractor, so the measured pose is split across two segments there.

* :class:`Ground` -- the slippage of Eq. (2). The wheel-soil interaction scales the
  commanded velocities, ``u_eff = (rho_1 omega, rho_2 v)``, so the plant moves differently
  from the model the controller and the estimator carry. ``rho = 1`` is no slip; the paper
  uses 0.5, 0.75 and 0.9. The estimator never sees rho: to it, that mismatch is a bounded
  process disturbance, which is the whole point of preferring it to a Kalman filter here.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SensorConfig:
    """Standard deviations follow the paper's simulation study (Section 3.1)."""

    pose_segments: tuple = (0,)          # which segments report a pose; 0 = tractor
    measure_position: bool = True        # x, y of those segments
    measure_attitude: bool = True        # theta of those segments
    sigma_beta: float = 0.2 * np.pi / 180      # joint-angle encoders [rad]
    sigma_theta: float = 0.2 * np.pi / 180     # tractor attitude [rad]
    sigma_xy: float = np.sqrt(0.05)            # position [m]
    outlier_prob: float = 0.0            # paper: 1 in 5 position measurements, scenario (b)
    outlier_sigma: float = 0.2           # N(0, 0.2) added to the position when one occurs
    attitude_scale_noise: float = 0.0    # m_theta of scenario (c): r = theta (1 + m), IMU error

    def __post_init__(self):
        if not self.pose_segments:
            raise ValueError("at least one segment must report a pose (Eq. 3)")
        if not (self.measure_position or self.measure_attitude):
            raise ValueError("a pose measurement must carry a position, an attitude, or both")


class Sensors:
    """Measurement equation and its noise, for a given vehicle."""

    def __init__(self, model, cfg: SensorConfig = SensorConfig()):
        self.model = model
        self.cfg = cfg
        self.n_trailers = model.geo.n_trailers
        per_pose = (2 if cfg.measure_position else 0) + (1 if cfg.measure_attitude else 0)
        self.ny = self.n_trailers + per_pose * len(cfg.pose_segments)

    # ------------------------------------------------------------------ h(x)
    def h(self, x):
        """Noise-free measurement. Works on numeric arrays and on CasADi symbols alike."""
        import casadi as ca
        symbolic = isinstance(x, (ca.SX, ca.MX))
        poses = self.model.poses(x) if symbolic else self.model.pose_array(x)
        rows = [x[3 + i] for i in range(self.n_trailers)]        # beta_1 .. beta_N
        for s in self.cfg.pose_segments:
            if self.cfg.measure_attitude:
                rows.append(poses[2, s])
            if self.cfg.measure_position:
                rows += [poses[0, s], poses[1, s]]
        return ca.vertcat(*rows) if symbolic else np.array(rows, dtype=float)

    @property
    def sigma(self) -> np.ndarray:
        """Per-channel standard deviation, in the order h() returns."""
        s = [self.cfg.sigma_beta] * self.n_trailers
        for _ in self.cfg.pose_segments:
            if self.cfg.measure_attitude:
                s.append(self.cfg.sigma_theta)
            if self.cfg.measure_position:
                s += [self.cfg.sigma_xy, self.cfg.sigma_xy]
        return np.array(s)

    def measure(self, x, rng: np.random.Generator) -> np.ndarray:
        """A noisy reading of the true state, with outliers and IMU scale error."""
        y = self.h(np.asarray(x, dtype=float)).astype(float)
        y = y + rng.normal(0.0, self.sigma)
        cfg = self.cfg
        idx = self.n_trailers
        for _ in cfg.pose_segments:
            if cfg.measure_attitude:
                if cfg.attitude_scale_noise:            # scenario (c): uncalibrated IMU
                    y[idx] *= 1.0 + rng.normal(0.0, cfg.attitude_scale_noise)
                idx += 1
            if cfg.measure_position:
                if cfg.outlier_prob and rng.random() < cfg.outlier_prob:
                    y[idx:idx + 2] += rng.normal(0.0, cfg.outlier_sigma, 2)
                idx += 2
        return y


@dataclass
class Ground:
    """Slippage, Eq. (2): the commanded velocities are not the ones the vehicle achieves."""

    rho_omega: float = 1.0     # rho_1, on the angular velocity
    rho_v: float = 1.0         # rho_2, on the linear velocity
    onset: float = 0.0         # seconds before the slip starts (0 = from the beginning)

    def apply(self, u, t: float = np.inf) -> np.ndarray:
        u = np.asarray(u, dtype=float)
        if t < self.onset:
            return u.copy()
        return u * np.array([self.rho_omega, self.rho_v])

    @property
    def slipping(self) -> bool:
        return self.rho_omega != 1.0 or self.rho_v != 1.0
