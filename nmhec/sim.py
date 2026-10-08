"""Closed loop: the controller is driven by the estimate, never by the true state.

    truth --> sensors --> estimator --> path-following algorithm --> NMPC --> truth
                                                                       |
                                   the ground scales what the vehicle actually achieves

This is the arrangement the paper reports. Nothing downstream of :class:`~nmhec.sensing.Ground`
knows the slip coefficients, and nothing downstream of the sensors sees the true state -- with
one exception: ``estimator="oracle"`` feeds the controller the truth, which is not a method,
only a yardstick. It answers "how much of this path error is the estimator's fault?".
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .controller import ControllerConfig, NMPPFC
from .estimator import EKF, NMHE, EstimatorConfig
from .model import NTrailer, VehicleGeometry
from .pfa import PFAConfig, TargetUpdater
from .redundant import RedundantEKF, RedundantModel, feasibility_error
from .sensing import Ground, SensorConfig, Sensors


@dataclass
class SimulationResult:
    time: np.ndarray
    states: np.ndarray            # (steps + 1, nx) true state
    estimates: np.ndarray         # (steps + 1, nx)
    poses: np.ndarray             # (steps + 1, 3, n_segments) true poses
    controls: np.ndarray          # (steps, 2) commanded (omega, v)
    targets: np.ndarray           # (steps, 3)
    off_track: np.ndarray         # (steps,) guided segment to the path [m]
    estimation_error: np.ndarray  # (steps,) position error of the guided segment [m]
    joint_angles: np.ndarray      # (steps + 1, n_trailers) true beta
    t_estimator: np.ndarray       # (steps,) seconds per estimation step
    t_controller: np.ndarray      # (steps,) seconds per control step
    infeasibility: np.ndarray     # (steps,) how far the estimate is from any real configuration

    def summary(self, ts: float) -> dict:
        return {
            "off_track_mean": float(np.mean(self.off_track)),
            "off_track_max": float(np.max(self.off_track)),
            "estimation_mean": float(np.mean(self.estimation_error)),
            "beta_max": float(np.max(np.abs(self.joint_angles))),
            "rtf_estimator": float(np.mean(self.t_estimator) / ts),
            "rtf_controller": float(np.mean(self.t_controller) / ts),
            "infeasibility_mean": float(np.mean(self.infeasibility)),
            "infeasibility_max": float(np.max(self.infeasibility)),
        }


class ClosedLoop:
    def __init__(self, path_points, geometry: VehicleGeometry = VehicleGeometry(),
                 ts: float = 0.1, sensors: SensorConfig = SensorConfig(),
                 ground: Ground = Ground(), estimator: str = "nmhe",
                 est_cfg: EstimatorConfig = EstimatorConfig(),
                 ctrl_cfg: ControllerConfig | None = None,
                 pfa_cfg: PFAConfig = PFAConfig(), seed: int = 0, closed_path: bool = True):
        self.model = NTrailer(geometry, ts=ts)
        self.sensors = Sensors(self.model, sensors)
        self.ground = ground
        self.ts = float(ts)
        self.kind = estimator
        self.rng = np.random.default_rng(seed)

        last = geometry.n_trailers
        if ctrl_cfg is None:
            # defaults chosen so the chain tracks a tight path without jackknifing; the
            # resulting speed (~0.23 m/s) is close to the 0.28 m/s of the field trials
            ctrl_cfg = ControllerConfig(guided_segment=last, beta_max=est_cfg.beta_max,
                                        r=(0.02, 0.05), v_max=0.8)
        self.ctrl_cfg = ctrl_cfg
        self.controller = NMPPFC(self.model, ctrl_cfg, n_obstacles=0)

        pfa_cfg.guided_segment = last if pfa_cfg.guided_segment < 0 else pfa_cfg.guided_segment
        self.pfa_cfg = pfa_cfg
        self.path_points = np.asarray(path_points, float)[:, :2]
        self.closed_path = closed_path

        if estimator == "nmhe":
            self.estimator = NMHE(self.model, self.sensors, est_cfg)
        elif estimator == "ekf":
            # the paper's comparison: a filter on the redundant state q, so its estimate of a
            # trailer pose is a free variable rather than a consequence of the chain
            self.estimator = RedundantEKF(RedundantModel(geometry, ts=ts), self.sensors)
        elif estimator == "ekf_minimal":
            self.estimator = EKF(self.model, self.sensors)
        elif estimator == "oracle":
            self.estimator = None
        else:
            raise ValueError(
                "estimator must be 'nmhe', 'ekf', 'ekf_minimal' or 'oracle'")

    # ------------------------------------------------------------------- run
    def run(self, n_steps: int = 300, x0=None, progress=None) -> SimulationResult:
        import time as _time
        model, cfg = self.model, self.ctrl_cfg
        pfa = TargetUpdater(self.path_points, self.pfa_cfg, closed=self.closed_path)
        guided = self.pfa_cfg.guided_segment

        if x0 is None:
            p0 = pfa.target
            x0 = model.state_from_pose(p0[0], p0[1], p0[2])
        x = np.asarray(x0, float).copy()
        if self.estimator is not None:
            self.estimator.reset(x)

        states, ests, poses = [x.copy()], [x.copy()], [model.pose_array(x)]
        controls, targets, off, eerr, te, tc, infeas = [], [], [], [], [], [], []
        u_prev = np.zeros(2)

        for k in range(n_steps):
            if self.estimator is None:
                x_hat, dt_e = x.copy(), 0.0
            else:
                y = self.sensors.measure(x, self.rng)
                t0 = _time.perf_counter()
                x_hat = self.estimator(y, u_prev if k else None)
                dt_e = _time.perf_counter() - t0

            # a filter carrying trailer poses as states reports its own, which is the pose
            # the path-following algorithm and the controller would actually be given
            p_hat = (self.estimator.pose(guided) if hasattr(self.estimator, "pose")
                     else model.pose_array(x_hat)[:, guided])
            estimation_error = float(np.linalg.norm(
                model.pose_array(x)[:, guided][:2] - p_hat[:2]))
            pfa.update(p_hat[:2])
            ref = pfa.target

            t0 = _time.perf_counter()
            u, _ = self.controller(x_hat, u_prev, ref, np.zeros((0, 2, cfg.n_prediction)),
                                   np.ones(1), np.zeros(1))
            dt_c = _time.perf_counter() - t0

            x = model.simulate(x, self.ground.apply(u, k * self.ts))
            p_true = model.pose_array(x)[:, guided]

            states.append(x.copy())
            ests.append(np.asarray(x_hat, float).copy())
            poses.append(model.pose_array(x))
            controls.append(np.asarray(u, float).ravel())
            targets.append(ref)
            off.append(self._distance_to_path(p_true[:2]))
            eerr.append(estimation_error)
            te.append(dt_e)
            tc.append(dt_c)
            infeas.append(getattr(self.estimator, "feasibility_error", 0.0))
            u_prev = np.asarray(u, float).ravel()
            if progress:
                progress(k, n_steps)
            if pfa.finished:
                break

        states = np.array(states)
        return SimulationResult(
            time=np.arange(len(controls)) * self.ts, states=states,
            estimates=np.array(ests), poses=np.array(poses), controls=np.array(controls),
            targets=np.array(targets), off_track=np.array(off),
            estimation_error=np.array(eerr), joint_angles=states[:, 3:],
            t_estimator=np.array(te), t_controller=np.array(tc),
            infeasibility=np.array(infeas))

    def _distance_to_path(self, p) -> float:
        d = np.linalg.norm(self.path_points - np.asarray(p, float), axis=1)
        return float(d.min())
