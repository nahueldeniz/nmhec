# Shared with the nmppfc package (paper 02, IEEE T-ASE 2025): this file is a copy,
# kept byte-identical below the header so fixes can be ported between the two repos.
# Source: github.com/nahueldeniz/nmppfc
"""Nonlinear model predictive path-following controllers for GNT vehicles.

``NMPPFC``              obstacles as soft constraints (Gaussian terms in the cost), Eq. (13).
``HardConstraintMPC``   the same path-following problem with obstacles as hard constraints,
                        used here as the FNMPPC-style baseline.

Both build their NLP once, with CasADi/IPOPT, and take the current state, the reference over
the prediction window and the forecast obstacle centres as parameters, so nothing is rebuilt
while the loop runs. Multiple shooting; a fourth-order Runge-Kutta integrator inside.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import casadi as ca
import numpy as np

from .model import NTrailer


@dataclass
class ControllerConfig:
    """Tuning of the predictive controller.

    ``q`` and ``qN`` weight the deviation of the guided segment in the order
    ``(theta, x, y)`` -- the order the cost is written in, with the heading first.
    """

    n_control: int = 15              # N_c
    n_prediction: int = 15           # N_p  (>= N_c; inputs are held after N_c)
    q: tuple = (1.0, 10.0, 10.0)     # stage weights on (theta, x, y) of the guided segment
    qN: tuple = (1.0, 10.0, 10.0)    # terminal weights, same order
    r: tuple = (0.05, 0.1)           # weights on (omega, v)
    guided_segment: int = 0          # 0 = tractor
    v_max: float = 1.0
    w_max: float = 2.0
    dv_max: float = 3.0              # [m/s^2]
    dw_max: float = 6.0              # [rad/s^2]
    beta_max: float = 120 * np.pi / 180
    """Bound on the joint angles. Eq. (13) of the paper writes pi/2 - delta_beta (70 deg with
    delta_beta = 20 deg); the implementation that produced the results uses 120 deg, which is
    what this defaults to."""
    link_samples: int = 4            # points per link at which the obstacle cost is evaluated
    stage_growth: float = 0.1        # r of Eq. (15); 0 keeps the stage weights flat
    l_min: float = 0.707             # J_0min, the lower bound used to derive c
    l_max: float = 7.847             # J_0max, the upper bound used to derive c
    omega_half_width: float = 0.25   # half side of the box Omega around the reference [m]
    max_iter: int = 150
    warm_start: bool = True


class _BaseMPC:
    """Shared machinery: decision variables, dynamics, input and joint-angle constraints."""

    def __init__(self, model: NTrailer, cfg: ControllerConfig, n_obstacles: int):
        self.model = model
        self.cfg = cfg
        self.n_obs = int(n_obstacles)
        self.nx, self.nu = model.nx, model.nu
        self.nseg = model.geo.n_segments
        self._last_solution = None
        self.last_solve_time = 0.0
        self.last_status = "not solved"
        self._build()

    # ------------------------------------------------------------------- build
    def _build(self):
        cfg, nx, nu = self.cfg, self.nx, self.nu
        Nc, Np = cfg.n_control, cfg.n_prediction
        U = ca.SX.sym("U", nu, Nc)
        X = ca.SX.sym("X", nx, Np + 1)

        x0 = ca.SX.sym("x0", nx)
        u_prev = ca.SX.sym("u_prev", nu)
        ref = ca.SX.sym("ref", 3)                              # one pose, held over the window
        obs = ca.SX.sym("obs", 2 * max(self.n_obs, 1), Np)     # centres, stacked per obstacle
        rho = ca.SX.sym("rho", max(self.n_obs, 1))
        amp = ca.SX.sym("amp", max(self.n_obs, 1))
        w_stage = ca.SX.sym("w_stage", Np)

        Q = ca.diag(ca.DM(cfg.q))
        QN = ca.diag(ca.DM(cfg.qN))
        R = ca.diag(ca.DM(cfg.r))

        g, lbg, ubg = [X[:, 0] - x0], [ca.DM.zeros(nx)], [ca.DM.zeros(nx)]
        cost = 0
        for k in range(Np):
            uk = U[:, min(k, Nc - 1)]
            g += [X[:, k + 1] - self.model.step(X[:, k], uk)]
            lbg += [ca.DM.zeros(nx)]
            ubg += [ca.DM.zeros(nx)]

            poses = self.model.poses(X[:, k + 1])
            pg = poses[:, cfg.guided_segment]
            e = ca.vertcat(ca.atan2(ca.sin(pg[2] - ref[2]), ca.cos(pg[2] - ref[2])),
                           pg[0] - ref[0], pg[1] - ref[1])
            W = QN if k == Np - 1 else Q
            stage = ca.mtimes([e.T, W, e]) + ca.mtimes([uk.T, R, uk])
            stage += self._obstacle_stage(poses, obs, rho, amp, k, g, lbg, ubg)
            cost += w_stage[k] * stage

            # rate limits on the inputs (u_{-1} is the control applied at the previous step)
            du = (uk - (u_prev if k == 0 else U[:, min(k - 1, Nc - 1)])) / self.model.ts
            if k < Nc:
                g += [du]
                lbg += [ca.DM([-cfg.dw_max, -cfg.dv_max])]
                ubg += [ca.DM([cfg.dw_max, cfg.dv_max])]

            # joint angles: |beta| <= beta_max (no jackknifing)
            g += [X[3:, k + 1]]
            lbg += [ca.DM([-cfg.beta_max] * self.model.geo.n_trailers)]
            ubg += [ca.DM([cfg.beta_max] * self.model.geo.n_trailers)]

        z = ca.vertcat(ca.reshape(U, -1, 1), ca.reshape(X, -1, 1))
        p = ca.vertcat(x0, u_prev, ref, ca.reshape(obs, -1, 1), rho, amp, w_stage)
        nlp = {"x": z, "f": cost, "g": ca.vertcat(*g), "p": p}
        opts = {"ipopt.print_level": 0, "print_time": False, "ipopt.sb": "yes",
                "ipopt.max_iter": cfg.max_iter, "ipopt.tol": 1e-4,
                "ipopt.acceptable_tol": 1e-3, "ipopt.acceptable_iter": 5}
        if cfg.warm_start:
            opts.update({"ipopt.warm_start_init_point": "yes"})
        self.solver = ca.nlpsol("mpc", "ipopt", nlp, opts)

        self.lbg = np.concatenate([np.array(v).ravel() for v in lbg])
        self.ubg = np.concatenate([np.array(v).ravel() for v in ubg])
        self.lbz = np.concatenate([np.tile([-cfg.w_max, -cfg.v_max], Nc),
                                   -np.inf * np.ones(nx * (Np + 1))])
        self.ubz = np.concatenate([np.tile([cfg.w_max, cfg.v_max], Nc),
                                   np.inf * np.ones(nx * (Np + 1))])

    def _link_points(self, poses):
        """Points at which the obstacle cost is evaluated: sampled along each link.

        The Gaussian is evaluated not only at the segment centres but at ``link_samples``
        points spread along every link of the chain, so an obstacle cannot slip between two
        segments without being seen by the cost.
        """
        pts = [(poses[0, 0], poses[1, 0])]
        for j in range(1, self.nseg):
            ax, ay = poses[0, j - 1], poses[1, j - 1]
            bx, by = poses[0, j], poses[1, j]
            n = max(self.cfg.link_samples, 1)
            for i in range(1, n + 1):
                l = i / n
                pts.append((ax + (bx - ax) * l, ay + (by - ay) * l))
        return pts

    def _obstacle_stage(self, poses, obs, rho, amp, k, g, lbg, ubg):
        raise NotImplementedError

    # ------------------------------------------------------------------- solve
    def c_stab(self) -> float:
        """``c = r * J_0min / (N_p * J_0max)``, the constant of Eq. (15)."""
        cfg = self.cfg
        return cfg.stage_growth * cfg.l_min / (cfg.n_prediction * cfg.l_max)

    def stage_weights(self, inside_omega: bool = True, c: float | None = None) -> np.ndarray:
        """Coefficients p_n of Eq. (15).

        Inside the set Omega around the reference every stage carries the same weight ``c``,
        derived from the bounds on the nominal cost; outside it the weights grow as
        ``1 + (n - 1) r``. This is what makes the scheme practically stable without requiring
        the terminal cost to be a control Lyapunov function.
        """
        Np, r = self.cfg.n_prediction, self.cfg.stage_growth
        if r <= 0:
            return np.ones(Np)
        if inside_omega:
            return np.full(Np, self.c_stab() if c is None else c)
        return 1.0 + r * np.arange(Np)

    def __call__(self, x, u_prev, ref, obs_forecast, rho, amp, stage_weights=None):
        """One control step.

        ``ref`` is a single pose ``(x, y, theta)``, held over the whole window -- the
        reference selector decides what it is. The weights ``q``/``qN`` are applied to the
        error in the order ``(theta, x, y)``.
        """
        cfg, Nc, Np = self.cfg, self.cfg.n_control, self.cfg.n_prediction
        if self.n_obs:
            # (2, n_obs, Np) -> rows [x_0; y_0; x_1; y_1; ...] as the symbols expect
            obs = np.asarray(obs_forecast).transpose(1, 0, 2).reshape(2 * self.n_obs, Np)
            rho_v, amp_v = np.asarray(rho, float), np.asarray(amp, float)
        else:  # placeholder values; the terms are multiplied by a zero amplitude
            obs, rho_v, amp_v = np.zeros((2, Np)), np.ones(1), np.zeros(1)
        w = self.stage_weights() if stage_weights is None else np.asarray(stage_weights, float)
        p = np.concatenate([np.asarray(x, float).ravel(), np.asarray(u_prev, float).ravel(),
                            np.asarray(ref, float).ravel(),
                            obs.reshape(-1, order="F"), rho_v, amp_v, w])

        best, best_cost, status = None, np.inf, "no solve"
        t0 = time.perf_counter()
        for z0 in self.initial_guesses(x, u_prev):
            sol = self.solver(x0=z0, p=p, lbx=self.lbz, ubx=self.ubz,
                              lbg=self.lbg, ubg=self.ubg)
            stats = self.solver.stats()
            status = stats.get("return_status", "unknown")
            if stats.get("success", False) and float(sol["f"]) < best_cost:
                best, best_cost = np.array(sol["x"]).ravel(), float(sol["f"])
                self.last_status = status
        self.last_solve_time = time.perf_counter() - t0

        ok = best is not None
        if ok:
            self._last_solution = best
            z = best
        else:
            self.last_status = status
            z = np.concatenate([np.tile(np.asarray(u_prev, float).ravel(), Nc),
                                np.tile(np.asarray(x, float).ravel(), Np + 1)])
        # a failed solve stops the vehicle rather than repeating the last input
        u = z[:self.nu] if ok else np.zeros(self.nu)
        plan = z[self.nu * Nc:].reshape(self.nx, Np + 1, order="F")
        return u, {"ok": bool(ok), "status": self.last_status, "cpu": self.last_solve_time,
                   "plan": plan, "cost": best_cost if ok else float("inf")}

    def initial_guesses(self, x, u_prev):
        """Initial guesses handed to IPOPT, tried in order (the best solution is kept)."""
        Nc, Np = self.cfg.n_control, self.cfg.n_prediction
        if self._last_solution is not None and self.cfg.warm_start:
            return [self._last_solution]
        return [np.concatenate([np.zeros(self.nu * Nc),
                                np.tile(np.asarray(x, float).ravel(), Np + 1)])]

    def _rollout_guess(self, x, u):
        """Build a feasible-by-construction guess by rolling the model out under ``u``."""
        Nc, Np = self.cfg.n_control, self.cfg.n_prediction
        xs, xi = [np.asarray(x, float).ravel()], np.asarray(x, float).ravel()
        for _ in range(Np):
            xi = self.model.simulate(xi, u)
            xs.append(xi)
        return np.concatenate([np.tile(np.asarray(u, float).ravel(), Nc),
                               np.concatenate(xs)])


class NMPPFC(_BaseMPC):
    """Obstacles enter the cost as Gaussian functions: the feasible set stays convex."""

    def _obstacle_stage(self, poses, obs, rho, amp, k, g, lbg, ubg):
        if not self.n_obs:
            return 0
        term = 0
        pts = self._link_points(poses)
        for j in range(self.n_obs):
            ox, oy = obs[2 * j, k], obs[2 * j + 1, k]
            for px, py in pts:
                d2 = (px - ox) ** 2 + (py - oy) ** 2
                term += amp[j] * ca.exp(-d2 / (2 * rho[j] ** 2))
        return term


class HardConstraintMPC(_BaseMPC):
    """Baseline: obstacles as hard constraints, which makes the feasible set non-convex.

    A collision constraint is imposed for every segment, every obstacle and every step of the
    prediction window. When IPOPT cannot find a feasible point the previous input is held,
    which is what a real deployment would do; those steps are reported in the result.

    With a single warm start this controller stops at the constraint boundary instead of
    going around an obstacle: stopping is a perfectly good local solution of a non-convex
    problem. To keep the comparison fair it is solved from several initial guesses (a left
    and a right detour besides the warm start) and the cheapest feasible solution is kept;
    ``multi_start=False`` shows the single-start behaviour.
    """

    multi_start = True

    def initial_guesses(self, x, u_prev):
        guesses = super().initial_guesses(x, u_prev)
        if not self.multi_start:
            return guesses
        v = 0.5 * self.cfg.v_max
        for w in (0.5 * self.cfg.w_max, -0.5 * self.cfg.w_max):
            guesses.append(self._rollout_guess(x, [w, v]))
        return guesses

    def _obstacle_stage(self, poses, obs, rho, amp, k, g, lbg, ubg):
        if not self.n_obs:
            return 0
        pts = self._link_points(poses)
        for j in range(self.n_obs):
            ox, oy = obs[2 * j, k], obs[2 * j + 1, k]
            for px, py in pts:
                d2 = (px - ox) ** 2 + (py - oy) ** 2
                # rho carries the clearance radius for this controller (see Simulator)
                g.append(d2 - rho[j] ** 2)
                lbg.append(ca.DM([0.0]))
                ubg.append(ca.DM([ca.inf]))
        return 0
