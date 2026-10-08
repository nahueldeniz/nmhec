# Shared with the nmppfc package (paper 02, IEEE T-ASE 2025): this file is a copy,
# kept byte-identical below the header so fixes can be ported between the two repos.
# Source: github.com/nahueldeniz/nmppfc
"""Parametric reference paths and the reference generator (GPPM).

A path is a curve ``p(gamma) = (x, y, theta)`` in the x-y-theta space. The path parameter
advances with a virtual input, ``gamma_dot = sigma``, so the reference moves along the curve
at a commanded speed; the reference generator slows it down (``sigma_occluded``) while the
line of sight to the reference is blocked, as in the paper.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class Path:
    """Parametric path. ``xy(gamma)`` returns a point; the heading is differentiated."""

    xy: Callable[[float], tuple]
    gamma_range: tuple = (0.0, 2.0 * np.pi)
    name: str = "path"
    closed: bool = True

    def point(self, gamma: float) -> np.ndarray:
        g = self._wrap(gamma)
        x, y = self.xy(g)
        return np.array([float(x), float(y)])

    def pose(self, gamma: float, dg: float = 1e-4) -> np.ndarray:
        """Reference pose r_c = (x, y, theta) at the path parameter ``gamma``."""
        a, b = self.point(gamma - dg), self.point(gamma + dg)
        p = self.point(gamma)
        return np.array([p[0], p[1], float(np.arctan2(b[1] - a[1], b[0] - a[0]))])

    def sample(self, n: int = 600) -> np.ndarray:
        g = np.linspace(self.gamma_range[0], self.gamma_range[1], n)
        return np.array([self.point(gi) for gi in g])

    def project(self, point, n: int = 2000) -> float:
        """Path parameter of the closest point on the path (used for the metrics)."""
        g = np.linspace(self.gamma_range[0], self.gamma_range[1], n)
        pts = np.array([self.point(gi) for gi in g])
        return float(g[int(np.argmin(np.hypot(pts[:, 0] - point[0], pts[:, 1] - point[1])))])

    def distance(self, point, n: int = 2000) -> float:
        g = self.project(point, n)
        p = self.point(g)
        return float(np.hypot(p[0] - point[0], p[1] - point[1]))

    def _wrap(self, gamma: float) -> float:
        lo, hi = self.gamma_range
        if self.closed:
            return lo + (gamma - lo) % (hi - lo)
        return float(np.clip(gamma, lo, hi))


def lemniscate(scale: float = 1.0, centre=(0.0, 0.0)) -> Path:
    """Lemniscate of Bernoulli, the path used in the paper's simulation study."""
    def xy(g):
        d = np.sin(g) ** 2 + 1.0
        return (centre[0] + scale * np.sqrt(32) * np.cos(g) / d,
                centre[1] + scale * np.sqrt(128) * np.cos(g) * np.sin(g) / d)
    return Path(xy=xy, gamma_range=(0.0, 2 * np.pi), name="lemniscate", closed=True)


def line(length: float = 25.0, start=(0.0, 6.0), angle: float = 0.0) -> Path:
    """Straight segment, as in the paper's first field experiment."""
    def xy(g):
        return (start[0] + g * length * np.cos(angle), start[1] + g * length * np.sin(angle))
    return Path(xy=xy, gamma_range=(0.0, 1.0), name="line", closed=False)


def sine(a: float = 7.5, offset=(6.0, 4.5)) -> Path:
    """x = a sin(g) + x0,  y = 7.5 sin(g)^2 cos(g) + y0  (paper, Eq. 19)."""
    def xy(g):
        return (a * np.sin(g) + offset[0], 7.5 * np.sin(g) ** 2 * np.cos(g) + offset[1])
    return Path(xy=xy, gamma_range=(0.0, np.pi), name="sine", closed=False)


LEMNISCATE, LINE, SINE = lemniscate, line, sine


class ReferenceGenerator:
    """Moves a target along the path at a commanded speed (the GPPM of the paper).

    The path is sampled once and re-parameterised by arc length, so the target advances by
    exactly ``speed * Ts`` metres per step regardless of how the parametric equation is
    sampled. The target slows to ``speed * slow_factor`` while the line of sight to it is
    blocked, which is the behaviour the paper describes for an occluded reference, and its
    speed can ramp up over ``ramp_time`` seconds at the start. The reference heading is the
    direction of travel along the path.
    """

    def __init__(self, path: Path, speed: float = 0.6, slow_factor: float = 0.4,
                 n_samples: int = 8000, start_distance: float = 0.0, ramp_time: float = 0.0):
        self.path = path
        self.speed = float(speed)
        self.slow_factor = float(slow_factor)
        self.ramp_time = float(ramp_time)
        pts = path.sample(n_samples)
        d = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1])))])
        self._pts, self._d = pts, d
        self.length = float(d[-1])
        self.distance = float(start_distance)
        self._elapsed = 0.0
        self.last_target = False

    # ------------------------------------------------------------------ target
    def _commanded_speed(self, occluded: bool) -> float:
        v = self.speed
        if self.ramp_time > 0:
            v *= min(1.0, self._elapsed / self.ramp_time)
        return v * (self.slow_factor if occluded else 1.0)

    def _at(self, distance: float) -> np.ndarray:
        d = distance % self.length if self.path.closed else np.clip(distance, 0, self.length)
        x = np.interp(d, self._d, self._pts[:, 0])
        y = np.interp(d, self._d, self._pts[:, 1])
        step = max(self.length * 1e-4, 1e-3)
        a = np.array([np.interp((d - step) % self.length if self.path.closed
                                else max(d - step, 0.0), self._d, self._pts[:, 0]),
                      np.interp((d - step) % self.length if self.path.closed
                                else max(d - step, 0.0), self._d, self._pts[:, 1])])
        b = np.array([np.interp((d + step) % self.length if self.path.closed
                                else min(d + step, self.length), self._d, self._pts[:, 0]),
                      np.interp((d + step) % self.length if self.path.closed
                                else min(d + step, self.length), self._d, self._pts[:, 1])])
        return np.array([x, y, float(np.arctan2(b[1] - a[1], b[0] - a[0]))])

    def pose(self) -> np.ndarray:
        """Current target pose (x, y, theta)."""
        return self._at(self.distance)

    def advance(self, ts: float, occluded: bool = False) -> np.ndarray:
        """Move the target at the commanded speed and return its pose."""
        self._elapsed += ts
        self.distance += self._commanded_speed(occluded) * ts
        if not self.path.closed and self.distance >= self.length:
            self.distance = self.length
            self.last_target = True
        return self.pose()

    def finished(self) -> bool:
        return self.last_target
