"""The path-following algorithm of Section 2.3: a target that is carried along the path.

This is *not* the reference generator of paper 02, which moves the target at a commanded
speed. Here the target waits for the vehicle. The perpendicular to the path at the current
target divides the plane; the guided segment's distance to that line is

    d_tgt = |A x + B y + C| / sqrt(A^2 + B^2),    (A, B) the path tangent at the target,

and the target only advances once ``d_tgt <= D_tgt``. The next target is the first sample
further along the path whose accumulated distance from the current one reaches ``D_ntgt``.
Small values of the two give a finely detailed reference; large values let the vehicle cut
corners. Both are parameters of the algorithm, exactly as in the paper.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PFAConfig:
    d_tgt: float = 0.65         # D_tgt [m], the field trials' value
    d_ntgt: float = 0.65        # D_ntgt [m]
    guided_segment: int = -1    # -1 = the last trailer, which is what the paper guides
    lookahead_heading: bool = True   # reference heading = path tangent at the target


class TargetUpdater:
    """Carries a target along a sampled path, advancing it as the vehicle closes in."""

    def __init__(self, path_points, cfg: PFAConfig = PFAConfig(), start_index: int = 0,
                 closed: bool = True):
        pts = np.asarray(path_points, dtype=float)[:, :2]
        if len(pts) < 3:
            raise ValueError("a path needs at least three samples")
        self.pts = pts
        self.cfg = cfg
        self.closed = bool(closed)
        self.index = int(start_index)
        self.laps = 0
        seg = np.diff(pts, axis=0, append=pts[:1] if closed else pts[-1:])
        norms = np.linalg.norm(seg, axis=1, keepdims=True)
        self.tangent = seg / np.maximum(norms, 1e-12)
        self.step_length = norms.ravel()

    # ------------------------------------------------------------------ state
    @property
    def target(self) -> np.ndarray:
        """Current target pose (x, y, theta)."""
        i = self.index % len(self.pts)
        t = self.tangent[i]
        theta = np.arctan2(t[1], t[0]) if self.cfg.lookahead_heading else 0.0
        return np.array([self.pts[i, 0], self.pts[i, 1], theta])

    def distance_to_target(self, position) -> float:
        """d_tgt: distance from the guided segment to the perpendicular at the target."""
        i = self.index % len(self.pts)
        a, b = self.tangent[i]
        c = -(a * self.pts[i, 0] + b * self.pts[i, 1])
        x, y = float(position[0]), float(position[1])
        return abs(a * x + b * y + c) / max(np.hypot(a, b), 1e-12)

    # ---------------------------------------------------------------- advance
    def update(self, position) -> bool:
        """Advance the target if the vehicle has reached it. Returns True if it moved."""
        if self.distance_to_target(position) > self.cfg.d_tgt:
            return False
        travelled, i = 0.0, self.index
        while travelled < self.cfg.d_ntgt:
            travelled += self.step_length[i % len(self.pts)]
            i += 1
            if not self.closed and i >= len(self.pts) - 1:
                i = len(self.pts) - 1
                break
        if self.closed and i // len(self.pts) > self.index // len(self.pts):
            self.laps += 1
        self.index = i
        return True

    @property
    def finished(self) -> bool:
        return not self.closed and self.index >= len(self.pts) - 1
