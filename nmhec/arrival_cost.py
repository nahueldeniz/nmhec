"""Adaptive arrival cost (Sánchez, Murillo & Giovanini, ISA Trans. 2017, Eq. 13).

The arrival cost is  || x_{k-N|k} - xbar ||^2_{P^{-1}}, where
  * xbar is the smoothed estimate of the new window start from the previous solution, and
  * P is updated by recursive least squares with a variable forgetting factor
    (Fortescue et al., 1981), with a trace bound that keeps P from blowing up.
"""
import numpy as np


class AdaptiveArrivalCost:
    def __init__(self, xbar0, P0, sigma=1.0, c=None, alpha_min=0.05):
        self.xbar = np.atleast_1d(np.asarray(xbar0, dtype=float))
        self.P = np.atleast_2d(np.asarray(P0, dtype=float))
        self.sigma = sigma                       # sigma / (process-noise variance) should be large
        self.c = c if c is not None else 10 * np.trace(self.P)   # bound on trace(P)
        self.alpha_min = alpha_min               # safeguard: alpha stays in [alpha_min, 1]
        self.alpha = 1.0                         # last forgetting factor (for inspection)

    @property
    def weight(self):
        """P^{-1}, the weight of the arrival cost."""
        return np.linalg.inv(self.P)

    def update(self, x_start, residual):
        """Shift the window: new prior x_start (smoothed estimate), RLS update of P (Eq. 13).

        x_start:  estimate of the new window start, x_{k-N+1|k}
        residual: output error at that sample, y - h(x_start)
        """
        phi = np.atleast_1d(np.asarray(x_start, dtype=float))
        eps2 = float(np.sum(np.square(residual))) + 1e-12
        gain = 1.0 + phi @ self.P @ phi
        n_k = gain * self.sigma / eps2                          # memory length
        self.alpha = float(np.clip(1.0 - 1.0 / n_k, self.alpha_min, 1.0))
        W = (np.eye(len(phi)) - np.outer(self.P @ phi, phi) / gain) @ self.P
        self.P = W / self.alpha if np.trace(W) / self.alpha <= self.c else W
        self.xbar = phi
