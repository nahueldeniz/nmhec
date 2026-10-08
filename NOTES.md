# Implementation notes

This package was written from the paper. It is **not** the code that produced the paper's
results, those workspaces were not available, and nothing here should be read as a statement
about what the authors ran. Where the paper is ambiguous, the choice made here is recorded
below with its reasoning.

## 1. What is reproduced, and what is not

Reproduced: the GNT kinematics, the moving horizon estimator with an adaptive arrival cost and
constrained joint angles, the path-following target updater, the predictive controller acting
on the tractor's velocities, and the closed loop in which the controller only ever sees the
estimate.

Not reproduced:

* **The field experiments.** No logs, so the G2T trials — including the scaled Monaco circuit —
  cannot be replayed. Everything here is simulation.
* **The paper's tables.** Different scenarios, disturbance realisations, tuning, solver
  (IPOPT here, ACADO there) and hardware. The numbers in `README.md` are this repository's,
  and they are not comparable with Tables 1–5.
* **The competitor.** The paper compares against Michałek and Kiełczewski (2017). Nothing here
  implements that control law. The EKF in this repository is a baseline of my own
  construction; it demonstrates that *a Kalman filter on the redundant state* fails under slip,
  and says nothing about anyone else's method.

## 2. The state vectors, and why there are two

Section 2.1 carries `q = (beta_i, theta_i, x_i, y_i)`: every segment's pose is a state, even
though the chain determines it from `(x_0, y_0, theta_0)` and the joint angles.

* `model.py` works on the **minimal** state `x = [x_0, y_0, theta_0, beta_1 .. beta_N]` and
  recovers the poses algebraically. Compact, and feasible by construction.
* `redundant.py` works on `q`, as the paper writes it.

This distinction is the repository's main finding and it is not cosmetic. An estimator on the
minimal state *cannot* produce an infeasible configuration, so an EKF written that way is much
stronger than the one the paper compares against, and the failure the paper reports does not
appear. The EKF reached by `estimator="ekf"` is therefore the redundant one;
`estimator="ekf_minimal"` is kept for contrast, and the difference between them is large.

`feasibility_error()` reconstructs the chain from `(x_0, y_0, theta_0, beta)` and returns the
distance to the filter's own last-trailer pose. It is zero for any real configuration.

## 3. Disturbances

Slippage (Eq. 2) scales the commanded velocities: `u_eff = (rho_1 omega, rho_2 v)`. It is
applied in `Ground`, inside the plant, and nothing downstream is told `rho`. To both
estimators that mismatch is simply a disturbance — bounded, and with no assumed distribution,
which is the property the paper argues for.

In the NMHE the disturbance enters on the **input channel** (`nw = 2`), rather than additively
on the state. That matches Eq. (2), where `phi` is `G(q)` times a perturbation of `u`, and it
keeps the NLP small.

## 4. Choices the paper leaves open

* **Joint-angle bound: ±(π/2 − δ_β) = ±1.396 rad**, from Section 3.2.1 with δ_β = 10π/180.
  Do not take 1.745 from the vertical axis of Fig. 15 — that is a tick label. The red
  constraint lines in that figure measure about ±1.57 against the figure's own axis, closer to
  π/2, and the caption does not say which the lines are. The two readings disagree and I could
  not resolve it from the PDF. 1.396 is what the text states, and it is also the value at which
  the chain tracks a tight path without folding; at 1.745 it jackknifes.
* **The guided segment.** The controller here weights the last trailer only. The paper's
  stage cost also carries the tractor's coordinates, with greater emphasis on the last trailer.
* **Controller tuning.** `r = (0.02, 0.05)`, `v_max = 0.8`, `D_tgt = 0.3 m` are chosen so the
  chain tracks the demo path without jackknifing. The resulting ~0.23 m/s is close to the
  0.28 m/s of the field trials, and slow for the same reason: the cost trades speed for
  deviation. These are not the paper's weights.
* **`D_tgt` dominates the off-track metric.** At large `D_tgt` the vehicle cuts corners and the
  path error stops being a measure of the estimator at all. The off-track column of
  `README.md` should be read with that in mind; the estimation and infeasibility columns are
  the ones that say something about estimation.

## 5. Known limitations

1. Only the simulated vehicle. No sensor latency, no asynchronous measurements, no dropout
   beyond the outlier model of scenario (b).
2. The real-time factors are this machine's with IPOPT and a warm start, not a real-time
   target. They are useful as a trend against `N_e`, not as an absolute.
3. The arrival cost follows Sánchez, Murillo and Giovanini (2017); the vendored implementation
   comes from the [smhec](https://github.com/nahueldeniz/smhec) repository (paper 01).
4. Seeds are fixed so runs are reproducible; the spread across seeds is reported where it
   matters, because at low slip it exceeds the difference between the estimators.
