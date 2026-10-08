# nmhec

Moving horizon estimation and predictive path following for generalised N-trailer vehicles.

A Python + CasADi re-implementation of

> N. Deniz, F. Jorquera, M. Torres-Torriti and F. Auat Cheein,
> **Model predictive path-following controller for Generalised N-Trailer vehicles with noisy
> sensors and disturbances**, *Control Engineering Practice* 142:105747, 2024.
> [doi:10.1016/j.conengprac.2023.105747](https://doi.org/10.1016/j.conengprac.2023.105747)

**[Open the demo in Colab](https://colab.research.google.com/github/nahueldeniz/nmhec/blob/main/notebooks/nmhec_demo.ipynb)**

This is an independent re-implementation written from the paper, not the code that produced
its results, and its numbers are not the paper's numbers. [`NOTES.md`](NOTES.md) says exactly
what is and is not reproduced.

## The point

A controller for a chained vehicle needs the pose of every segment. A real one measures the
joint angles and the pose of *one* segment, on ground that lets the wheels slip. Everything
else has to be inferred, fast enough to drive on.

The paper's state vector carries every segment's pose as a state of its own, even though the
chain determines them from the tractor's pose and the joint angles. That redundancy is where
the two estimators part company:

* the **moving horizon estimator** writes the chain relations as equality constraints, so
  every estimate it produces is a configuration the vehicle could actually hold;
* an **extended Kalman filter** has no such mechanism. Its correction pulls the whole state
  towards the measurement, and nothing in it knows that the last trailer has to stay on its
  hitch.

Under slip the filter does not merely become inaccurate. It becomes *infeasible*: it reports a
vehicle whose trailers have come off. `feasibility_error()` measures that in metres.

Closed loop on a lemniscate, 2 trailers, `N_e = 10`, `T_s = 0.1 s`, 3 seeds — the controller
driven by the estimate throughout, except for `oracle`, which is handed the true state and is
a yardstick rather than a method:

| slip `rho` | estimator | estimation error [m] | infeasibility [m] | off-track [m] |
|---|---|---|---|---|
| 1.00 | oracle | 0.000 | 0.000 | 0.162 |
| 1.00 | ekf    | **0.002** | 0.075 | 0.201 |
| 1.00 | nmhe   | 0.084 | **0.000** | 0.235 |
| 0.90 | ekf    | 0.266 | 0.271 | 0.316 ± 0.055 |
| 0.90 | nmhe   | **0.084** | **0.000** | 0.243 ± 0.013 |
| 0.50 | ekf    | 3.485 | 3.417 | 0.302 ± 0.147 |
| 0.50 | nmhe   | **0.086** | **0.000** | 0.218 ± 0.018 |

On firm ground the filter wins, and it is not close — it should, being optimal for Gaussian
noise on a model with no mismatch, at a fortieth of the cost. It loses when the disturbance
stops being small and Gaussian. Note also that the two middle columns move together: the
filter's error *is* its infeasibility.

## Install

```bash
pip install git+https://github.com/nahueldeniz/nmhec.git
```

## Use

```python
import numpy as np
from nmhec import ClosedLoop, EstimatorConfig, Ground, PFAConfig, SensorConfig, VehicleGeometry
from nmhec.path import lemniscate

path = lemniscate(scale=0.6).sample(600)[:, :2]

loop = ClosedLoop(
    path, VehicleGeometry(n_trailers=2), ts=0.1,
    estimator="nmhe",                                  # "ekf", "ekf_minimal", "oracle"
    sensors=SensorConfig(pose_segments=(0,)),          # (2,) puts the GPS on the last trailer
    ground=Ground(rho_omega=0.9, rho_v=0.9),           # Eq. (2); 1.0 is firm ground
    est_cfg=EstimatorConfig(n_estimation=10),
    pfa_cfg=PFAConfig(d_tgt=0.3, d_ntgt=0.3),
)
result = loop.run(300)
print(result.summary(0.1))
```

Estimation on its own, without a controller in the way:

```bash
python examples/slip_study.py --ne 10
```

## What is in it

| | |
|---|---|
| `model.py` | GNT kinematics with off-axle hitching, RK4, segment poses |
| `redundant.py` | the paper's redundant state `q`, and `feasibility_error()` |
| `sensing.py` | the measurement equation (Eq. 3) and the slippage of Eq. (2) |
| `estimator.py` | the NMHE, and an EKF on the minimal state |
| `pfa.py` | the path-following algorithm of Section 2.3 (`d_tgt <= D_tgt`) |
| `controller.py` | the NMPC, computing the tractor's velocities directly |
| `sim.py` | the closed loop, with the estimator inside it |
| `viz.py` | plots built around what the estimator is doing |

`model.py`, `path.py` and `controller.py` are shared with
[nmppfc](https://github.com/nahueldeniz/nmppfc) (paper 02) and kept byte-identical below their
headers so fixes can be ported between the two.

## Tests

```bash
pip install -e ".[dev]" && pytest
```

They pin the things that fail silently: a pose is `(x, y, theta)` but the controller's error
vector and weights are `(theta, x, y)`; the redundant and minimal models must agree after
integration; the NMHE's estimate must stay feasible.

## Licence

MIT. If you use it, cite the paper — see [`CITATION.cff`](CITATION.cff).
