"""Estimation under slippage: the moving horizon estimator against an extended Kalman filter.

Open loop -- a fixed input sequence, so the comparison is about estimation alone. The ground
scales the commanded velocities (Eq. 2); neither estimator is told the slip coefficients.

    python examples/slip_study.py --trailers 2 --ne 10
"""
import argparse
import numpy as np

from nmppfc.model import NTrailer, VehicleGeometry
from nmhec.sensing import Sensors, SensorConfig, Ground
from nmhec.estimator import NMHE, EKF, EstimatorConfig


def run(rho, n_trailers=2, ne=10, steps=120, ts=0.1, seed=1):
    model = NTrailer(VehicleGeometry(n_trailers=n_trailers), ts=ts)
    sensors = Sensors(model, SensorConfig())
    ground = Ground(rho_omega=rho, rho_v=rho)
    nmhe = NMHE(model, sensors, EstimatorConfig(n_estimation=ne))
    ekf = EKF(model, sensors)
    rng = np.random.default_rng(seed)

    x = model.state_from_pose(0.0, 0.0, 0.0, [0.05, -0.03][:n_trailers])
    errors = {"nmhe": [], "ekf": []}
    times = {"nmhe": [], "ekf": []}
    u_prev = None
    for k in range(steps):
        y = sensors.measure(x, rng)
        x_nmhe, x_ekf = nmhe(y, u_prev), ekf(y, u_prev)
        if k > ne:                                  # let both windows fill first
            errors["nmhe"].append(x_nmhe - x)
            errors["ekf"].append(x_ekf - x)
            times["nmhe"].append(nmhe.last_solve)
            times["ekf"].append(ekf.last_solve)
        u = np.array([0.35 * np.sin(0.05 * k), 0.5])
        x = model.simulate(x, ground.apply(u))      # the ground gets a say
        u_prev = u                                  # the estimators only know the command
    out = {}
    for name, e in errors.items():
        e = np.array(e)
        out[name] = {"position": float(np.sqrt((e[:, :2] ** 2).sum(1)).mean()),
                     "joint_angles": float(np.abs(e[:, 3:]).mean()),
                     "solve_ms": 1e3 * float(np.mean(times[name]))}
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trailers", type=int, default=2)
    p.add_argument("--ne", type=int, default=10, help="estimation window [samples]")
    p.add_argument("--steps", type=int, default=120)
    p.add_argument("--rho", type=float, nargs="+", default=[1.0, 0.9, 0.75, 0.5])
    a = p.parse_args()

    print(f"{'rho':>5} | {'NMHE pos [m]':>12} {'beta [rad]':>11} {'ms':>6} "
          f"| {'EKF pos [m]':>12} {'beta [rad]':>11} {'ms':>6}")
    for rho in a.rho:
        r = run(rho, a.trailers, a.ne, a.steps)
        print(f"{rho:5.2f} | {r['nmhe']['position']:12.3f} {r['nmhe']['joint_angles']:11.4f} "
              f"{r['nmhe']['solve_ms']:6.1f} | {r['ekf']['position']:12.3f} "
              f"{r['ekf']['joint_angles']:11.4f} {r['ekf']['solve_ms']:6.2f}")


if __name__ == "__main__":
    main()
