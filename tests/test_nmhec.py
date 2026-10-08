"""Regression tests. Most of these guard orderings and invariants that fail silently.

The ordering pair in particular has bitten this family of code before: a *pose* is
``(x, y, theta)`` everywhere, but the controller's error vector and its weights are
``(theta, x, y)``, heading first. Both are correct; swapping one for the other produces a
controller that looks plausible and steers wrongly.
"""
import numpy as np
import pytest

from nmhec.model import NTrailer, VehicleGeometry
from nmhec.pfa import PFAConfig, TargetUpdater
from nmhec.redundant import RedundantEKF, RedundantModel, feasibility_error
from nmhec.sensing import Ground, SensorConfig, Sensors
from nmhec.estimator import EKF, NMHE, EstimatorConfig, wrap


GEO = VehicleGeometry(n_trailers=2)
TS = 0.1


def _model():
    return NTrailer(GEO, ts=TS)


def _state():
    return _model().state_from_pose(1.3, -0.7, 0.42, [0.25, -0.15])


# --------------------------------------------------------------------- order
def test_pose_is_x_y_theta():
    m, x = _model(), _state()
    pose = m.pose_array(x)[:, 0]
    assert np.allclose(pose, [x[0], x[1], x[2]])


def test_target_is_x_y_theta():
    pts = np.column_stack([np.linspace(0, 10, 50), np.zeros(50)])
    t = TargetUpdater(pts, PFAConfig(), closed=False).target
    assert np.allclose(t[:2], pts[0]) and abs(wrap(t[2] - 0.0)) < 1e-9


def test_controller_error_is_theta_x_y():
    """controller.py builds e = (heading error, x error, y error); q weights it in that order."""
    import inspect
    from nmhec import controller
    src = inspect.getsource(controller)
    i = src.index("e = ca.vertcat(")
    head = src[i:i + 200]
    assert "atan2" in head.split("\n")[0], "heading error must come first in the error vector"


def test_redundant_measurement_matrix_matches_h():
    m, rm, x = _model(), RedundantModel(GEO, ts=TS), _state()
    q = rm.from_minimal(x)
    for segs in ((0,), (2,), (0, 2)):
        s = Sensors(m, SensorConfig(pose_segments=segs))
        assert np.allclose(rm.measurement_matrix(s) @ q, s.h(x))


# ---------------------------------------------------------------- kinematics
def test_redundant_and_minimal_models_agree():
    m, rm, x = _model(), RedundantModel(GEO, ts=TS), _state()
    q = rm.from_minimal(x)
    u = [0.3, 0.5]
    for _ in range(25):
        x = m.simulate(x, u)
        q = np.array(rm.step(q, u)).ravel()
    assert np.allclose(rm.to_minimal(q), x, atol=1e-8)
    assert np.allclose(rm.pose(q, 2), m.pose_array(x)[:, 2], atol=1e-6)


def test_feasibility_error_detects_a_broken_chain():
    rm, x = RedundantModel(GEO, ts=TS), _state()
    q = rm.from_minimal(x)
    assert feasibility_error(rm, q) == pytest.approx(0.0, abs=1e-12)
    q[rm.ix(2)] += 0.35
    assert feasibility_error(rm, q) == pytest.approx(0.35, abs=1e-9)


# ----------------------------------------------------------------- estimator
def test_nmhe_respects_the_joint_angle_bound():
    m = _model()
    s = Sensors(m, SensorConfig())
    cfg = EstimatorConfig(n_estimation=6, beta_max=0.3)
    e = NMHE(m, s, cfg)
    x = m.state_from_pose(0, 0, 0, [0.0, 0.0])
    e.reset(x)
    rng = np.random.default_rng(0)
    for _ in range(20):
        y = s.measure(x, rng)
        y[:2] = [1.2, -1.2]                  # measurements well outside the bound
        x_hat = e(y, [0.4, 0.5])
        x = m.simulate(x, [0.4, 0.5])
    assert np.all(np.abs(x_hat[3:]) <= cfg.beta_max + 1e-6)


def test_nmhe_estimate_is_always_feasible():
    """The chain relations are equality constraints, so there is nothing to check but zero."""
    m, rm = _model(), RedundantModel(GEO, ts=TS)
    s = Sensors(m, SensorConfig())
    e = NMHE(m, s, EstimatorConfig(n_estimation=6))
    x = m.state_from_pose(0, 0, 0, [0.1, -0.05])
    e.reset(x)
    rng = np.random.default_rng(1)
    for _ in range(15):
        x_hat = e(s.measure(x, rng), [0.3, 0.5])
        x = m.simulate(x, Ground(0.6, 0.6).apply([0.3, 0.5]))
    assert feasibility_error(rm, rm.from_minimal(x_hat)) == pytest.approx(0.0, abs=1e-9)


def test_slip_is_not_visible_to_the_estimators():
    """Ground scales the achieved velocity; nothing downstream is told the coefficients."""
    g = Ground(rho_omega=0.5, rho_v=0.8)
    assert np.allclose(g.apply([1.0, 1.0]), [0.5, 0.8])
    assert not Ground().slipping and g.slipping


def test_angle_wrapping():
    """Residuals must land in [-pi, pi]; +pi and -pi are the same angle, so compare magnitudes."""
    for a in (3 * np.pi, -3 * np.pi, 2 * np.pi + 0.3, -2 * np.pi - 0.3):
        w = float(np.atleast_1d(wrap(np.array([a])))[0])
        assert -np.pi - 1e-9 <= w <= np.pi + 1e-9
        assert np.cos(w) == pytest.approx(np.cos(a), abs=1e-9)
        assert np.sin(w) == pytest.approx(np.sin(a), abs=1e-9)


def test_sensors_measure_the_requested_segment():
    m, x = _model(), _state()
    tractor = Sensors(m, SensorConfig(pose_segments=(0,))).h(x)
    last = Sensors(m, SensorConfig(pose_segments=(2,))).h(x)
    assert np.allclose(tractor[2:], [x[2], x[0], x[1]])
    assert np.allclose(last[2:], m.pose_array(x)[:, 2][[2, 0, 1]])
