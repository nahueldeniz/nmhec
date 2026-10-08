"""Nonlinear moving horizon estimation and control for generalised N-trailer vehicles.

Deniz, N., Jorquera, F., Torres-Torriti, M. and Auat Cheein, F., "Model predictive
path-following controller for Generalised N-Trailer vehicles with noisy sensors and
disturbances", Control Engineering Practice 142:105747, 2024.
doi:10.1016/j.conengprac.2023.105747

An independent Python + CasADi re-implementation written from the paper; see NOTES.md for
what is and is not reproduced. The vehicle model, the predictive controller and the
path-following target updater are shared with the nmppfc package (paper 02).
"""
from .model import NTrailer, VehicleGeometry
from .path import Path, lemniscate, line, sine
from .controller import NMPPFC, ControllerConfig
from .sensing import Sensors, SensorConfig, Ground
from .estimator import NMHE, EKF, EstimatorConfig
from .arrival_cost import AdaptiveArrivalCost
from .pfa import TargetUpdater, PFAConfig
from .redundant import RedundantModel, RedundantEKF, feasibility_error
from .sim import ClosedLoop, SimulationResult
from . import viz

__all__ = ["NTrailer", "VehicleGeometry", "Path", "lemniscate", "line", "sine",
           "NMPPFC", "ControllerConfig", "Sensors", "SensorConfig", "Ground",
           "NMHE", "EKF", "EstimatorConfig", "AdaptiveArrivalCost",
           "TargetUpdater", "PFAConfig", "ClosedLoop", "SimulationResult",
           "RedundantModel", "RedundantEKF", "feasibility_error", "viz"]
__version__ = "0.1.0"
