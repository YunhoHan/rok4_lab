"""Robot configurations for RoK4 tasks."""

from .rok4 import ROK4_TEST_CFG, ROK4_TRAIN_CFG
from .rok4_adapt import RoK4AdaptActuator, RoK4AdaptActuatorCfg, RoK4AdaptTransmission

__all__ = [
    "ROK4_TEST_CFG",
    "ROK4_TRAIN_CFG",
    "RoK4AdaptActuator",
    "RoK4AdaptActuatorCfg",
    "RoK4AdaptTransmission",
]
