"""Training package: config, hardware, checkpoints, splits, tracking."""

from nova.training.checkpoints import CheckpointManager, IncompatibleCheckpointError
from nova.training.config import TrainingConfig
from nova.training.datasets import SPLITS, HeldOutLeakError, SplitEnforcer
from nova.training.hardware import describe_device
from nova.training.tracking import ExperimentSnapshot, LossTracker

__all__ = [
    "SPLITS",
    "CheckpointManager",
    "ExperimentSnapshot",
    "HeldOutLeakError",
    "IncompatibleCheckpointError",
    "LossTracker",
    "SplitEnforcer",
    "TrainingConfig",
    "describe_device",
]
