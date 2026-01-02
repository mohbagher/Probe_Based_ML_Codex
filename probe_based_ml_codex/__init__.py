"""Probe-based RIS control utilities."""

from .metrics import power_ratio
from .models import ModelType, ReceivedPowerModel
from .probes import ProbeBank, StructuredOrthogonalProbeBank
from .problem import ProbeSelectionProblem
from .selectors import Selector, ThresholdSelector, select_fixed, select_oracle
from .simulation import EvaluationResult, evaluate_selector
from .types import PhaseVector, PowerVector

__all__ = [
    "ModelType",
    "ReceivedPowerModel",
    "ProbeBank",
    "StructuredOrthogonalProbeBank",
    "ProbeSelectionProblem",
    "Selector",
    "ThresholdSelector",
    "select_fixed",
    "select_oracle",
    "EvaluationResult",
    "evaluate_selector",
    "PhaseVector",
    "PowerVector",
    "power_ratio",
]
