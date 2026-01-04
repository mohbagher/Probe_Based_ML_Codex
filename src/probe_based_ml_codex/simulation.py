"""Simulation helpers for probe-based evaluation.

These utilities remain model-agnostic and only operate on received power arrays.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from .problem import ProbeSelectionProblem
from .selectors import Selector


@dataclass(frozen=True)
class EvaluationResult:
    """Aggregate metrics for a batch of probe-selection outcomes."""

    ratios: tuple[float, ...]

    @property
    def mean_ratio(self) -> float:
        if not self.ratios:
            raise ValueError("No ratios provided.")
        return sum(self.ratios) / len(self.ratios)


def evaluate_selector(
    problem: ProbeSelectionProblem,
    selector: Selector,
    received_power_batches: Iterable[Sequence[float]],
) -> EvaluationResult:
    """Evaluate a selector over multiple channel realizations."""

    ratios = []
    for powers in received_power_batches:
        ratios.append(problem.run_selector(selector, powers))
    return EvaluationResult(tuple(ratios))
