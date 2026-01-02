"""Selector utilities for probe-based RIS control."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence


Selector = Callable[[Sequence[float]], int]


def select_oracle(received_powers: Sequence[float]) -> int:
    """Select the probe index with the highest received power."""

    if not received_powers:
        raise ValueError("received_powers cannot be empty.")
    return max(range(len(received_powers)), key=received_powers.__getitem__)


def select_fixed(index: int) -> Selector:
    """Return a selector that always chooses a fixed index."""

    def _selector(received_powers: Sequence[float]) -> int:
        if not received_powers:
            raise ValueError("received_powers cannot be empty.")
        if not (0 <= index < len(received_powers)):
            raise IndexError("fixed index is out of range.")
        return index

    return _selector


@dataclass(frozen=True)
class ThresholdSelector:
    """Select the first probe exceeding a threshold, else fallback to oracle.

    This provides a simple, interpretable heuristic useful for baselines.
    """

    threshold: float
    fallback: Selector = select_oracle

    def __call__(self, received_powers: Sequence[float]) -> int:
        if not received_powers:
            raise ValueError("received_powers cannot be empty.")
        for index, power in enumerate(received_powers):
            if power >= self.threshold:
                return index
        return self.fallback(received_powers)
