"""Metrics for evaluating probe-based selection."""

from __future__ import annotations


def power_ratio(selected_power: float, best_probe_power: float) -> float:
    """Compute the oracle-normalized power ratio.

    Args:
        selected_power: Received power for the chosen probe.
        best_probe_power: Received power of the oracle best probe.

    Returns:
        Dimensionless ratio in [0, 1] when selected_power <= best_probe_power.
    """

    if best_probe_power <= 0:
        raise ValueError("best_probe_power must be positive.")
    if selected_power < 0:
        raise ValueError("selected_power must be non-negative.")
    return selected_power / best_probe_power
