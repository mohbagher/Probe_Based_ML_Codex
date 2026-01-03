"""Core problem definition for probe-based RIS control."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from .metrics import power_ratio
from .models import ReceivedPowerModel
from .probes import ProbeBank


@dataclass(frozen=True)
class ProbeSelectionProblem:
    """Encapsulate the probe-selection task and evaluation metric."""

    model: ReceivedPowerModel
    probe_bank: ProbeBank

    def evaluate_selection(
        self,
        selected_index: int,
        received_powers: Sequence[float],
    ) -> float:
        """Evaluate the selected probe against the oracle best probe.

        Args:
            selected_index: Index of the probe chosen by a controller.
            received_powers: Received power values for each probe.

        Returns:
            Oracle-normalized power ratio.
        """

        if len(received_powers) != len(self.probe_bank):
            raise ValueError("received_powers must match probe bank length.")
        if not (0 <= selected_index < len(self.probe_bank)):
            raise IndexError("selected_index is out of range.")
        best_probe_power = max(received_powers)
        selected_power = received_powers[selected_index]
        return power_ratio(selected_power=selected_power, best_probe_power=best_probe_power)

    def select_best_probe(self, received_powers: Sequence[float]) -> int:
        """Return the oracle-best probe index for the given power list."""

        if len(received_powers) != len(self.probe_bank):
            raise ValueError("received_powers must match probe bank length.")
        return max(range(len(received_powers)), key=received_powers.__getitem__)

    def run_selector(
        self,
        selector: Callable[[Sequence[float]], int],
        received_powers: Sequence[float],
    ) -> float:
        """Run an external selector and return its normalized performance."""

        selected_index = selector(received_powers)
        return self.evaluate_selection(selected_index, received_powers)
