"""CLI entry point for quick sanity checks."""

from __future__ import annotations

from .models import ReceivedPowerModel
from .probes import StructuredOrthogonalProbeBank
from .problem import ProbeSelectionProblem


def main() -> None:
    model = ReceivedPowerModel.general_model(
        assumptions=(
            "Element-wise channel summation",
            "Phase-only RIS control",
            "Fixed geometry per realization",
        )
    )
    probe_bank = StructuredOrthogonalProbeBank.from_codebook(
        phase_vectors=((0.0, 0.0), (0.0, 3.14159), (3.14159, 0.0))
    )
    problem = ProbeSelectionProblem(model=model, probe_bank=probe_bank)
    received_powers = [0.8, 1.0, 0.7]
    best_index = problem.select_best_probe(received_powers)
    ratio = problem.evaluate_selection(best_index, received_powers)

    print(model.describe())
    print(f"Probe bank size: {len(probe_bank)}")
    print(f"Best probe index: {best_index}")
    print(f"Power ratio (oracle): {ratio:.3f}")


if __name__ == "__main__":
    main()
