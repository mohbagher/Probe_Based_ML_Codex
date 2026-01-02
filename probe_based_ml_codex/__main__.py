"""CLI entry point for quick sanity checks."""

from __future__ import annotations

from .models import ReceivedPowerModel
from .probes import StructuredOrthogonalProbeBank
from .problem import ProbeSelectionProblem
from .selectors import ThresholdSelector
from .simulation import evaluate_selector


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
    selector = ThresholdSelector(threshold=0.9)

    received_powers = [0.8, 1.0, 0.7]
    ratio = problem.run_selector(selector, received_powers)
    batch = evaluate_selector(problem, selector, [received_powers])

    print(model.describe())
    print(f"Probe bank size: {len(probe_bank)}")
    print(f"Threshold selector ratio: {ratio:.3f}")
    print(f"Mean ratio: {batch.mean_ratio:.3f}")


if __name__ == "__main__":
    main()
