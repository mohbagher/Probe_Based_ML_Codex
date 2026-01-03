# Usage

This repository provides lightweight, code-first primitives for framing probe-based RIS
control experiments. The API focuses on:

- selecting a fixed analytical model as a baseline,
- defining a structured probe bank,
- computing the oracle-normalized power ratio.

## Example

```python
from probe_based_ml_codex.models import ReceivedPowerModel
from probe_based_ml_codex.probes import StructuredOrthogonalProbeBank
from probe_based_ml_codex.problem import ProbeSelectionProblem

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

oracle_ratio = problem.evaluate_selection(
    selected_index=problem.select_best_probe(received_powers),
    received_powers=received_powers,
)

print(f"Oracle ratio: {oracle_ratio:.3f}")
```
