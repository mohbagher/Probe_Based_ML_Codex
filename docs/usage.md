# Usage

This repository provides lightweight, code-first primitives for framing probe-based RIS
control experiments. The API focuses on:

- selecting a fixed analytical model as a baseline,
- defining a structured probe bank,
- computing the oracle-normalized power ratio,
- evaluating simple selector heuristics.

## Example

```python
from probe_based_ml_codex.models import ReceivedPowerModel
from probe_based_ml_codex.probes import StructuredOrthogonalProbeBank
from probe_based_ml_codex.problem import ProbeSelectionProblem
from probe_based_ml_codex.selectors import ThresholdSelector
from probe_based_ml_codex.simulation import evaluate_selector

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
print(f"Threshold selector ratio: {ratio:.3f}")

batch_result = evaluate_selector(problem, selector, [received_powers])
print(f"Mean ratio: {batch_result.mean_ratio:.3f}")
```
