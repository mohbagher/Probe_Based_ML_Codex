# Usage

This repository provides lightweight, code-first primitives for framing probe-based RIS
control experiments. The API focuses on:

- selecting a fixed analytical model as a baseline,
- defining structured or random probe banks,
- computing the oracle-normalized power ratio,
- running probe-design and limited probing experiments.

## Requirements

Install runtime dependencies before running experiments:

```bash
pip install numpy matplotlib
```

## CLI quickstart

Run a single task with defaults (N=64, K=64, M=2/4/8/16, seed=1):

```bash
python -m probe_based_ml_codex --tasks A1
```

If your environment has an older installed package, use the local runner:

```bash
python run_experiments.py --tasks A1 A2 A3 B1 B2 B3 --models all
```

Run all A1-B3 tasks for all probe models and save outputs under `results/`:

```bash
python -m probe_based_ml_codex --tasks A1 A2 A3 B1 B2 B3 --models all
```

Baseline plots in B3 include a simple ML-free regression baseline (`ml_linear`) that
fits observed probe powers against cosine/sine phase features.

Override defaults:

```bash
python -m probe_based_ml_codex --tasks B1 --models continuous --n-elements 32 --n-probes 64 --m-observed 4 8 --seed 7
```

## Python API example

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
