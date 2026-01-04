# Probe_Based_ML_Codex

Probe-based RIS control primitives grounded in analytical received-power models.

## Documentation
- [Project story](docs/project_story.md)
- [Usage](docs/usage.md)

## Quickstart
Install dependencies:

```bash
pip install numpy matplotlib
```

Run an experiment task:

```bash
python -m probe_based_ml_codex --tasks A1
```

If you suspect an older installed package is being picked up, use:

```bash
python run_experiments.py --tasks A1 A2 A3 B1 B2 B3 --models all
```

Outputs are written under the `results/` directory, grouped by task and probe model.
