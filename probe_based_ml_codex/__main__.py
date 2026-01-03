"""CLI entry point for probe-based RIS control experiments."""

from __future__ import annotations

import argparse
from importlib.util import find_spec
from pathlib import Path

from .experiments import (
    ExperimentConfig,
    generate_probe_bank_binary,
    generate_probe_bank_continuous,
    generate_probe_bank_hadamard,
    run_all_tasks,
    run_task_a1,
    run_task_a2,
    run_task_a3,
    run_task_b1,
    run_task_b2,
    run_task_b3,
)


TASK_OPTIONS = ["A1", "A2", "A3", "A4", "B1", "B2", "B3", "C1", "C2", "D1", "D2"]
MODEL_OPTIONS = ["continuous", "binary", "hadamard", "all"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe-based RIS control tasks")
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["A1"],
        choices=TASK_OPTIONS,
        help="Tasks to run (A1-D2).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["binary"],
        choices=MODEL_OPTIONS,
        help="Probe models to run (continuous, binary, hadamard, all).",
    )
    parser.add_argument("--n-elements", type=int, default=64, help="Number of RIS elements (N)")
    parser.add_argument("--n-probes", type=int, default=64, help="Number of probes (K)")
    parser.add_argument(
        "--m-observed",
        type=int,
        nargs="+",
        default=[2, 4, 8, 16],
        help="Observed probes per trial (M)",
    )
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--trials", type=int, default=200, help="Number of trials")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory for outputs",
    )
    return parser.parse_args()


def _selected_banks(models: list[str], config: ExperimentConfig):
    if "all" in models:
        return [
            generate_probe_bank_continuous(config.n_elements, config.n_probes, config.seed),
            generate_probe_bank_binary(config.n_elements, config.n_probes, config.seed),
            generate_probe_bank_hadamard(config.n_elements, config.n_probes),
        ]
    banks = []
    if "continuous" in models:
        banks.append(generate_probe_bank_continuous(config.n_elements, config.n_probes, config.seed))
    if "binary" in models:
        banks.append(generate_probe_bank_binary(config.n_elements, config.n_probes, config.seed))
    if "hadamard" in models:
        banks.append(generate_probe_bank_hadamard(config.n_elements, config.n_probes))
    return banks


def _write_run_summary(results_dir: Path, args: argparse.Namespace) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    summary = [
        "Probe-based RIS experiment run summary",
        f"Tasks: {', '.join(args.tasks)}",
        f"Models: {', '.join(args.models)}",
        f"N (elements): {args.n_elements}",
        f"K (probes): {args.n_probes}",
        f"M (observed): {args.m_observed}",
        f"Seed: {args.seed}",
        f"Trials: {args.trials}",
        f"Results directory: {results_dir.resolve()}",
    ]
    (results_dir / "run_summary.txt").write_text("\n".join(summary), encoding="utf-8")


def main() -> None:
    args = _parse_args()
    print(f"Running probe_based_ml_codex from {Path(__file__).resolve()}")

    if find_spec("numpy") is None or find_spec("matplotlib") is None:
        raise ModuleNotFoundError(
            "numpy and matplotlib are required to run experiments. "
            "Install with `pip install numpy matplotlib`."
        )

    config = ExperimentConfig(
        n_elements=args.n_elements,
        n_probes=args.n_probes,
        m_observed=tuple(args.m_observed),
        seed=args.seed,
        n_trials=args.trials,
    )

    _write_run_summary(args.results_dir, args)

    if args.models == ["all"] and set(args.tasks) == {"A1", "A2", "A3", "B1", "B2", "B3"}:
        run_all_tasks(args.results_dir, config)
        print(f"Saved results to {args.results_dir.resolve()}")
        return

    banks = _selected_banks(args.models, config)

    for task in args.tasks:
        if task == "A1":
            run_task_a1(args.results_dir, config)
        elif task == "A2":
            run_task_a2(args.results_dir, config)
        elif task == "A3":
            for bank in banks:
                run_task_a3(args.results_dir, bank)
        elif task == "B1":
            for bank in banks:
                run_task_b1(args.results_dir, config, bank)
        elif task == "B2":
            for bank in banks:
                run_task_b2(args.results_dir, config, bank)
        elif task == "B3":
            for bank in banks:
                run_task_b3(args.results_dir, config, bank)
        else:
            print(f"Task {task} is registered but not implemented yet.")

    print(f"Saved results to {args.results_dir.resolve()}")


if __name__ == "__main__":
    main()
