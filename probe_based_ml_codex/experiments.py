"""Experiment runners for probe-based RIS control tasks (A1-B3)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np

from .metrics import power_ratio


@dataclass(frozen=True)
class ExperimentConfig:
    n_elements: int
    n_probes: int
    m_observed: tuple[int, ...]
    seed: int
    n_trials: int = 200


@dataclass(frozen=True)
class ProbeBankResult:
    name: str
    phases: np.ndarray


def generate_probe_bank_continuous(n_elements: int, n_probes: int, seed: int) -> ProbeBankResult:
    rng = np.random.default_rng(seed)
    phases = rng.uniform(0.0, 2 * np.pi, size=(n_probes, n_elements))
    return ProbeBankResult(name="continuous_random", phases=phases)


def generate_probe_bank_binary(n_elements: int, n_probes: int, seed: int) -> ProbeBankResult:
    rng = np.random.default_rng(seed)
    phases = rng.integers(0, 2, size=(n_probes, n_elements)) * np.pi
    return ProbeBankResult(name="binary_random", phases=phases)


def _hadamard_matrix(order: int) -> np.ndarray:
    if order < 1 or (order & (order - 1)) != 0:
        raise ValueError("Hadamard order must be a power of two.")
    matrix = np.array([[1.0]])
    while matrix.shape[0] < order:
        matrix = np.block([[matrix, matrix], [matrix, -matrix]])
    return matrix


def generate_probe_bank_hadamard(n_elements: int, n_probes: int) -> ProbeBankResult:
    hadamard = _hadamard_matrix(n_elements)
    if n_probes > hadamard.shape[0]:
        raise ValueError("n_probes exceeds available Hadamard patterns.")
    patterns = hadamard[:n_probes]
    phases = np.where(patterns > 0, 0.0, np.pi)
    return ProbeBankResult(name="hadamard_structured", phases=phases)


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _save_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def plot_phase_heatmap(phases: np.ndarray, out_path: Path, title: str) -> None:
    plt.figure(figsize=(8, 4))
    plt.imshow(phases, aspect="auto", cmap="twilight")
    plt.colorbar(label="Phase (rad)")
    plt.title(title)
    plt.xlabel("Element index")
    plt.ylabel("Probe index")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_phase_histogram(phases: np.ndarray, out_path: Path, title: str) -> None:
    plt.figure(figsize=(6, 4))
    plt.hist(phases.flatten(), bins=30, color="#4c78a8")
    plt.title(title)
    plt.xlabel("Phase (rad)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_distribution(values: np.ndarray, out_path: Path, title: str, xlabel: str) -> None:
    plt.figure(figsize=(6, 4))
    plt.hist(values, bins=30, color="#54a24b")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _pairwise_cosine_similarity(phases: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(phases, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    normalized = phases / norms
    similarities = normalized @ normalized.T
    indices = np.triu_indices(similarities.shape[0], k=1)
    return similarities[indices]


def _pairwise_hamming_distance(phases: np.ndarray) -> np.ndarray:
    binary = (phases == 0.0).astype(int)
    distances = []
    for i in range(binary.shape[0]):
        for j in range(i + 1, binary.shape[0]):
            distances.append(np.sum(binary[i] != binary[j]))
    return np.array(distances, dtype=float)


def compute_received_powers(phases: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n_probes, n_elements = phases.shape
    channel = (rng.normal(size=n_elements) + 1j * rng.normal(size=n_elements)) / np.sqrt(2)
    responses = np.exp(1j * phases) @ channel
    powers = np.abs(responses) ** 2
    return powers


def _cosine_similarity_scores(
    phases: np.ndarray,
    observed_idx: Sequence[int],
    observed_powers: Sequence[float],
) -> np.ndarray:
    observed_phases = phases[np.array(observed_idx)]
    normalized = phases / np.linalg.norm(phases, axis=1, keepdims=True)
    observed_norm = observed_phases / np.linalg.norm(observed_phases, axis=1, keepdims=True)
    similarities = normalized @ observed_norm.T
    weights = np.maximum(similarities, 0.0)
    if np.all(weights == 0):
        return np.zeros(phases.shape[0])
    weighted = weights @ np.array(observed_powers)
    return weighted / np.sum(weights, axis=1)


def evaluate_baselines(
    phases: np.ndarray,
    m_observed: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    powers = compute_received_powers(phases, rng)
    oracle_power = float(powers.max())

    random_full_idx = int(rng.integers(0, len(powers)))
    observed_idx = rng.choice(len(powers), size=m_observed, replace=False)
    observed_powers = powers[observed_idx]
    best_observed_idx = int(observed_idx[np.argmax(observed_powers)])

    similarity_scores = _cosine_similarity_scores(phases, observed_idx, observed_powers)
    similarity_idx = int(np.argmax(similarity_scores))

    return {
        "random_full": power_ratio(powers[random_full_idx], oracle_power),
        "random_subset_best": power_ratio(powers[best_observed_idx], oracle_power),
        "best_observed": power_ratio(powers[best_observed_idx], oracle_power),
        "similarity_baseline": power_ratio(powers[similarity_idx], oracle_power),
    }


def run_task_a1(result_dir: Path, config: ExperimentConfig) -> None:
    bank = generate_probe_bank_binary(config.n_elements, config.n_probes, config.seed)
    task_dir = _ensure_dir(result_dir / "A1" / bank.name)
    plot_phase_heatmap(bank.phases, task_dir / "phase_heatmap.png", "Binary probe phases")
    plot_phase_histogram(bank.phases, task_dir / "phase_histogram.png", "Phase histogram")
    _save_text(task_dir / "summary.txt", f"Generated {bank.name} with shape {bank.phases.shape}")


def run_task_a2(result_dir: Path, config: ExperimentConfig) -> None:
    bank = generate_probe_bank_hadamard(config.n_elements, config.n_probes)
    task_dir = _ensure_dir(result_dir / "A2" / bank.name)
    plot_phase_heatmap(bank.phases, task_dir / "phase_heatmap.png", "Hadamard probe phases")
    distances = _pairwise_hamming_distance(bank.phases)
    plot_distribution(distances, task_dir / "hamming_distance.png", "Pairwise Hamming distance", "Distance")
    summary = (
        f"Generated {bank.name} with shape {bank.phases.shape}\n"
        f"Mean Hamming distance: {distances.mean():.2f}\n"
        f"Std Hamming distance: {distances.std():.2f}"
    )
    _save_text(task_dir / "summary.txt", summary)


def run_task_a3(result_dir: Path, bank: ProbeBankResult) -> None:
    task_dir = _ensure_dir(result_dir / "A3" / bank.name)
    if bank.name == "continuous_random":
        similarities = _pairwise_cosine_similarity(bank.phases)
        plot_distribution(
            similarities,
            task_dir / "cosine_similarity.png",
            "Cosine similarity",
            "Similarity",
        )
        summary = (
            f"Mean cosine similarity: {similarities.mean():.3f}\n"
            f"Std cosine similarity: {similarities.std():.3f}"
        )
    else:
        distances = _pairwise_hamming_distance(bank.phases)
        plot_distribution(
            distances,
            task_dir / "hamming_distance.png",
            "Hamming distance",
            "Distance",
        )
        summary = (
            f"Mean Hamming distance: {distances.mean():.2f}\n"
            f"Std Hamming distance: {distances.std():.2f}"
        )
    _save_text(task_dir / "summary.txt", summary)


def _eta_vs_m(
    phases: np.ndarray,
    m_observed: Iterable[int],
    n_trials: int,
    rng: np.random.Generator,
) -> tuple[list[float], list[float]]:
    mean_values = []
    median_values = []
    for m in m_observed:
        ratios = []
        for _ in range(n_trials):
            powers = compute_received_powers(phases, rng)
            observed_idx = rng.choice(len(powers), size=m, replace=False)
            ratios.append(powers[observed_idx].max() / powers.max())
        mean_values.append(float(np.mean(ratios)))
        median_values.append(float(np.median(ratios)))
    return mean_values, median_values


def run_task_b1(result_dir: Path, config: ExperimentConfig, bank: ProbeBankResult) -> None:
    task_dir = _ensure_dir(result_dir / "B1" / bank.name)
    rng = np.random.default_rng(config.seed)
    mean_values, median_values = _eta_vs_m(bank.phases, config.m_observed, config.n_trials, rng)

    plt.figure(figsize=(6, 4))
    plt.plot(config.m_observed, mean_values, marker="o", label="Mean eta")
    plt.plot(config.m_observed, median_values, marker="s", label="Median eta")
    plt.title(f"Eta vs M ({bank.name})")
    plt.xlabel("M (observed probes)")
    plt.ylabel("Eta")
    plt.legend()
    plt.tight_layout()
    plt.savefig(task_dir / "eta_vs_m.png")
    plt.close()

    summary = "\n".join(
        f"M={m}: mean={mean_values[i]:.3f}, median={median_values[i]:.3f}"
        for i, m in enumerate(config.m_observed)
    )
    _save_text(task_dir / "summary.txt", summary)


def run_task_b2(result_dir: Path, config: ExperimentConfig, bank: ProbeBankResult) -> None:
    task_dir = _ensure_dir(result_dir / "B2" / bank.name)
    rng = np.random.default_rng(config.seed)
    top_values: dict[int, list[float]] = {1: [], 2: [], 4: [], 8: []}

    for _ in range(config.n_trials):
        powers = compute_received_powers(bank.phases, rng)
        sorted_powers = np.sort(powers)[::-1]
        for top_m in top_values:
            if top_m <= len(sorted_powers):
                top_values[top_m].append(sorted_powers[top_m - 1] / sorted_powers[0])

    plt.figure(figsize=(6, 4))
    means = []
    for top_m, values in top_values.items():
        mean_val = float(np.mean(values))
        means.append(mean_val)
        plt.plot([top_m], [mean_val], marker="o", label=f"Top-{top_m}")
    plt.title(f"Mean eta vs Top-m ({bank.name})")
    plt.xlabel("Top-m")
    plt.ylabel("Mean eta")
    plt.tight_layout()
    plt.savefig(task_dir / "top_m_mean_eta.png")
    plt.close()

    summary_lines = [
        f"Top-{top_m}: mean={np.mean(values):.3f}, median={np.median(values):.3f}"
        for top_m, values in top_values.items()
    ]
    _save_text(task_dir / "summary.txt", "\n".join(summary_lines))


def run_task_b3(result_dir: Path, config: ExperimentConfig, bank: ProbeBankResult) -> None:
    task_dir = _ensure_dir(result_dir / "B3" / bank.name)
    rng = np.random.default_rng(config.seed)

    summary_lines = []
    for m in config.m_observed:
        baseline_samples = {
            "random_full": [],
            "random_subset_best": [],
            "best_observed": [],
            "similarity_baseline": [],
        }
        for _ in range(config.n_trials):
            baselines = evaluate_baselines(bank.phases, m, rng)
            for key in baseline_samples:
                baseline_samples[key].append(baselines[key])

        means = {key: float(np.mean(vals)) for key, vals in baseline_samples.items()}
        plt.figure(figsize=(7, 4))
        plt.bar(means.keys(), means.values(), color="#4c78a8")
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Mean eta")
        plt.title(f"Baselines (M={m}) - {bank.name}")
        plt.tight_layout()
        plt.savefig(task_dir / f"baselines_m_{m}.png")
        plt.close()

        summary_lines.append(
            f"M={m}: " + ", ".join(f"{key} mean={val:.3f}" for key, val in means.items())
        )

    _save_text(task_dir / "summary.txt", "\n".join(summary_lines))


def run_all_tasks(result_dir: Path, config: ExperimentConfig) -> None:
    banks = [
        generate_probe_bank_continuous(config.n_elements, config.n_probes, config.seed),
        generate_probe_bank_binary(config.n_elements, config.n_probes, config.seed),
        generate_probe_bank_hadamard(config.n_elements, config.n_probes),
    ]

    run_task_a1(result_dir, config)
    run_task_a2(result_dir, config)
    for bank in banks:
        run_task_a3(result_dir, bank)
        run_task_b1(result_dir, config, bank)
        run_task_b2(result_dir, config, bank)
        run_task_b3(result_dir, config, bank)

    compare_dir = _ensure_dir(result_dir / "compare")
    plt.figure(figsize=(6, 4))
    for bank in banks:
        rng = np.random.default_rng(config.seed)
        mean_values, _ = _eta_vs_m(bank.phases, config.m_observed, config.n_trials, rng)
        plt.plot(config.m_observed, mean_values, marker="o", label=bank.name)
    plt.xlabel("M (observed probes)")
    plt.ylabel("Mean eta")
    plt.title("Mean eta vs M (all models)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(compare_dir / "mean_eta_vs_m_all.png")
    plt.close()
