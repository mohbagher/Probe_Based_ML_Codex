import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("matplotlib")

from probe_based_ml_codex.experiments import (
    generate_probe_bank_binary,
    generate_probe_bank_continuous,
    generate_probe_bank_hadamard,
)


def test_generate_probe_bank_binary_values() -> None:
    bank = generate_probe_bank_binary(4, 3, seed=0)
    assert bank.phases.shape == (3, 4)
    assert set(np.unique(bank.phases)).issubset({0.0, np.pi})


def test_generate_probe_bank_continuous_range() -> None:
    bank = generate_probe_bank_continuous(4, 3, seed=0)
    assert bank.phases.shape == (3, 4)
    assert np.all(bank.phases >= 0.0)
    assert np.all(bank.phases <= 2 * np.pi)


def test_generate_probe_bank_hadamard_requires_power_of_two() -> None:
    with pytest.raises(ValueError):
        generate_probe_bank_hadamard(6, 3)
