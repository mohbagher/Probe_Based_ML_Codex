import pytest

from probe_based_ml_codex.probes import StructuredOrthogonalProbeBank


def test_probe_bank_from_codebook() -> None:
    bank = StructuredOrthogonalProbeBank.from_codebook(((0.0, 0.0), (0.0, 3.14)))
    assert len(bank) == 2
    assert bank.name == "structured_orthogonal"


def test_probe_bank_rejects_empty() -> None:
    with pytest.raises(ValueError):
        StructuredOrthogonalProbeBank.from_codebook(())


def test_probe_bank_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError):
        StructuredOrthogonalProbeBank.from_codebook(((0.0,), (0.0, 3.14)))
