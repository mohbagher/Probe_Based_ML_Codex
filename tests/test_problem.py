import pytest

from probe_based_ml_codex.models import ReceivedPowerModel
from probe_based_ml_codex.probes import StructuredOrthogonalProbeBank
from probe_based_ml_codex.problem import ProbeSelectionProblem


def _problem() -> ProbeSelectionProblem:
    model = ReceivedPowerModel.general_model(["Element-wise sum"])
    bank = StructuredOrthogonalProbeBank.from_codebook(((0.0, 0.0), (0.0, 3.14)))
    return ProbeSelectionProblem(model=model, probe_bank=bank)


def test_select_best_probe() -> None:
    problem = _problem()
    assert problem.select_best_probe([0.1, 0.3]) == 1


def test_evaluate_selection_matches_oracle() -> None:
    problem = _problem()
    ratio = problem.evaluate_selection(1, [0.1, 0.3])
    assert ratio == pytest.approx(1.0)


def test_evaluate_selection_invalid_length() -> None:
    problem = _problem()
    with pytest.raises(ValueError):
        problem.evaluate_selection(0, [0.1])
