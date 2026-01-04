import pytest

from probe_based_ml_codex.models import ReceivedPowerModel
from probe_based_ml_codex.probes import StructuredOrthogonalProbeBank
from probe_based_ml_codex.problem import ProbeSelectionProblem
from probe_based_ml_codex.selectors import select_oracle
from probe_based_ml_codex.simulation import evaluate_selector


def test_evaluate_selector_mean_ratio() -> None:
    model = ReceivedPowerModel.general_model(["Element-wise sum"])
    bank = StructuredOrthogonalProbeBank.from_codebook(((0.0, 0.0), (0.0, 3.14)))
    problem = ProbeSelectionProblem(model=model, probe_bank=bank)

    result = evaluate_selector(problem, select_oracle, [[0.2, 0.7], [0.4, 0.1]])
    assert result.mean_ratio == pytest.approx(1.0)
