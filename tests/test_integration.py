from probe_based_ml_codex.models import ReceivedPowerModel
from probe_based_ml_codex.probes import StructuredOrthogonalProbeBank
from probe_based_ml_codex.problem import ProbeSelectionProblem
from probe_based_ml_codex.selectors import ThresholdSelector


def test_threshold_selector_in_problem() -> None:
    model = ReceivedPowerModel.general_model(["Element-wise sum"])
    bank = StructuredOrthogonalProbeBank.from_codebook(((0.0, 0.0), (0.0, 3.14)))
    problem = ProbeSelectionProblem(model=model, probe_bank=bank)
    selector = ThresholdSelector(threshold=0.9)

    ratio = problem.run_selector(selector, [0.2, 0.7])
    assert ratio == 1.0
