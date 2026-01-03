from probe_based_ml_codex.types import PhaseVector, PowerVector


def test_type_aliases_are_sequences() -> None:
    phase: PhaseVector = [0.0, 3.14]
    power: PowerVector = [0.1, 0.2]
    assert phase[0] == 0.0
    assert power[-1] == 0.2
