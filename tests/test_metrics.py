from probe_based_ml_codex.metrics import power_ratio


def test_power_ratio_valid() -> None:
    assert power_ratio(0.5, 1.0) == 0.5


def test_power_ratio_rejects_non_positive_best() -> None:
    try:
        power_ratio(0.5, 0.0)
    except ValueError as exc:
        assert "best_probe_power" in str(exc)
    else:
        raise AssertionError("Expected ValueError for non-positive best power")


def test_power_ratio_rejects_negative_selected() -> None:
    try:
        power_ratio(-0.1, 1.0)
    except ValueError as exc:
        assert "selected_power" in str(exc)
    else:
        raise AssertionError("Expected ValueError for negative selected power")
