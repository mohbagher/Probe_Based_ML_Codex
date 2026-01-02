import pytest

from probe_based_ml_codex.selectors import ThresholdSelector, select_fixed, select_oracle


def test_select_oracle() -> None:
    assert select_oracle([0.2, 0.5, 0.1]) == 1


def test_select_fixed() -> None:
    selector = select_fixed(0)
    assert selector([0.2, 0.5]) == 0


def test_select_fixed_out_of_range() -> None:
    selector = select_fixed(3)
    with pytest.raises(IndexError):
        selector([0.2, 0.5])


def test_threshold_selector() -> None:
    selector = ThresholdSelector(threshold=0.6)
    assert selector([0.2, 0.7, 0.5]) == 1


def test_threshold_selector_fallback() -> None:
    selector = ThresholdSelector(threshold=0.9)
    assert selector([0.2, 0.7, 0.5]) == 1
