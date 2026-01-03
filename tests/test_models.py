from probe_based_ml_codex.models import ModelType, ReceivedPowerModel


def test_model_description_includes_assumptions() -> None:
    model = ReceivedPowerModel.far_field_model(["Far-field", "Narrow angles"])
    description = model.describe()
    assert "far_field" in description
    assert "Far-field" in description


def test_model_type_values() -> None:
    assert ModelType.GENERAL.value == "general"
    assert ModelType.FAR_FIELD.value == "far_field"
    assert ModelType.ALTERNATIVE.value == "alternative"
