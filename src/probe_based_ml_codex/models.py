"""Analytical received-power model definitions.

These models provide the physical baseline against which probe selection is judged.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable


class ModelType(str, Enum):
    """Supported analytical model families."""

    GENERAL = "general"
    FAR_FIELD = "far_field"
    ALTERNATIVE = "alternative"


@dataclass(frozen=True)
class ReceivedPowerModel:
    """Metadata describing the analytical model chosen as baseline.

    Attributes:
        model_type: High-level model family.
        assumptions: Short statements about propagation, geometry, and scaling.
        reference_notes: Optional notes for provenance or citations.
    """

    model_type: ModelType
    assumptions: tuple[str, ...]
    reference_notes: tuple[str, ...] = ()

    def describe(self) -> str:
        """Return a human-readable description of the model."""

        assumption_block = "\n".join(f"- {assumption}" for assumption in self.assumptions)
        notes_block = "\n".join(f"- {note}" for note in self.reference_notes)
        description = [f"Model: {self.model_type.value}", "Assumptions:", assumption_block]
        if notes_block:
            description.extend(["Reference notes:", notes_block])
        return "\n".join(description)

    @classmethod
    def general_model(cls, assumptions: Iterable[str]) -> "ReceivedPowerModel":
        """Create a general per-element summation model descriptor."""

        return cls(model_type=ModelType.GENERAL, assumptions=tuple(assumptions))

    @classmethod
    def far_field_model(cls, assumptions: Iterable[str]) -> "ReceivedPowerModel":
        """Create a far-field beamforming model descriptor."""

        return cls(model_type=ModelType.FAR_FIELD, assumptions=tuple(assumptions))

    @classmethod
    def alternative_model(cls, assumptions: Iterable[str]) -> "ReceivedPowerModel":
        """Create an alternative general model descriptor."""

        return cls(model_type=ModelType.ALTERNATIVE, assumptions=tuple(assumptions))
