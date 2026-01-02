"""Probe bank definitions for RIS phase configurations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class ProbeBank:
    """Container for a fixed set of RIS phase configurations."""

    name: str
    phase_vectors: tuple[tuple[float, ...], ...]

    def __len__(self) -> int:
        return len(self.phase_vectors)

    def as_list(self) -> list[list[float]]:
        return [list(vector) for vector in self.phase_vectors]


@dataclass(frozen=True)
class StructuredOrthogonalProbeBank(ProbeBank):
    """Structured probe bank with orthogonal phase patterns."""

    @classmethod
    def from_codebook(
        cls,
        phase_vectors: Iterable[Sequence[float]],
        name: str = "structured_orthogonal",
    ) -> "StructuredOrthogonalProbeBank":
        """Build a structured probe bank from an explicit codebook."""

        vectors = tuple(tuple(vector) for vector in phase_vectors)
        if not vectors:
            raise ValueError("Probe codebook cannot be empty.")
        vector_length = len(vectors[0])
        if any(len(vector) != vector_length for vector in vectors):
            raise ValueError("All probe vectors must have the same length.")
        return cls(name=name, phase_vectors=vectors)
