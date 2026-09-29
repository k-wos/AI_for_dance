from dataclasses import dataclass


@dataclass(frozen=True)
class VideoRequirements:
    """Minimalne wymagania nagrania przyjęte dla MVP."""

    min_duration_seconds: float = 3.0
