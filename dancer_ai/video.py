from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from dancer_ai.config import VideoRequirements


@dataclass(frozen=True)
class VideoMetadata:
    path: str
    width: int
    height: int
    fps: float
    frame_count: int
    duration_seconds: float


@dataclass(frozen=True)
class VideoInspection:
    metadata: VideoMetadata
    accepted: bool
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata": asdict(self.metadata),
            "accepted": self.accepted,
            "warnings": self.warnings,
        }


def inspect_video(
    path: str | Path,
    requirements: VideoRequirements | None = None,
) -> VideoInspection:
    """Odczytaj metadane i sprawdź podstawowe wymagania materiału."""
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "Brak OpenCV. Uruchom: python -m pip install -r requirements.txt"
        ) from exc

    requirements = requirements or VideoRequirements()
    video_path = Path(path).expanduser().resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"Nie znaleziono pliku wideo: {video_path}")

    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise ValueError(f"OpenCV nie potrafi otworzyć pliku: {video_path}")

        width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    finally:
        capture.release()

    duration = frame_count / fps if fps > 0 else 0.0
    metadata = VideoMetadata(
        path=str(video_path),
        width=width,
        height=height,
        fps=round(fps, 3),
        frame_count=frame_count,
        duration_seconds=round(duration, 3),
    )

    warnings: list[str] = []
    if fps <= 0:
        warnings.append("Nie udało się odczytać liczby klatek na sekundę.")
    if duration < requirements.min_duration_seconds:
        warnings.append(
            f"Nagranie jest za krótkie: {duration:.2f} s; minimum to "
            f"{requirements.min_duration_seconds:.0f} s."
        )
    if frame_count <= 0:
        warnings.append("Plik nie zawiera możliwej do odczytania liczby klatek.")

    return VideoInspection(metadata=metadata, accepted=not warnings, warnings=warnings)
