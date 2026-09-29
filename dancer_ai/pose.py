from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import urlretrieve


MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
)


def _ensure_heavy_model(model_path: Path) -> Path:
    """Pobierz oficjalny model Heavy, jeśli nie ma go jeszcze lokalnie."""
    if model_path.is_file() and model_path.stat().st_size > 0:
        return model_path

    model_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = model_path.with_suffix(".task.download")
    print("Pobieranie modelu MediaPipe Pose Landmarker Heavy...")

    try:
        urlretrieve(MODEL_URL, temporary_path)
        temporary_path.replace(model_path)
    except (OSError, URLError) as exc:
        temporary_path.unlink(missing_ok=True)
        raise RuntimeError(
            "Nie udało się pobrać modelu Pose Heavy. Pobierz go ręcznie z:\n"
            f"{MODEL_URL}\n"
            f"i zapisz jako: {model_path}"
        ) from exc

    return model_path


def _landmarks_to_array(landmarks: Any, np: Any) -> Any:
    result = np.full((33, 4), np.nan, dtype=np.float32)
    if not landmarks:
        return result

    for index, landmark in enumerate(landmarks[:33]):
        visibility = getattr(landmark, "visibility", None)
        result[index] = [
            landmark.x,
            landmark.y,
            landmark.z,
            np.nan if visibility is None else visibility,
        ]
    return result


def extract_pose(
    video_path: str | Path,
    output_path: str | Path,
    model_path: str | Path | None = None,
) -> Path:
    """Estymuj pozę jednej osoby i zapisz punkty wszystkich klatek do NPZ."""
    try:
        import cv2
        import mediapipe as mp
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "Brakuje OpenCV, MediaPipe lub NumPy. Uruchom: "
            "python -m pip install -r requirements.txt"
        ) from exc

    if not hasattr(mp, "tasks"):
        raise RuntimeError(
            f"MediaPipe {mp.__version__} nie zawiera API Tasks. Uruchom: "
            "python -m pip install --upgrade mediapipe"
        )

    video_path = Path(video_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    if model_path is None:
        project_root = Path(__file__).resolve().parents[1]
        model_path = project_root / "models" / "pose_landmarker_heavy.task"
    model_path = _ensure_heavy_model(Path(model_path).expanduser().resolve())

    if not video_path.is_file():
        raise FileNotFoundError(f"Nie znaleziono nagrania: {video_path}")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"Nie można otworzyć nagrania: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    declared_frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))

    pose_frames: list[Any] = []
    world_frames: list[Any] = []
    timestamps: list[float] = []
    detected: list[bool] = []

    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False,
    )

    try:
        with mp.tasks.vision.PoseLandmarker.create_from_options(options) as landmarker:
            frame_index = 0
            previous_timestamp_ms = -1

            while True:
                success, frame = capture.read()
                if not success:
                    break

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb_frame = np.ascontiguousarray(rgb_frame)
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb_frame,
                )

                if fps > 0:
                    timestamp_ms = int(round(frame_index * 1000.0 / fps))
                else:
                    timestamp_ms = frame_index * 33
                timestamp_ms = max(timestamp_ms, previous_timestamp_ms + 1)
                previous_timestamp_ms = timestamp_ms

                result = landmarker.detect_for_video(mp_image, timestamp_ms)
                has_pose = bool(result.pose_landmarks)

                image_landmarks = result.pose_landmarks[0] if has_pose else None
                world_landmarks = (
                    result.pose_world_landmarks[0]
                    if result.pose_world_landmarks
                    else None
                )
                pose_frames.append(_landmarks_to_array(image_landmarks, np))
                world_frames.append(_landmarks_to_array(world_landmarks, np))
                timestamps.append(timestamp_ms / 1000.0)
                detected.append(has_pose)

                frame_index += 1
                if frame_index % 100 == 0:
                    total = declared_frame_count if declared_frame_count > 0 else "?"
                    print(f"Przetworzono {frame_index}/{total} klatek")
    finally:
        capture.release()

    if not pose_frames:
        raise ValueError("Nie udało się odczytać żadnej klatki nagrania.")

    pose_array = np.stack(pose_frames)
    world_array = np.stack(world_frames)
    timestamp_array = np.asarray(timestamps, dtype=np.float64)
    detected_array = np.asarray(detected, dtype=bool)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        landmarks=pose_array,
        world_landmarks=world_array,
        timestamps=timestamp_array,
        detected=detected_array,
        fps=np.float32(fps),
        width=np.int32(width),
        height=np.int32(height),
        source_video=str(video_path),
        pose_model="pose_landmarker_heavy",
    )

    detection_percentage = float(detected_array.mean() * 100)
    print(f"Zapisano: {output_path}")
    print(f"Liczba klatek: {len(pose_array)}")
    print(f"Wykrycie sylwetki: {detection_percentage:.1f}%")
    return output_path
