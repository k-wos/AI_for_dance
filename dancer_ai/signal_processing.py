from __future__ import annotations

from pathlib import Path
from typing import Any


LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24


def _interpolate_short_gaps(values: Any, max_gap_frames: int, np: Any) -> Any:
    """Interpoluj tylko krótkie przerwy ograniczone poprawnymi próbkami."""
    result = values.copy()
    frame_count, landmark_count, coordinate_count = result.shape

    for landmark_index in range(landmark_count):
        for coordinate_index in range(coordinate_count):
            series = result[:, landmark_index, coordinate_index]
            missing = ~np.isfinite(series)
            start = None

            for frame_index in range(frame_count + 1):
                is_missing = frame_index < frame_count and missing[frame_index]
                if is_missing and start is None:
                    start = frame_index
                elif not is_missing and start is not None:
                    end = frame_index
                    gap_length = end - start
                    bounded = start > 0 and end < frame_count

                    if bounded and gap_length <= max_gap_frames:
                        left_value = series[start - 1]
                        right_value = series[end]
                        if np.isfinite(left_value) and np.isfinite(right_value):
                            series[start:end] = np.linspace(
                                left_value,
                                right_value,
                                gap_length + 2,
                                dtype=np.float64,
                            )[1:-1]
                    start = None

    return result


def _finite_runs(mask: Any) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start = None
    for index in range(len(mask) + 1):
        active = index < len(mask) and bool(mask[index])
        if active and start is None:
            start = index
        elif not active and start is not None:
            runs.append((start, index))
            start = None
    return runs


def _butterworth_filter(
    values: Any,
    fps: float,
    cutoff_hz: float,
    order: int,
    np: Any,
    butter: Any,
    sosfiltfilt: Any,
) -> Any:
    """Filtruj oddzielnie każdy ciągły fragment bez przechodzenia przez luki."""
    result = values.copy()
    effective_cutoff = min(cutoff_hz, fps * 0.45)
    if effective_cutoff <= 0:
        raise ValueError("Częstotliwość odcięcia filtra musi być dodatnia.")

    sos = butter(order, effective_cutoff, btype="low", fs=fps, output="sos")
    minimum_samples = max(12, order * 3 + 1)

    for landmark_index in range(result.shape[1]):
        for coordinate_index in range(result.shape[2]):
            series = result[:, landmark_index, coordinate_index]
            for start, end in _finite_runs(np.isfinite(series)):
                if end - start < minimum_samples:
                    continue
                try:
                    series[start:end] = sosfiltfilt(sos, series[start:end])
                except ValueError:
                    # Bardzo krótki fragment zostaje bez filtrowania.
                    continue
    return result


def _normalize_skeleton(values: Any, np: Any) -> Any:
    """Wycentruj szkielet na biodrach i podziel przez długość tułowia."""
    hip_center = (values[:, LEFT_HIP] + values[:, RIGHT_HIP]) / 2.0
    shoulder_center = (
        values[:, LEFT_SHOULDER] + values[:, RIGHT_SHOULDER]
    ) / 2.0
    torso_length = np.linalg.norm(shoulder_center - hip_center, axis=1)
    torso_length[torso_length < 1e-6] = np.nan
    return (values - hip_center[:, None, :]) / torso_length[:, None, None]


def process_pose(
    input_path: str | Path,
    output_path: str | Path,
    cutoff_hz: float = 6.0,
    filter_order: int = 4,
    visibility_threshold: float = 0.3,
    max_gap_seconds: float = 0.25,
) -> Path:
    """Interpoluj, filtruj i normalizuj dane szkieletowe z pliku NPZ."""
    try:
        import numpy as np
        from scipy.signal import butter, sosfiltfilt
    except ImportError as exc:
        raise RuntimeError(
            "Brakuje NumPy lub SciPy. Uruchom: "
            "python -m pip install -r requirements.txt"
        ) from exc

    input_path = Path(input_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Nie znaleziono danych pozy: {input_path}")
    if not 0.0 <= visibility_threshold <= 1.0:
        raise ValueError("Próg widoczności musi należeć do zakresu 0–1.")
    if max_gap_seconds < 0:
        raise ValueError("Maksymalna długość przerwy nie może być ujemna.")
    if filter_order < 1:
        raise ValueError("Rząd filtra musi być dodatni.")

    with np.load(input_path, allow_pickle=False) as data:
        landmarks = data["landmarks"].astype(np.float64)
        world_landmarks = data["world_landmarks"].astype(np.float64)
        timestamps = data["timestamps"].copy()
        detected = data["detected"].copy()
        fps = float(data["fps"])
        width = int(data["width"])
        height = int(data["height"])
        source_video = str(data["source_video"])
        pose_model = str(data["pose_model"]) if "pose_model" in data.files else "unknown"

    if landmarks.shape[1:] != (33, 4):
        raise ValueError(f"Niepoprawny kształt landmarks: {landmarks.shape}")
    if world_landmarks.shape != landmarks.shape:
        raise ValueError("Punkty obrazu i świata mają różne kształty.")
    if fps <= 0:
        raise ValueError("Nie można filtrować danych bez poprawnego FPS.")

    max_gap_frames = int(round(max_gap_seconds * fps))
    image_xyz = landmarks[:, :, :3].copy()
    world_xyz = world_landmarks[:, :, :3].copy()

    image_xyz[landmarks[:, :, 3] < visibility_threshold] = np.nan
    world_xyz[world_landmarks[:, :, 3] < visibility_threshold] = np.nan

    if max_gap_frames > 0:
        image_xyz = _interpolate_short_gaps(image_xyz, max_gap_frames, np)
        world_xyz = _interpolate_short_gaps(world_xyz, max_gap_frames, np)

    image_filtered_xyz = _butterworth_filter(
        image_xyz, fps, cutoff_hz, filter_order, np, butter, sosfiltfilt
    )
    world_filtered_xyz = _butterworth_filter(
        world_xyz, fps, cutoff_hz, filter_order, np, butter, sosfiltfilt
    )

    # Współrzędne obrazu zamieniamy na proporcjonalną przestrzeń pikselową,
    # aby pion i poziom miały tę samą skalę przed normalizacją.
    image_metric = image_filtered_xyz.copy()
    image_metric[:, :, 0] *= width
    image_metric[:, :, 1] *= height
    image_metric[:, :, 2] *= width

    image_normalized_xyz = _normalize_skeleton(image_metric, np)
    world_normalized_xyz = _normalize_skeleton(world_filtered_xyz, np)

    image_filtered = np.concatenate(
        [image_filtered_xyz, landmarks[:, :, 3:4]], axis=2
    ).astype(np.float32)
    world_filtered = np.concatenate(
        [world_filtered_xyz, world_landmarks[:, :, 3:4]], axis=2
    ).astype(np.float32)
    image_normalized = np.concatenate(
        [image_normalized_xyz, landmarks[:, :, 3:4]], axis=2
    ).astype(np.float32)
    world_normalized = np.concatenate(
        [world_normalized_xyz, world_landmarks[:, :, 3:4]], axis=2
    ).astype(np.float32)

    usable = np.isfinite(world_normalized_xyz[:, :, 0]).sum(axis=1) >= 20
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        landmarks_filtered=image_filtered,
        world_landmarks_filtered=world_filtered,
        landmarks_normalized=image_normalized,
        world_landmarks_normalized=world_normalized,
        timestamps=timestamps,
        detected=detected,
        usable=usable,
        fps=np.float32(fps),
        width=np.int32(width),
        height=np.int32(height),
        source_video=source_video,
        source_pose=str(input_path),
        pose_model=pose_model,
        cutoff_hz=np.float32(min(cutoff_hz, fps * 0.45)),
        filter_order=np.int32(filter_order),
        visibility_threshold=np.float32(visibility_threshold),
        max_gap_frames=np.int32(max_gap_frames),
    )

    print(f"Zapisano przetworzone dane: {output_path}")
    print(f"Krótka przerwa: maksymalnie {max_gap_frames} klatek")
    print(f"Klatki użyteczne: {usable.mean() * 100:.1f}%")
    return output_path
