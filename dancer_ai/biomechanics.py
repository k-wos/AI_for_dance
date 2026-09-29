from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _angle_degrees(a: Any, b: Any, c: Any, np: Any) -> Any:
    """Kąt ABC w stopniach dla każdej klatki."""
    first = a - b
    second = c - b
    denominator = np.linalg.norm(first, axis=1) * np.linalg.norm(second, axis=1)
    cosine = np.divide(
        np.sum(first * second, axis=1),
        denominator,
        out=np.full(len(first), np.nan),
        where=denominator > 1e-9,
    )
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


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


def _derivative(values: Any, timestamps: Any, np: Any) -> Any:
    """Pochodna liczona oddzielnie dla każdego ciągłego fragmentu."""
    values = np.asarray(values, dtype=np.float64)
    result = np.full_like(values, np.nan, dtype=np.float64)
    finite = np.isfinite(values)
    if values.ndim > 1:
        finite = finite.all(axis=tuple(range(1, values.ndim)))

    for start, end in _finite_runs(finite):
        if end - start < 2:
            continue
        time_part = timestamps[start:end]
        if np.any(np.diff(time_part) <= 0):
            continue
        result[start:end] = np.gradient(
            values[start:end], time_part, axis=0, edge_order=1
        )
    return result


def _line_angle_degrees(left: Any, right: Any, np: Any) -> Any:
    difference = right - left
    angle = np.degrees(np.arctan2(-difference[:, 1], difference[:, 0]))
    # Linia barków/miednicy jest osią, więc 180° oznacza to samo co 0°.
    return (angle + 90.0) % 180.0 - 90.0


def _unwrap_degrees(values: Any, np: Any) -> Any:
    result = values.copy()
    for start, end in _finite_runs(np.isfinite(values)):
        result[start:end] = np.degrees(
            np.unwrap(np.radians(values[start:end]))
        )
    return result


def _hilbert_envelope(signal: Any, usable: Any, np: Any, hilbert: Any) -> Any:
    result = np.full(len(signal), np.nan, dtype=np.float64)
    mask = np.isfinite(signal) & usable
    for start, end in _finite_runs(mask):
        part = signal[start:end]
        if len(part) < 4:
            continue
        centered = part - np.median(part)
        result[start:end] = np.abs(hilbert(centered))
    return result


def _row_rms(values: Any, np: Any) -> Any:
    finite = np.isfinite(values)
    count = finite.sum(axis=1)
    mean_square = np.divide(
        np.nansum(values * values, axis=1),
        count,
        out=np.full(values.shape[0], np.nan),
        where=count > 0,
    )
    return np.sqrt(mean_square)


def _dominant_frequency(signal: Any, usable: Any, fps: float, np: Any) -> float:
    """Dominująca częstotliwość ruchu z najdłuższego poprawnego fragmentu."""
    runs = _finite_runs(np.isfinite(signal) & usable)
    if not runs:
        return float("nan")
    start, end = max(runs, key=lambda item: item[1] - item[0])
    part = signal[start:end]
    if len(part) < max(16, int(round(fps * 2))):
        return float("nan")

    centered = part - np.mean(part)
    spectrum = np.abs(np.fft.rfft(centered))
    frequencies = np.fft.rfftfreq(len(centered), d=1.0 / fps)
    allowed = (frequencies >= 0.2) & (frequencies <= min(8.0, fps * 0.45))
    if not allowed.any():
        return float("nan")
    allowed_indices = np.flatnonzero(allowed)
    return float(frequencies[allowed_indices[np.argmax(spectrum[allowed])]])


def extract_biomechanical_features(
    input_path: str | Path,
    output_path: str | Path,
) -> Path:
    """Wylicz cechy ruchu z przefiltrowanego i znormalizowanego szkieletu."""
    try:
        import numpy as np
        from scipy.signal import hilbert
    except ImportError as exc:
        raise RuntimeError(
            "Brakuje NumPy lub SciPy. Uruchom: "
            "python -m pip install -r requirements.txt"
        ) from exc

    input_path = Path(input_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Nie znaleziono danych: {input_path}")

    with np.load(input_path, allow_pickle=False) as data:
        image_filtered = data["landmarks_filtered"].astype(np.float64)
        image_normalized = data["landmarks_normalized"].astype(np.float64)
        world_normalized = data["world_landmarks_normalized"].astype(np.float64)
        timestamps = data["timestamps"].astype(np.float64)
        usable = data["usable"].astype(bool)
        fps = float(data["fps"])
        width = int(data["width"])
        height = int(data["height"])
        source_video = str(data["source_video"])

    if image_filtered.shape[1:] != (33, 4):
        raise ValueError(f"Niepoprawny kształt danych: {image_filtered.shape}")
    if len(timestamps) != len(image_filtered) or len(usable) != len(image_filtered):
        raise ValueError("Liczby klatek danych, czasu i maski użyteczności są różne.")

    image = image_normalized[:, :, :3]
    world = world_normalized[:, :, :3]

    left_knee = _angle_degrees(world[:, 23], world[:, 25], world[:, 27], np)
    right_knee = _angle_degrees(world[:, 24], world[:, 26], world[:, 28], np)
    left_hip = _angle_degrees(world[:, 11], world[:, 23], world[:, 25], np)
    right_hip = _angle_degrees(world[:, 12], world[:, 24], world[:, 26], np)

    shoulder_tilt = _line_angle_degrees(image[:, 11], image[:, 12], np)
    pelvis_tilt = _line_angle_degrees(image[:, 23], image[:, 24], np)
    shoulder_center = (image[:, 11] + image[:, 12]) / 2.0
    hip_center = (image[:, 23] + image[:, 24]) / 2.0
    torso_vector = shoulder_center - hip_center
    torso_lean = np.degrees(np.arctan2(torso_vector[:, 0], -torso_vector[:, 1]))

    pelvis_vector_world = world[:, 24] - world[:, 23]
    pelvis_yaw = _unwrap_degrees(
        np.degrees(
            np.arctan2(pelvis_vector_world[:, 2], pelvis_vector_world[:, 0])
        ),
        np,
    )
    pelvis_rotation_speed = np.abs(_derivative(pelvis_yaw, timestamps, np))

    left_ankle_velocity = _derivative(world[:, 27], timestamps, np)
    right_ankle_velocity = _derivative(world[:, 28], timestamps, np)
    left_wrist_velocity = _derivative(world[:, 15], timestamps, np)
    right_wrist_velocity = _derivative(world[:, 16], timestamps, np)
    left_ankle_speed = np.linalg.norm(left_ankle_velocity, axis=1)
    right_ankle_speed = np.linalg.norm(right_ankle_velocity, axis=1)
    left_wrist_speed = np.linalg.norm(left_wrist_velocity, axis=1)
    right_wrist_speed = np.linalg.norm(right_wrist_velocity, axis=1)

    raw_image = image_filtered[:, :, :3].copy()
    raw_image[:, :, 0] *= width
    raw_image[:, :, 1] *= height
    raw_image[:, :, 2] *= width
    raw_hip_center = (raw_image[:, 23] + raw_image[:, 24]) / 2.0
    raw_shoulder_center = (raw_image[:, 11] + raw_image[:, 12]) / 2.0
    torso_scale = np.linalg.norm(raw_shoulder_center - raw_hip_center, axis=1)
    median_torso_scale = np.nanmedian(torso_scale[usable])
    if not np.isfinite(median_torso_scale) or median_torso_scale < 1e-6:
        median_torso_scale = 1.0
    center_velocity = _derivative(
        raw_hip_center / median_torso_scale, timestamps, np
    )
    center_speed = np.linalg.norm(center_velocity, axis=1)

    ankle_midpoint = (image[:, 27] + image[:, 28]) / 2.0
    balance_offset = hip_center[:, 0] - ankle_midpoint[:, 0]
    step_width = np.linalg.norm(image[:, 27, :2] - image[:, 28, :2], axis=1)
    knee_asymmetry = np.abs(left_knee - right_knee)

    extremity_speeds = np.column_stack(
        [left_ankle_speed, right_ankle_speed, left_wrist_speed, right_wrist_speed]
    )
    motion_intensity = _row_rms(extremity_speeds, np)
    motion_envelope = _hilbert_envelope(
        motion_intensity, usable, np, hilbert
    )

    feature_names = np.asarray(
        [
            "left_knee_angle_deg",
            "right_knee_angle_deg",
            "knee_asymmetry_deg",
            "left_hip_angle_deg",
            "right_hip_angle_deg",
            "shoulder_tilt_deg",
            "pelvis_tilt_deg",
            "torso_lean_deg",
            "pelvis_yaw_deg",
            "pelvis_rotation_speed_deg_s",
            "left_ankle_speed_body_s",
            "right_ankle_speed_body_s",
            "left_wrist_speed_body_s",
            "right_wrist_speed_body_s",
            "center_speed_body_s",
            "balance_offset_body",
            "step_width_body",
            "motion_intensity",
            "motion_envelope",
        ]
    )
    features = np.column_stack(
        [
            left_knee, right_knee, knee_asymmetry, left_hip, right_hip,
            shoulder_tilt, pelvis_tilt, torso_lean, pelvis_yaw,
            pelvis_rotation_speed, left_ankle_speed, right_ankle_speed,
            left_wrist_speed, right_wrist_speed, center_speed,
            balance_offset, step_width, motion_intensity, motion_envelope,
        ]
    )
    features[~usable] = np.nan

    statistic_names = ("mean", "std", "p10", "p90")
    summary_values: list[float] = []
    summary_names: list[str] = []
    feature_statistics: dict[str, dict[str, float | None]] = {}
    for column_index, feature_name in enumerate(feature_names):
        valid_values = features[:, column_index]
        valid_values = valid_values[np.isfinite(valid_values)]
        if len(valid_values):
            values = (
                float(np.mean(valid_values)),
                float(np.std(valid_values)),
                float(np.percentile(valid_values, 10)),
                float(np.percentile(valid_values, 90)),
            )
        else:
            values = (float("nan"),) * 4

        feature_statistics[str(feature_name)] = {}
        for statistic_name, value in zip(statistic_names, values):
            summary_names.append(f"{feature_name}__{statistic_name}")
            summary_values.append(value)
            feature_statistics[str(feature_name)][statistic_name] = (
                value if np.isfinite(value) else None
            )

    dominant_frequency = _dominant_frequency(
        motion_intensity, usable, fps, np
    )
    summary_names.extend(["usable_frame_ratio", "dominant_motion_frequency_hz"])
    summary_values.extend([float(usable.mean()), dominant_frequency])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        features=features.astype(np.float32),
        feature_names=feature_names,
        summary=np.asarray(summary_values, dtype=np.float32),
        summary_names=np.asarray(summary_names),
        timestamps=timestamps,
        usable=usable,
        fps=np.float32(fps),
        source_video=source_video,
        source_pose=str(input_path),
    )

    report = {
        "source_video": source_video,
        "source_pose": str(input_path),
        "frame_count": int(len(features)),
        "usable_frame_ratio": float(usable.mean()),
        "dominant_motion_frequency_hz": (
            dominant_frequency if np.isfinite(dominant_frequency) else None
        ),
        "feature_statistics": feature_statistics,
    }
    report_path = output_path.with_suffix(".json")
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    print(f"Zapisano cechy: {output_path}")
    print(f"Zapisano raport: {report_path}")
    print(f"Liczba cech czasowych: {len(feature_names)}")
    print(f"Długość wektora dla modelu: {len(summary_values)}")
    return output_path
