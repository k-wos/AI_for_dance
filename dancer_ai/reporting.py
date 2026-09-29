from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from dancer_ai.visualization import POSE_CONNECTIONS

LEFT_LEG = ((23, 25), (25, 27), (27, 29), (29, 31))
RIGHT_LEG = ((24, 26), (26, 28), (28, 30), (30, 32))
PELVIS = ((23, 24),)
COLORS = {
    "green": (50, 210, 50), "yellow": (0, 220, 255),
    "red": (30, 30, 230), "gray": (145, 145, 145),
}


def _feature_map(data: Any) -> dict[str, Any]:
    return {str(name): data["features"][:, i].astype(float)
            for i, name in enumerate(data["feature_names"])}


def _robust_anomaly(values: Any, usable: Any, np: Any) -> Any:
    result = np.full(len(values), np.nan)
    valid = np.isfinite(values) & usable
    if not valid.any():
        return result
    median = float(np.median(values[valid]))
    mad = float(np.median(np.abs(values[valid] - median)))
    scale = max(1.4826 * mad, float(np.std(values[valid])) * 0.25, 1e-6)
    result[valid] = np.minimum(np.abs(values[valid] - median) / scale, 4.0)
    return result


def _safe_ratio(values: Any, scale: float, usable: Any, np: Any) -> Any:
    result = np.full(len(values), np.nan)
    valid = np.isfinite(values) & usable
    result[valid] = np.minimum(np.abs(values[valid]) / scale, 4.0)
    return result


def _rolling_mean(values: Any, window: int, np: Any) -> Any:
    finite = np.isfinite(values)
    kernel = np.ones(max(1, window))
    total = np.convolve(np.where(finite, values, 0.0), kernel, mode="same")
    count = np.convolve(finite.astype(float), kernel, mode="same")
    return np.divide(total, count, out=np.full(len(values), np.nan), where=count > 0)


def _quality_arrays(features: dict[str, Any], usable: Any, fps: float, np: Any) -> tuple[Any, Any, Any]:
    left_speed = features["left_ankle_speed_body_s"]
    right_speed = features["right_ankle_speed_body_s"]
    hip_speed = features["pelvis_rotation_speed_deg_s"]
    left_jerk = _robust_anomaly(np.gradient(left_speed), usable, np)
    right_jerk = _robust_anomaly(np.gradient(right_speed), usable, np)
    hip_jerk = _robust_anomaly(np.gradient(hip_speed), usable, np)
    asymmetry = _safe_ratio(features["knee_asymmetry_deg"], 14.0, usable, np)
    balance = _safe_ratio(features["balance_offset_body"], 0.15, usable, np)
    torso = _safe_ratio(features["torso_lean_deg"], 8.0, usable, np)
    leg_activity = np.fmax(left_speed, right_speed)
    leg_valid = leg_activity[np.isfinite(leg_activity) & usable]
    hip_valid = hip_speed[np.isfinite(hip_speed) & usable]
    leg_scale = max(float(np.percentile(leg_valid, 90)), 1e-6) if len(leg_valid) else 1.0
    hip_scale = max(float(np.percentile(hip_valid, 90)), 1e-6) if len(hip_valid) else 1.0
    coordination = np.abs(leg_activity / leg_scale - hip_speed / hip_scale) * 2.0
    hip_error = .40 * hip_jerk + .30 * coordination + .20 * torso + .10 * balance
    left_error = .40 * left_jerk + .30 * asymmetry + .20 * balance + .10 * coordination
    right_error = .40 * right_jerk + .30 * asymmetry + .20 * balance + .10 * coordination
    window = max(1, int(round(fps * .20)))
    return tuple(_rolling_mean(x, window, np) for x in (hip_error, left_error, right_error))


def _stabilize_by_beats(errors: Any, timestamps: Any, beats: Any, np: Any) -> Any:
    result = errors.copy()
    if len(beats) < 2:
        return result
    bounds = np.concatenate(([timestamps[0]], beats, [timestamps[-1] + 1e-6]))
    for start, end in zip(bounds[:-1], bounds[1:]):
        mask = (timestamps >= start) & (timestamps < end) & np.isfinite(errors)
        if mask.any():
            result[mask] = np.mean(errors[mask])
    return result


def _status(error: float) -> str:
    if error != error:
        return "gray"
    return "green" if error < 1.0 else "yellow" if error < 2.0 else "red"


def _draw_chart(timestamps: Any, f: dict[str, Any], beats: Any, error: Any,
                dance: str, figure: str, output: Path, np: Any) -> None:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    fig.suptitle(f"Współpraca bioder i nóg — {dance}, {figure}")
    hip = f["pelvis_rotation_speed_deg_s"]
    legs = np.fmax(f["left_ankle_speed_body_s"], f["right_ankle_speed_body_s"])
    hip_scale = max(float(np.nanpercentile(hip, 90)), 1e-6)
    leg_scale = max(float(np.nanpercentile(legs, 90)), 1e-6)
    axes[0].plot(timestamps, hip / hip_scale, label="aktywność bioder")
    axes[0].plot(timestamps, legs / leg_scale, label="aktywność nóg")
    axes[0].set_ylabel("aktywność względna")
    axes[0].legend(loc="upper right")
    axes[1].plot(timestamps, 180 - f["left_knee_angle_deg"], label="lewe kolano")
    axes[1].plot(timestamps, 180 - f["right_knee_angle_deg"], label="prawe kolano")
    axes[1].set_ylabel("ugięcie kolana [°]")
    axes[1].legend(loc="upper right")
    axes[2].plot(timestamps, error, color="#444444")
    axes[2].axhspan(0, 1, color="#4CAF50", alpha=.18, label="poprawnie")
    axes[2].axhspan(1, 2, color="#FFC107", alpha=.18, label="do kontroli")
    axes[2].axhspan(2, 4, color="#F44336", alpha=.14, label="do poprawy")
    axes[2].set_ylim(0, 4)
    axes[2].set_ylabel("wskaźnik odchylenia")
    axes[2].set_xlabel("czas [s]")
    axes[2].legend(loc="upper right", ncol=3)
    for axis in axes:
        for beat in beats:
            axis.axvline(float(beat), color="#777777", alpha=.12, linewidth=.7)
        axis.grid(alpha=.18)
    fig.tight_layout(rect=(0, 0, 1, .96))
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _draw_connections(frame: Any, points: dict[int, tuple[int, int]],
                      connections: Any, color: Any, width: int, cv2: Any) -> None:
    for start, end in connections:
        if start in points and end in points:
            cv2.line(frame, points[start], points[end], color, width, cv2.LINE_AA)


def _create_video(video: Path, landmarks: Any, usable: Any, hip_error: Any,
                  left_error: Any, right_error: Any, dance: str, figure: str,
                  output: Path, cv2: Any, np: Any, imageio_ffmpeg: Any) -> None:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f"Nie można otworzyć nagrania: {video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    silent = output.with_name(output.stem + "_silent.mp4")
    writer = cv2.VideoWriter(str(silent), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise ValueError(f"Nie można utworzyć wideo: {silent}")
    index = 0
    try:
        while index < len(landmarks):
            ok, frame = capture.read()
            if not ok:
                break
            canvas = np.zeros_like(frame)
            current = landmarks[index]
            valid = np.isfinite(current[:, 0]) & np.isfinite(current[:, 1]) & (current[:, 3] >= .3)
            points = {int(i): (int(round(float(current[i, 0]) * width)),
                               int(round(float(current[i, 1]) * height)))
                      for i in np.flatnonzero(valid)}
            _draw_connections(canvas, points, POSE_CONNECTIONS, COLORS["gray"], 2, cv2)
            if usable[index]:
                hs, ls, rs = (_status(float(x[index])) for x in (hip_error, left_error, right_error))
            else:
                hs = ls = rs = "gray"
            _draw_connections(canvas, points, PELVIS, COLORS[hs], 6, cv2)
            _draw_connections(canvas, points, LEFT_LEG, COLORS[ls], 6, cv2)
            _draw_connections(canvas, points, RIGHT_LEG, COLORS[rs], 6, cv2)
            for point in points.values():
                cv2.circle(canvas, point, 4, (235, 235, 235), -1, cv2.LINE_AA)
            label = f"{dance} | {figure} | biodra:{hs} L:{ls} P:{rs}"
            cv2.putText(canvas, label, (18, 38), cv2.FONT_HERSHEY_SIMPLEX,
                        .72, (255, 255, 255), 2, cv2.LINE_AA)
            writer.write(canvas)
            index += 1
    finally:
        capture.release()
        writer.release()
    command = [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
               "-i", str(silent), "-i", str(video), "-map", "0:v:0", "-map", "1:a?",
               "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
               "-shortest", str(output)]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        silent.unlink(missing_ok=True)
    except subprocess.CalledProcessError:
        silent.replace(output)
        print("Uwaga: nie udało się zachować dźwięku w kolorowym nagraniu.")


def generate_figure_report(video_path: str | Path, processed_pose_path: str | Path,
                           features_path: str | Path, audio_path: str | Path,
                           output_directory: str | Path, dance: str, figure: str) -> Path:
    """Generuj wykres biodra–nogi i wideo z kolorową oceną odcinków."""
    try:
        import cv2
        import imageio_ffmpeg
        import matplotlib
        import numpy as np
        matplotlib.use("Agg")
    except ImportError as exc:
        raise RuntimeError("Brakuje bibliotek raportowania. Uruchom: python -m pip install -r requirements.txt") from exc
    paths = [Path(x).expanduser().resolve() for x in
             (video_path, processed_pose_path, features_path, audio_path)]
    video, pose, feature_file, audio = paths
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Nie znaleziono pliku: {path}")
    output = Path(output_directory).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    with np.load(pose, allow_pickle=False) as data:
        landmarks, usable, fps = data["landmarks_filtered"].copy(), data["usable"].astype(bool), float(data["fps"])
    with np.load(feature_file, allow_pickle=False) as data:
        timestamps, features = data["timestamps"].astype(float), _feature_map(data)
    with np.load(audio, allow_pickle=False) as data:
        beats = data["beat_times"].astype(float)
    hip, left, right = _quality_arrays(features, usable, fps, np)
    hip, left, right = (_stabilize_by_beats(x, timestamps, beats, np) for x in (hip, left, right))
    stacked = np.column_stack((hip, left, right))
    count = np.isfinite(stacked).sum(axis=1)
    combined = np.divide(np.nansum(stacked, axis=1), count,
                         out=np.full(len(stacked), np.nan), where=count > 0)
    chart = output / "hip_leg_cooperation.png"
    colored_video = output / "colored_skeleton.mp4"
    quality = output / "segment_quality.npz"
    _draw_chart(timestamps, features, beats, combined, dance, figure, chart, np)
    _create_video(video, landmarks, usable, hip, left, right, dance, figure,
                  colored_video, cv2, np, imageio_ffmpeg)
    np.savez_compressed(quality, timestamps=timestamps, hip_error=hip,
                        left_leg_error=left, right_leg_error=right,
                        usable=usable, beat_times=beats)
    statuses = [_status(float(v)) for v in np.concatenate((hip, left, right))]
    report = {
        "dance": dance, "figure": figure,
        "method": "provisional_biomechanical_rules_v1",
        "interpretation": {"green": "ruch w przyjętym zakresie",
                           "yellow": "fragment do kontroli",
                           "red": "fragment wymagający poprawy",
                           "gray": "niewystarczająca jakość detekcji"},
        "segment_color_ratio": {c: statuses.count(c) / len(statuses) for c in COLORS},
        "chart": str(chart), "video": str(colored_video), "quality_data": str(quality),
    }
    report_path = output / "figure_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Zapisano wykres: {chart}")
    print(f"Zapisano kolorowe wideo: {colored_video}")
    print(f"Zapisano raport figury: {report_path}")
    return report_path
