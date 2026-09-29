from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any


DANCE_TEMPO_RANGES = {
    "cha-cha": (118.0, 132.0),
    "rumba": (96.0, 112.0),
}


def _adjust_tempo(raw_tempo: float, expected_range: tuple[float, float]) -> float:
    """Skoryguj typowy błąd half-time/double-time algorytmu beat tracking."""
    midpoint = sum(expected_range) / 2.0
    candidates = [raw_tempo * factor for factor in (0.5, 1.0, 2.0)]
    candidates = [tempo for tempo in candidates if 40.0 <= tempo <= 240.0]
    return min(candidates, key=lambda tempo: abs(tempo - midpoint))


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


def _movement_peaks(
    signal: Any,
    timestamps: Any,
    usable: Any,
    fps: float,
    np: Any,
    find_peaks: Any,
) -> Any:
    """Znajdź piki ruchu w najdłuższym poprawnym fragmencie nagrania."""
    runs = _finite_runs(np.isfinite(signal) & usable)
    if not runs:
        return np.asarray([], dtype=np.float64)

    start, end = max(runs, key=lambda item: item[1] - item[0])
    part = signal[start:end]
    if len(part) < 3:
        return np.asarray([], dtype=np.float64)

    prominence = max(float(np.std(part)) * 0.25, 1e-6)
    minimum_distance = max(1, int(round(fps * 0.18)))
    peak_indices, _ = find_peaks(
        part,
        distance=minimum_distance,
        prominence=prominence,
    )
    return timestamps[start + peak_indices]


def _beat_offsets(beat_times: Any, peak_times: Any, np: Any) -> Any:
    """Dla każdego beatu zwróć czas najbliższego piku ruchu minus czas beatu."""
    if len(beat_times) == 0 or len(peak_times) == 0:
        return np.asarray([], dtype=np.float64)

    offsets = np.empty(len(beat_times), dtype=np.float64)
    for index, beat_time in enumerate(beat_times):
        insertion = int(np.searchsorted(peak_times, beat_time))
        candidates = []
        if insertion < len(peak_times):
            candidates.append(peak_times[insertion])
        if insertion > 0:
            candidates.append(peak_times[insertion - 1])
        nearest = min(candidates, key=lambda value: abs(value - beat_time))
        offsets[index] = nearest - beat_time
    return offsets


def analyze_audio_and_synchronization(
    video_path: str | Path,
    features_path: str | Path,
    dance: str,
    output_path: str | Path,
    synchronization_tolerance_seconds: float = 0.25,
) -> Path:
    """Wykryj beaty i porównaj je z pikami intensywności ruchu."""
    try:
        import imageio_ffmpeg
        import librosa
        import numpy as np
        from scipy.signal import find_peaks
    except ImportError as exc:
        raise RuntimeError(
            "Brakuje bibliotek audio. Uruchom: "
            "python -m pip install -r requirements.txt"
        ) from exc

    if dance not in DANCE_TEMPO_RANGES:
        raise ValueError("Taniec musi mieć wartość 'cha-cha' albo 'rumba'.")
    if synchronization_tolerance_seconds <= 0:
        raise ValueError("Tolerancja synchronizacji musi być dodatnia.")

    video_path = Path(video_path).expanduser().resolve()
    features_path = Path(features_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"Nie znaleziono nagrania: {video_path}")
    if not features_path.is_file():
        raise FileNotFoundError(f"Nie znaleziono cech ruchu: {features_path}")

    with np.load(features_path, allow_pickle=False) as feature_data:
        features = feature_data["features"].astype(np.float64)
        feature_names = [str(name) for name in feature_data["feature_names"]]
        timestamps = feature_data["timestamps"].astype(np.float64)
        usable = feature_data["usable"].astype(bool)
        motion_fps = float(feature_data["fps"])

    try:
        motion_column = feature_names.index("motion_intensity")
    except ValueError as exc:
        raise ValueError("Plik cech nie zawiera kolumny motion_intensity.") from exc
    motion_signal = features[:, motion_column]

    sample_rate = 22050
    hop_length = 512
    with tempfile.TemporaryDirectory(prefix="dancer_ai_audio_") as temp_directory:
        wave_path = Path(temp_directory) / "audio.wav"
        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
        command = [
            ffmpeg_path,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(video_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            str(wave_path),
        ]
        try:
            subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            details = exc.stderr.strip() or "nieznany błąd FFmpeg"
            raise RuntimeError(f"Nie udało się wydzielić audio: {details}") from exc

        audio, sample_rate = librosa.load(
            wave_path,
            sr=sample_rate,
            mono=True,
        )

    if len(audio) == 0 or float(np.max(np.abs(audio))) < 1e-6:
        raise ValueError("Nagranie nie zawiera użytecznej ścieżki dźwiękowej.")

    onset_strength = librosa.onset.onset_strength(
        y=audio,
        sr=sample_rate,
        hop_length=hop_length,
    )
    raw_tempo_value, beat_frames = librosa.beat.beat_track(
        onset_envelope=onset_strength,
        sr=sample_rate,
        hop_length=hop_length,
    )
    raw_tempo_array = np.asarray(raw_tempo_value).reshape(-1)
    raw_tempo = float(raw_tempo_array[0]) if len(raw_tempo_array) else 0.0
    beat_times = librosa.frames_to_time(
        beat_frames,
        sr=sample_rate,
        hop_length=hop_length,
    ).astype(np.float64)
    onset_times = librosa.frames_to_time(
        np.arange(len(onset_strength)),
        sr=sample_rate,
        hop_length=hop_length,
    ).astype(np.float64)

    expected_range = DANCE_TEMPO_RANGES[dance]
    adjusted_tempo = _adjust_tempo(raw_tempo, expected_range) if raw_tempo > 0 else 0.0
    tempo_in_range = expected_range[0] <= adjusted_tempo <= expected_range[1]

    movement_peak_times = _movement_peaks(
        motion_signal,
        timestamps,
        usable,
        motion_fps,
        np,
        find_peaks,
    )
    if len(movement_peak_times):
        beats_in_motion = beat_times[
            (beat_times >= movement_peak_times[0])
            & (beat_times <= movement_peak_times[-1])
        ]
    else:
        beats_in_motion = np.asarray([], dtype=np.float64)

    beat_offsets = _beat_offsets(beats_in_motion, movement_peak_times, np)
    if len(beat_offsets):
        mean_absolute_offset = float(np.mean(np.abs(beat_offsets)))
        mean_signed_offset = float(np.mean(beat_offsets))
        synchronization_ratio = float(
            np.mean(np.abs(beat_offsets) <= synchronization_tolerance_seconds)
        )
    else:
        mean_absolute_offset = float("nan")
        mean_signed_offset = float("nan")
        synchronization_ratio = float("nan")

    rms = librosa.feature.rms(y=audio, hop_length=hop_length)[0]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        dance=dance,
        raw_tempo_bpm=np.float32(raw_tempo),
        adjusted_tempo_bpm=np.float32(adjusted_tempo),
        expected_tempo_min=np.float32(expected_range[0]),
        expected_tempo_max=np.float32(expected_range[1]),
        tempo_in_range=np.bool_(tempo_in_range),
        beat_times=beat_times,
        onset_times=onset_times,
        onset_strength=onset_strength.astype(np.float32),
        movement_peak_times=movement_peak_times,
        analyzed_beat_times=beats_in_motion,
        beat_offsets_seconds=beat_offsets,
        mean_absolute_offset_seconds=np.float32(mean_absolute_offset),
        mean_signed_offset_seconds=np.float32(mean_signed_offset),
        synchronization_ratio=np.float32(synchronization_ratio),
        audio_rms_mean=np.float32(np.mean(rms)),
        source_video=str(video_path),
        source_features=str(features_path),
    )

    def optional_number(value: float) -> float | None:
        return value if np.isfinite(value) else None

    report = {
        "dance": dance,
        "raw_tempo_bpm": raw_tempo,
        "adjusted_tempo_bpm": adjusted_tempo,
        "expected_tempo_bpm": list(expected_range),
        "tempo_in_range": bool(tempo_in_range),
        "detected_beats": int(len(beat_times)),
        "movement_peaks": int(len(movement_peak_times)),
        "analyzed_beats": int(len(beats_in_motion)),
        "synchronization_tolerance_seconds": synchronization_tolerance_seconds,
        "synchronization_ratio": optional_number(synchronization_ratio),
        "mean_absolute_offset_seconds": optional_number(mean_absolute_offset),
        "mean_signed_offset_seconds": optional_number(mean_signed_offset),
        "audio_rms_mean": float(np.mean(rms)),
        "source_video": str(video_path),
        "source_features": str(features_path),
    }
    report_path = output_path.with_suffix(".json")
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    print(f"Tempo surowe: {raw_tempo:.1f} BPM")
    print(f"Tempo dopasowane do {dance}: {adjusted_tempo:.1f} BPM")
    print(f"Tempo w oczekiwanym zakresie: {'tak' if tempo_in_range else 'nie'}")
    if np.isfinite(synchronization_ratio):
        print(f"Zgodność pików ruchu z beatem: {synchronization_ratio * 100:.1f}%")
        print(f"Średni błąd czasowy: {mean_absolute_offset * 1000:.0f} ms")
    else:
        print("Nie udało się wyliczyć synchronizacji ruchu z beatem.")
    print(f"Zapisano analizę audio: {output_path}")
    print(f"Zapisano raport: {report_path}")
    return output_path
