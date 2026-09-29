from __future__ import annotations

from pathlib import Path


POSE_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
    (11, 12), (11, 13), (13, 15), (15, 17), (17, 19),
    (15, 19), (15, 21), (12, 14), (14, 16), (16, 18),
    (18, 20), (16, 20), (16, 22), (11, 23), (12, 24),
    (23, 24), (23, 25), (25, 27), (27, 29), (29, 31),
    (27, 31), (24, 26), (26, 28), (28, 30), (30, 32),
    (28, 32),
)


def create_pose_preview(
    video_path: str | Path,
    pose_path: str | Path,
    output_path: str | Path,
    visibility_threshold: float = 0.5,
) -> Path:
    """Nałóż zapisane punkty szkieletu na oryginalne nagranie."""
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "Brakuje OpenCV lub NumPy. Uruchom: "
            "python -m pip install -r requirements.txt"
        ) from exc

    video_path = Path(video_path).expanduser().resolve()
    pose_path = Path(pose_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()

    if not video_path.is_file():
        raise FileNotFoundError(f"Nie znaleziono nagrania: {video_path}")
    if not pose_path.is_file():
        raise FileNotFoundError(f"Nie znaleziono danych pozy: {pose_path}")
    if not 0.0 <= visibility_threshold <= 1.0:
        raise ValueError("Próg widoczności musi należeć do zakresu 0–1.")

    with np.load(pose_path, allow_pickle=False) as pose_data:
        landmarks = pose_data["landmarks"].copy()
        detected = (
            pose_data["detected"].copy()
            if "detected" in pose_data.files
            else np.isfinite(landmarks[:, :, :2]).any(axis=(1, 2))
        )

    if landmarks.ndim != 3 or landmarks.shape[1:] != (33, 4):
        raise ValueError(
            f"Niepoprawny kształt danych pozy: {landmarks.shape}; "
            "oczekiwano [liczba_klatek, 33, 4]."
        )

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"Nie można otworzyć nagrania: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    fps = fps if fps > 0 else 30.0
    width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        writer.release()
        raise ValueError(f"Nie można utworzyć pliku wynikowego: {output_path}")

    frame_index = 0
    try:
        while frame_index < len(landmarks):
            success, frame = capture.read()
            if not success:
                break

            frame_landmarks = landmarks[frame_index]
            valid = (
                np.isfinite(frame_landmarks[:, 0])
                & np.isfinite(frame_landmarks[:, 1])
                & (frame_landmarks[:, 3] >= visibility_threshold)
            )

            if detected[frame_index] and valid.any():
                points: dict[int, tuple[int, int]] = {}
                for index in np.flatnonzero(valid):
                    x = int(round(float(frame_landmarks[index, 0]) * width))
                    y = int(round(float(frame_landmarks[index, 1]) * height))
                    points[int(index)] = (x, y)

                for start, end in POSE_CONNECTIONS:
                    if start in points and end in points:
                        cv2.line(
                            frame, points[start], points[end],
                            (50, 220, 50), 3, cv2.LINE_AA,
                        )

                for point in points.values():
                    cv2.circle(frame, point, 5, (0, 220, 255), -1, cv2.LINE_AA)

                status = f"POSE OK | widoczne punkty: {len(points)}/33"
                status_color = (50, 220, 50)
            else:
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (width, 65), (0, 0, 180), -1)
                cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)
                status = "BRAK DETEKCJI SYLWETKI"
                status_color = (255, 255, 255)

            cv2.putText(
                frame, status, (20, 42), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, status_color, 2, cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"Klatka: {frame_index + 1}/{len(landmarks)}",
                (20, height - 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (255, 255, 255), 2, cv2.LINE_AA,
            )

            writer.write(frame)
            frame_index += 1
            if frame_index % 200 == 0:
                print(f"Wizualizacja: {frame_index}/{len(landmarks)} klatek")
    finally:
        capture.release()
        writer.release()

    if frame_index == 0:
        output_path.unlink(missing_ok=True)
        raise ValueError("Nie udało się przetworzyć żadnej klatki.")
    if frame_index != len(landmarks):
        print(
            f"Uwaga: wideo miało mniej klatek niż dane pozy "
            f"({frame_index}/{len(landmarks)})."
        )

    print(f"Zapisano wizualizację: {output_path}")
    print("Uwaga: podgląd diagnostyczny nie zawiera ścieżki dźwiękowej.")
    return output_path
