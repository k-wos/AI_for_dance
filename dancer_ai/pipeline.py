from __future__ import annotations

import html
import json
import re
from pathlib import Path

from dancer_ai.audio_analysis import analyze_audio_and_synchronization
from dancer_ai.biomechanics import extract_biomechanical_features
from dancer_ai.modeling import predict_score
from dancer_ai.pose import extract_pose
from dancer_ai.reporting import generate_figure_report
from dancer_ai.signal_processing import process_pose
from dancer_ai.video import inspect_video


def _slug(value: str) -> str:
    result = re.sub(r"[^a-zA-Z0-9_-]+", "-", value.strip()).strip("-")
    return result or "figure"


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _percent(value: float | None) -> str:
    return "brak danych" if value is None else f"{value * 100:.1f}%"


def _milliseconds(value: float | None) -> str:
    return "brak danych" if value is None else f"{value * 1000:.0f} ms"


def _score_block(prediction: dict | None) -> str:
    if prediction is None:
        return """
        <section>
          <h2>Ocena modelu</h2>
          <p>Model Random Forest nie został jeszcze wytrenowany. Raport zawiera
          analizę pomiarową, ale nie przyznaje ocen 1–10.</p>
        </section>
        """

    scores = prediction["scores"]
    strengths = "".join(
        f"<li>{html.escape(item)}</li>" for item in prediction.get("strengths", [])
    ) or "<li>Brak wystarczających danych porównawczych.</li>"
    improvements = "".join(
        f"<li>{html.escape(item)}</li>"
        for item in prediction.get("areas_for_improvement", [])
    )
    return f"""
    <section>
      <h2>Ocena modelu</h2>
      <div class="scores">
        <div><span>Aspekty techniczne</span><strong>{scores['technical_aspects']:.1f}/10</strong></div>
        <div><span>Ruch w rytm muzyki</span><strong>{scores['movement_to_music']:.1f}/10</strong></div>
      </div>
      <h3>Mocne strony</h3><ul>{strengths}</ul>
      <h3>Elementy wymagające poprawy</h3><ul>{improvements}</ul>
    </section>
    """


def _write_html_report(
    output_path: Path,
    dance: str,
    figure: str,
    inspection: dict,
    audio: dict,
    figure_report: dict,
    prediction: dict | None,
) -> None:
    metadata = inspection["metadata"]
    ratios = figure_report["segment_color_ratio"]
    tempo_status = "tak" if audio["tempo_in_range"] else "nie"
    document = f"""<!doctype html>
<html lang="pl">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Raport Dancer AI — {html.escape(dance)}</title>
  <style>
    body {{ margin: 0; background: #111318; color: #edf0f4; font-family: Arial, sans-serif; }}
    main {{ max-width: 1050px; margin: auto; padding: 28px 18px 60px; }}
    h1, h2, h3 {{ font-weight: 500; }}
    section {{ background: #1b1e25; margin: 18px 0; padding: 20px; border-radius: 10px; }}
    .meta, .scores, .colors {{ display: grid; grid-template-columns: repeat(auto-fit,minmax(180px,1fr)); gap: 12px; }}
    .meta div, .scores div, .colors div {{ background: #242833; padding: 14px; border-radius: 8px; }}
    span {{ display: block; color: #b7bfcc; margin-bottom: 6px; }}
    strong {{ font-size: 1.55rem; }}
    img, video {{ display: block; width: 100%; height: auto; background: #000; border-radius: 8px; }}
    .green {{ border-left: 6px solid #42d15b; }} .yellow {{ border-left: 6px solid #ffd21f; }}
    .red {{ border-left: 6px solid #e83e3e; }} .gray {{ border-left: 6px solid #888; }}
    li {{ margin: 7px 0; line-height: 1.45; }}
    footer {{ color: #aab2bf; margin-top: 25px; }}
  </style>
</head>
<body><main>
  <h1>Raport analizy tancerza</h1>
  <section class="meta">
    <div><span>Taniec</span><strong>{html.escape(dance)}</strong></div>
    <div><span>Krok / figura</span><strong>{html.escape(figure)}</strong></div>
    <div><span>Nagranie</span><strong>{metadata['duration_seconds']:.1f} s</strong></div>
    <div><span>Parametry</span><strong>{metadata['width']}×{metadata['height']}, {metadata['fps']:.1f} fps</strong></div>
  </section>
  {_score_block(prediction)}
  <section>
    <h2>Muzyka i synchronizacja</h2>
    <div class="meta">
      <div><span>Tempo</span><strong>{audio['adjusted_tempo_bpm']:.1f} BPM</strong></div>
      <div><span>Tempo w zakresie</span><strong>{tempo_status}</strong></div>
      <div><span>Synchronizacja</span><strong>{_percent(audio['synchronization_ratio'])}</strong></div>
      <div><span>Średni błąd</span><strong>{_milliseconds(audio['mean_absolute_offset_seconds'])}</strong></div>
    </div>
  </section>
  <section>
    <h2>Współpraca bioder i nóg</h2>
    <img src="hip_leg_cooperation.png" alt="Wykres współpracy bioder i nóg">
  </section>
  <section>
    <h2>Szkielet i jakość odcinków</h2>
    <video controls preload="metadata" src="colored_skeleton.mp4"></video>
    <div class="colors">
      <div class="green"><span>Zielone</span><strong>{ratios['green'] * 100:.1f}%</strong></div>
      <div class="yellow"><span>Żółte</span><strong>{ratios['yellow'] * 100:.1f}%</strong></div>
      <div class="red"><span>Czerwone</span><strong>{ratios['red'] * 100:.1f}%</strong></div>
      <div class="gray"><span>Brak detekcji</span><strong>{ratios['gray'] * 100:.1f}%</strong></div>
    </div>
  </section>
  <footer>Kolorowanie jest obecnie pomiarowym MVP. Progi zostaną skalibrowane na ocenionych nagraniach danej figury.</footer>
</main></body></html>"""
    output_path.write_text(document, encoding="utf-8")


def analyze_video(
    video_path: str | Path,
    dance: str,
    figure: str = "basic-step",
    model_path: str | Path | None = None,
) -> Path:
    """Uruchom wszystkie etapy analizy i zbuduj pojedynczy raport HTML."""
    video_path = Path(video_path).expanduser().resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"Nie znaleziono nagrania: {video_path}")
    project_root = Path(__file__).resolve().parents[1]
    processed = project_root / "data" / "processed"
    report_directory = project_root / "reports" / f"{video_path.stem}_{dance}_{_slug(figure)}"
    processed.mkdir(parents=True, exist_ok=True)
    report_directory.mkdir(parents=True, exist_ok=True)

    pose_path = processed / f"{video_path.stem}_pose.npz"
    processed_pose_path = processed / f"{video_path.stem}_pose_processed.npz"
    features_path = processed / f"{video_path.stem}_pose_processed_features.npz"
    audio_path = processed / f"{video_path.stem}_{dance}_audio.npz"

    print("[1/6] Kontrola nagrania")
    inspection = inspect_video(video_path)
    inspection_path = report_directory / "video_inspection.json"
    inspection_path.write_text(
        json.dumps(inspection.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("[2/6] Estymacja pozy")
    extract_pose(video_path, pose_path)
    print("[3/6] Filtrowanie i normalizacja")
    process_pose(pose_path, processed_pose_path)
    print("[4/6] Cechy biomechaniczne")
    extract_biomechanical_features(processed_pose_path, features_path)
    print("[5/6] Muzyka i synchronizacja")
    analyze_audio_and_synchronization(video_path, features_path, dance, audio_path)
    print("[6/6] Wizualizacja figury")
    figure_report_path = generate_figure_report(
        video_path, processed_pose_path, features_path, audio_path,
        report_directory, dance, figure,
    )

    selected_model = Path(model_path).expanduser().resolve() if model_path else project_root / "models" / "dance_score_rf.joblib"
    prediction_path = report_directory / "prediction.json"
    prediction = None
    if selected_model.is_file():
        predict_score(features_path, audio_path, selected_model, prediction_path)
        prediction = _read_json(prediction_path)

    final_report = report_directory / "final_report.html"
    _write_html_report(
        final_report, dance, figure, inspection.to_dict(),
        _read_json(audio_path.with_suffix(".json")),
        _read_json(figure_report_path), prediction,
    )
    print(f"Gotowy raport: {final_report}")
    return final_report
