from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from dancer_ai.audio_analysis import analyze_audio_and_synchronization
from dancer_ai.biomechanics import extract_biomechanical_features
from dancer_ai.modeling import add_labeled_sample, predict_score, train_random_forest
from dancer_ai.pipeline import analyze_video
from dancer_ai.pose import extract_pose
from dancer_ai.reporting import generate_figure_report
from dancer_ai.signal_processing import process_pose
from dancer_ai.video import inspect_video
from dancer_ai.visualization import create_pose_preview


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dancer-ai",
        description="Analiza pojedynczego tancerza cha-chy i rumby.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser(
        "inspect-video", help="Sprawdź parametry wejściowego nagrania."
    )
    inspect_parser.add_argument("video", type=Path, help="Ścieżka do pliku wideo.")
    inspect_parser.add_argument(
        "--output", type=Path, help="Ścieżka raportu JSON (domyślnie obok wideo)."
    )

    pose_parser = subparsers.add_parser(
        "extract-pose",
        help="Wyodrębnij punkty szkieletu przez MediaPipe Pose Heavy.",
    )
    pose_parser.add_argument("video", type=Path, help="Ścieżka do pliku wideo.")
    pose_parser.add_argument("--output", type=Path, help="Ścieżka pliku NPZ.")

    preview_parser = subparsers.add_parser(
        "visualize-pose", help="Nałóż zapisany szkielet na nagranie."
    )
    preview_parser.add_argument("video", type=Path, help="Ścieżka do wideo.")
    preview_parser.add_argument("pose", type=Path, help="Ścieżka do pliku NPZ.")
    preview_parser.add_argument("--output", type=Path, help="Wynikowy plik MP4.")
    preview_parser.add_argument(
        "--visibility-threshold",
        type=float,
        default=0.5,
        help="Minimalna widoczność punktu od 0 do 1 (domyślnie 0.5).",
    )

    process_parser = subparsers.add_parser(
        "process-pose",
        help="Interpoluj, filtruj i normalizuj dane szkieletowe.",
    )
    process_parser.add_argument("pose", type=Path, help="Wejściowy plik NPZ.")
    process_parser.add_argument("--output", type=Path, help="Wynikowy plik NPZ.")
    process_parser.add_argument("--cutoff", type=float, default=6.0)
    process_parser.add_argument("--filter-order", type=int, default=4)
    process_parser.add_argument("--min-visibility", type=float, default=0.3)
    process_parser.add_argument("--max-gap", type=float, default=0.25)

    features_parser = subparsers.add_parser(
        "extract-features",
        help="Wylicz cechy biomechaniczne z przetworzonego szkieletu.",
    )
    features_parser.add_argument(
        "pose", type=Path, help="Przetworzony plik NPZ."
    )
    features_parser.add_argument("--output", type=Path, help="Wynikowy plik NPZ.")

    audio_parser = subparsers.add_parser(
        "analyze-audio",
        help="Przeanalizuj tempo muzyki i synchronizację ruchu.",
    )
    audio_parser.add_argument("video", type=Path, help="Nagranie z dźwiękiem.")
    audio_parser.add_argument("features", type=Path, help="Plik cech ruchu NPZ.")
    audio_parser.add_argument(
        "--dance", required=True, choices=("cha-cha", "rumba")
    )
    audio_parser.add_argument("--output", type=Path, help="Wynikowy plik NPZ.")
    audio_parser.add_argument(
        "--tolerance",
        type=float,
        default=0.25,
        help="Tolerancja synchronizacji w sekundach (domyślnie 0.25).",
    )

    sample_parser = subparsers.add_parser(
        "add-sample", help="Dodaj ocenione nagranie do zbioru treningowego."
    )
    sample_parser.add_argument("features", type=Path)
    sample_parser.add_argument("audio", type=Path)
    sample_parser.add_argument(
        "--technical-score", type=float, required=True, help="Technika 1–10."
    )
    sample_parser.add_argument(
        "--rhythm-score", type=float, required=True, help="Ruch do muzyki 1–10."
    )
    sample_parser.add_argument("--id", required=True, dest="recording_id")
    sample_parser.add_argument(
        "--dataset", type=Path, default=Path("data/dance_scores.csv")
    )

    train_parser = subparsers.add_parser(
        "train-model", help="Wytrenuj model Random Forest."
    )
    train_parser.add_argument(
        "dataset", type=Path, nargs="?", default=Path("data/dance_scores.csv")
    )
    train_parser.add_argument(
        "--output", type=Path, default=Path("models/dance_score_rf.joblib")
    )

    predict_parser = subparsers.add_parser(
        "predict-score", help="Przewidź ocenę za pomocą Random Forest."
    )
    predict_parser.add_argument("features", type=Path)
    predict_parser.add_argument("audio", type=Path)
    predict_parser.add_argument(
        "--model", type=Path, default=Path("models/dance_score_rf.joblib")
    )
    predict_parser.add_argument("--output", type=Path)

    report_parser = subparsers.add_parser(
        "generate-report", help="Utwórz wykres i kolorowe wideo dla figury."
    )
    report_parser.add_argument("video", type=Path)
    report_parser.add_argument("pose", type=Path)
    report_parser.add_argument("features", type=Path)
    report_parser.add_argument("audio", type=Path)
    report_parser.add_argument("--dance", required=True, choices=("cha-cha", "rumba"))
    report_parser.add_argument("--figure", default="basic-step")
    report_parser.add_argument("--output-dir", type=Path)

    pipeline_parser = subparsers.add_parser(
        "analyze-video", help="Uruchom pełną analizę i utwórz raport HTML."
    )
    pipeline_parser.add_argument("video", type=Path)
    pipeline_parser.add_argument("--dance", required=True, choices=("cha-cha", "rumba"))
    pipeline_parser.add_argument("--figure", default="basic-step")
    pipeline_parser.add_argument("--model", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "inspect-video":
            inspection = inspect_video(args.video)
            output = args.output or args.video.with_suffix(".inspection.json")
            output = output.expanduser().resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(
                json.dumps(inspection.to_dict(), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

            metadata = inspection.metadata
            print(
                f"Wideo: {metadata.width}x{metadata.height}, "
                f"{metadata.fps:.3f} fps, {metadata.duration_seconds:.2f} s"
            )
            print(f"Raport: {output}")
            if inspection.accepted:
                print("Wynik: materiał spełnia podstawowe wymagania.")
                return 0

            print("Wynik: materiał wymaga uwagi:")
            for warning in inspection.warnings:
                print(f"- {warning}")
            return 2

        if args.command == "extract-pose":
            output = args.output
            if output is None:
                output = Path("data/processed") / f"{args.video.stem}_pose.npz"
            extract_pose(args.video, output)
            return 0

        if args.command == "visualize-pose":
            output = args.output
            if output is None:
                output = Path("reports") / f"{args.video.stem}_skeleton.mp4"
            create_pose_preview(
                args.video,
                args.pose,
                output,
                visibility_threshold=args.visibility_threshold,
            )
            return 0

        if args.command == "process-pose":
            output = args.output
            if output is None:
                output = Path("data/processed") / f"{args.pose.stem}_processed.npz"
            process_pose(
                args.pose,
                output,
                cutoff_hz=args.cutoff,
                filter_order=args.filter_order,
                visibility_threshold=args.min_visibility,
                max_gap_seconds=args.max_gap,
            )
            return 0

        if args.command == "extract-features":
            output = args.output
            if output is None:
                output = Path("data/processed") / f"{args.pose.stem}_features.npz"
            extract_biomechanical_features(args.pose, output)
            return 0

        if args.command == "analyze-audio":
            output = args.output
            if output is None:
                output = Path("data/processed") / f"{args.video.stem}_{args.dance}_audio.npz"
            analyze_audio_and_synchronization(
                args.video,
                args.features,
                args.dance,
                output,
                synchronization_tolerance_seconds=args.tolerance,
            )
            return 0

        if args.command == "add-sample":
            add_labeled_sample(
                args.features,
                args.audio,
                args.dataset,
                args.technical_score,
                args.rhythm_score,
                args.recording_id,
            )
            return 0

        if args.command == "train-model":
            train_random_forest(args.dataset, args.output)
            return 0

        if args.command == "predict-score":
            output = args.output or Path("reports") / f"{args.audio.stem}_score.json"
            predict_score(args.features, args.audio, args.model, output)
            return 0

        if args.command == "generate-report":
            output_directory = args.output_dir or Path("reports") / f"{args.video.stem}_{args.dance}_{args.figure}"
            generate_figure_report(
                args.video, args.pose, args.features, args.audio,
                output_directory, args.dance, args.figure,
            )
            return 0

        if args.command == "analyze-video":
            analyze_video(args.video, args.dance, args.figure, args.model)
            return 0
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Błąd: {exc}")
        return 1

    return 1
