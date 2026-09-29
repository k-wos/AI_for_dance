from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TARGET_COLUMNS = ("technical_score", "rhythm_score")
META_COLUMNS = ("recording_id", "dance", *TARGET_COLUMNS)
AUDIO_FEATURE_KEYS = (
    "raw_tempo_bpm",
    "adjusted_tempo_bpm",
    "tempo_in_range",
    "synchronization_ratio",
    "mean_absolute_offset_seconds",
    "mean_signed_offset_seconds",
    "audio_rms_mean",
)


def _scalar(value: Any, np: Any) -> float:
    array = np.asarray(value).reshape(-1)
    return float(array[0]) if len(array) else float("nan")


def load_sample_vector(
    features_path: str | Path,
    audio_path: str | Path,
) -> tuple[list[str], Any, str]:
    """Połącz zagregowane cechy biomechaniczne i audio w jeden wektor."""
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("Brakuje NumPy.") from exc

    features_path = Path(features_path).expanduser().resolve()
    audio_path = Path(audio_path).expanduser().resolve()
    if not features_path.is_file():
        raise FileNotFoundError(f"Nie znaleziono cech ruchu: {features_path}")
    if not audio_path.is_file():
        raise FileNotFoundError(f"Nie znaleziono analizy audio: {audio_path}")

    with np.load(features_path, allow_pickle=False) as data:
        names = [f"motion__{name}" for name in data["summary_names"]]
        values = data["summary"].astype(np.float64).tolist()

    with np.load(audio_path, allow_pickle=False) as data:
        dance = str(data["dance"])
        for key in AUDIO_FEATURE_KEYS:
            names.append(f"audio__{key}")
            values.append(_scalar(data[key], np))

    names.extend(("dance__cha_cha", "dance__rumba"))
    values.extend(
        (1.0 if dance == "cha-cha" else 0.0, 1.0 if dance == "rumba" else 0.0)
    )
    return names, np.asarray(values, dtype=np.float64), dance


def add_labeled_sample(
    features_path: str | Path,
    audio_path: str | Path,
    dataset_path: str | Path,
    technical_score: float,
    rhythm_score: float,
    recording_id: str,
) -> Path:
    """Dodaj nagranie ocenione w dwóch kategoriach do zbioru CSV."""
    for label, score in (
        ("aspekty techniczne", technical_score),
        ("ruch w rytm muzyki", rhythm_score),
    ):
        if not 1.0 <= score <= 10.0:
            raise ValueError(f"Ocena '{label}' musi należeć do zakresu 1–10.")
    recording_id = recording_id.strip()
    if not recording_id:
        raise ValueError("Identyfikator nagrania nie może być pusty.")

    names, values, dance = load_sample_vector(features_path, audio_path)
    dataset_path = Path(dataset_path).expanduser().resolve()
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(META_COLUMNS) + names

    existing_rows: list[dict[str, str]] = []
    if dataset_path.is_file():
        with dataset_path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != fieldnames:
                raise ValueError(
                    "Schemat istniejącego zbioru różni się od bieżącego. "
                    "Utwórz nowy zbiór dla dwukategorowej oceny 1–10."
                )
            existing_rows = list(reader)
        if any(row["recording_id"] == recording_id for row in existing_rows):
            raise ValueError(f"Nagranie '{recording_id}' jest już w zbiorze.")

    row: dict[str, Any] = {
        "recording_id": recording_id,
        "dance": dance,
        "technical_score": float(technical_score),
        "rhythm_score": float(rhythm_score),
    }
    row.update({name: float(value) for name, value in zip(names, values)})

    temporary_path = dataset_path.with_suffix(dataset_path.suffix + ".tmp")
    try:
        with temporary_path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(existing_rows)
            writer.writerow(row)
        temporary_path.replace(dataset_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    print(
        f"Dodano '{recording_id}' ({dance}): technika {technical_score:.1f}/10, "
        f"rytm {rhythm_score:.1f}/10"
    )
    print(f"Zbiór danych: {dataset_path}")
    print(f"Liczba nagrań: {len(existing_rows) + 1}")
    return dataset_path


def _read_dataset(dataset_path: Path, np: Any) -> tuple[Any, Any, list[str], list[str]]:
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Nie znaleziono zbioru: {dataset_path}")
    with dataset_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fieldnames = reader.fieldnames or []
        if not all(column in fieldnames for column in META_COLUMNS):
            raise ValueError("Zbiór nie zawiera wymaganych kolumn metadanych.")
        feature_names = [name for name in fieldnames if name not in META_COLUMNS]
        rows = list(reader)
    if not rows:
        raise ValueError("Zbiór danych jest pusty.")

    x = np.asarray(
        [[float(row[name]) for name in feature_names] for row in rows],
        dtype=np.float64,
    )
    y = np.asarray(
        [[float(row[target]) for target in TARGET_COLUMNS] for row in rows],
        dtype=np.float64,
    )
    return x, y, feature_names, [row["recording_id"] for row in rows]


def _average_ranks(values: Any, np: Any) -> Any:
    """Wyznacz rangi średnie, również dla powtarzających się ocen."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = (start + 1 + stop) / 2.0
        start = stop
    return ranks


def _icc_absolute_agreement(reference: Any, predicted: Any, np: Any) -> float:
    """ICC(A,1): pojedynczy pomiar i zgodność bezwzględna."""
    ratings = np.column_stack((reference, predicted))
    sample_count, rater_count = ratings.shape
    grand_mean = float(np.mean(ratings))
    row_means = np.mean(ratings, axis=1)
    column_means = np.mean(ratings, axis=0)
    ms_rows = rater_count * np.sum((row_means - grand_mean) ** 2) / (sample_count - 1)
    ms_columns = sample_count * np.sum((column_means - grand_mean) ** 2) / (rater_count - 1)
    residual = ratings - row_means[:, None] - column_means[None, :] + grand_mean
    ms_error = np.sum(residual**2) / ((sample_count - 1) * (rater_count - 1))
    denominator = (
        ms_rows
        + (rater_count - 1) * ms_error
        + rater_count * (ms_columns - ms_error) / sample_count
    )
    return float((ms_rows - ms_error) / denominator) if denominator else float("nan")


def _validation_statistics(reference: Any, predicted: Any, np: Any) -> dict[str, float]:
    """Oblicz metryki zgodności dla predykcji pozapróbkowych."""
    differences = predicted - reference
    difference_sd = float(np.std(differences, ddof=1))
    bias = float(np.mean(differences))
    reference_ranks = _average_ranks(reference, np)
    prediction_ranks = _average_ranks(predicted, np)
    return {
        "mae": float(np.mean(np.abs(differences))),
        "rmse": float(np.sqrt(np.mean(differences**2))),
        "spearman": float(np.corrcoef(reference_ranks, prediction_ranks)[0, 1]),
        "icc_a1": _icc_absolute_agreement(reference, predicted, np),
        "bias_model_minus_trainer": bias,
        "loa_lower": float(bias - 1.96 * difference_sd),
        "loa_upper": float(bias + 1.96 * difference_sd),
    }


def train_random_forest(dataset_path: str | Path, model_path: str | Path) -> Path:
    """Wytrenuj Random Forest dla techniki i rytmu w skali 1–10."""
    try:
        import joblib
        import numpy as np
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.impute import SimpleImputer
        from sklearn.model_selection import KFold, cross_val_predict
        from sklearn.pipeline import Pipeline
    except ImportError as exc:
        raise RuntimeError(
            "Brakuje scikit-learn lub joblib. Uruchom: "
            "python -m pip install -r requirements.txt"
        ) from exc

    dataset_path = Path(dataset_path).expanduser().resolve()
    model_path = Path(model_path).expanduser().resolve()
    x, y, feature_names, recording_ids = _read_dataset(dataset_path, np)
    if len(y) < 2:
        raise ValueError("Do treningu potrzebne są co najmniej 2 ocenione nagrania.")
    for index, target in enumerate(TARGET_COLUMNS):
        if len(np.unique(y[:, index])) < 2:
            raise ValueError(f"Kategoria '{target}' wymaga co najmniej 2 różnych ocen.")

    def make_pipeline() -> Any:
        return Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median",
                        add_indicator=True,
                        keep_empty_features=True,
                    ),
                ),
                (
                    "forest",
                    RandomForestRegressor(
                        n_estimators=400,
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        )

    validation_mae = None
    validation_metrics = None
    out_of_fold_predictions = None
    if len(y) >= 5:
        splitter = KFold(n_splits=min(5, len(y)), shuffle=True, random_state=42)
        predictions = cross_val_predict(make_pipeline(), x, y, cv=splitter, n_jobs=-1)
        validation_mae = {
            "technical": float(np.mean(np.abs(y[:, 0] - predictions[:, 0]))),
            "rhythm": float(np.mean(np.abs(y[:, 1] - predictions[:, 1]))),
        }
        validation_metrics = {
            "technical": _validation_statistics(y[:, 0], predictions[:, 0], np),
            "rhythm": _validation_statistics(y[:, 1], predictions[:, 1], np),
        }
        out_of_fold_predictions = [
            {
                "recording_id": recording_id,
                "technical_trainer": float(expected[0]),
                "technical_model": float(predicted[0]),
                "rhythm_trainer": float(expected[1]),
                "rhythm_model": float(predicted[1]),
            }
            for recording_id, expected, predicted in zip(recording_ids, y, predictions)
        ]

    pipeline = make_pipeline()
    pipeline.fit(x, y)
    artifact = {
        "pipeline": pipeline,
        "feature_names": feature_names,
        "targets": list(TARGET_COLUMNS),
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_samples": len(y),
        "recording_ids": recording_ids,
        "technical_score_range": [float(np.min(y[:, 0])), float(np.max(y[:, 0]))],
        "rhythm_score_range": [float(np.min(y[:, 1])), float(np.max(y[:, 1]))],
        "validation_mae": validation_mae,
        "validation_metrics": validation_metrics,
        "out_of_fold_predictions": out_of_fold_predictions,
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_path)

    report_path = model_path.with_suffix(".json")
    report_path.write_text(
        json.dumps(
            {key: value for key, value in artifact.items() if key != "pipeline"},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wytrenowano model na {len(y)} nagraniach.")
    if validation_mae:
        print(f"MAE technika: {validation_mae['technical']:.2f}")
        print(f"MAE rytm: {validation_mae['rhythm']:.2f}")
        for target, metrics in validation_metrics.items():
            print(
                f"{target}: RMSE={metrics['rmse']:.3f}, "
                f"Spearman={metrics['spearman']:.3f}, ICC(A,1)={metrics['icc_a1']:.3f}, "
                f"bias={metrics['bias_model_minus_trainer']:.3f}, "
                f"LOA=[{metrics['loa_lower']:.3f}, {metrics['loa_upper']:.3f}]"
            )
    else:
        print("Za mało danych na walidację; potrzeba co najmniej 5 nagrań.")
    print(f"Zapisano model: {model_path}")
    return model_path


def _value_map(features_path: str | Path, audio_path: str | Path, np: Any) -> dict[str, float]:
    values: dict[str, float] = {}
    with np.load(features_path, allow_pickle=False) as data:
        values.update(
            {str(name): float(value) for name, value in zip(data["summary_names"], data["summary"])}
        )
    with np.load(audio_path, allow_pickle=False) as data:
        for key in AUDIO_FEATURE_KEYS:
            values[key] = _scalar(data[key], np)
    return values


def _development_opinion(
    values: dict[str, float], technical_score: float, rhythm_score: float
) -> dict[str, list[str] | str]:
    strengths: list[str] = []
    improvements: list[str] = []

    knee_asymmetry = values.get("knee_asymmetry_deg__mean", float("nan"))
    torso_stability = values.get("torso_lean_deg__std", float("nan"))
    balance_variation = values.get("balance_offset_body__std", float("nan"))
    sync = values.get("synchronization_ratio", float("nan"))
    offset = values.get("mean_signed_offset_seconds", float("nan"))
    absolute_offset = values.get("mean_absolute_offset_seconds", float("nan"))
    tempo_ok = values.get("tempo_in_range", 0.0) >= 0.5
    usable = values.get("usable_frame_ratio", 0.0)

    if technical_score >= 7.0:
        strengths.append("Ogólny poziom techniczny jest stabilny w odniesieniu do danych treningowych.")
    if knee_asymmetry == knee_asymmetry and knee_asymmetry <= 10.0:
        strengths.append("Praca kolan jest względnie symetryczna.")
    elif knee_asymmetry == knee_asymmetry and knee_asymmetry > 14.0:
        improvements.append(
            "Wyrównaj pracę obu nóg: ćwicz powolne przenoszenie ciężaru przed lustrem, "
            "kontrolując podobny zakres ugięcia kolan."
        )

    if torso_stability == torso_stability and torso_stability <= 5.0:
        strengths.append("Tułów pozostaje stosunkowo stabilny podczas ruchu.")
    elif torso_stability == torso_stability and torso_stability > 8.0:
        improvements.append(
            "Popracuj nad stabilizacją tułowia: wykonuj podstawowy krok wolniej, "
            "utrzymując żebra nad miednicą."
        )

    if balance_variation == balance_variation and balance_variation > 0.18:
        improvements.append(
            "Zmienność balansu jest podwyższona; zatrzymuj pozycję na każdej nodze "
            "i kontroluj ustawienie środka ciężkości nad stopą podporową."
        )

    if sync == sync and sync >= 0.85:
        strengths.append("Większość akcentów ruchowych jest dobrze zsynchronizowana z beatem.")
    elif sync == sync and sync < 0.70:
        improvements.append(
            "Synchronizacja wymaga pracy: ćwicz sam transfer ciężaru do metronomu, "
            "najpierw w wolniejszym tempie."
        )

    if absolute_offset == absolute_offset and absolute_offset > 0.18:
        direction = "po" if offset > 0 else "przed"
        improvements.append(
            f"Akcent ruchu pojawia się przeciętnie {direction} beatem; ćwicz zakończenie "
            "transferu ciężaru dokładnie na uderzeniu muzyki."
        )
    if not tempo_ok:
        improvements.append(
            "Tempo nagrania jest poza przyjętym zakresem tego tańca; do porównań "
            "rozwojowych używaj utworów o zbliżonym, standardowym tempie."
        )
    if usable < 0.90:
        improvements.append(
            "Część nagrania nie zawierała pewnej detekcji całej sylwetki; ustaw kamerę tak, "
            "aby stopy i głowa pozostawały w kadrze do końca."
        )
    if not improvements:
        improvements.append(
            "Utrzymuj obecną jakość i zwiększaj trudność stopniowo, zachowując kontrolę rytmu."
        )

    opinion = "Mocne strony: " + " ".join(strengths or ["Wymagane są kolejne nagrania porównawcze."])
    opinion += " Elementy do poprawy: " + " ".join(improvements)
    return {"strengths": strengths, "areas_for_improvement": improvements, "opinion": opinion}


def predict_score(
    features_path: str | Path,
    audio_path: str | Path,
    model_path: str | Path,
    output_path: str | Path,
) -> Path:
    """Przewidź dwie oceny 1–10 i wygeneruj opinię rozwojową."""
    try:
        import joblib
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("Brakuje NumPy lub joblib.") from exc

    model_path = Path(model_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"Nie znaleziono modelu: {model_path}")

    artifact = joblib.load(model_path)
    names, sample, dance = load_sample_vector(features_path, audio_path)
    if names != artifact["feature_names"]:
        raise ValueError("Cechy nagrania nie pasują do schematu modelu.")

    pipeline = artifact["pipeline"]
    x = sample.reshape(1, -1)
    prediction = np.clip(pipeline.predict(x)[0], 1.0, 10.0)
    technical_score, rhythm_score = map(float, prediction)

    transformed = pipeline.named_steps["imputer"].transform(x)
    tree_predictions = np.asarray(
        [tree.predict(transformed)[0] for tree in pipeline.named_steps["forest"].estimators_]
    )
    uncertainty = np.std(tree_predictions, axis=0)
    opinion = _development_opinion(
        _value_map(features_path, audio_path, np), technical_score, rhythm_score
    )

    report = {
        "dance": dance,
        "scores": {
            "technical_aspects": technical_score,
            "movement_to_music": rhythm_score,
        },
        "uncertainty": {
            "technical_aspects_std": float(uncertainty[0]),
            "movement_to_music_std": float(uncertainty[1]),
        },
        **opinion,
        "training_samples": int(artifact["training_samples"]),
        "validation_mae": artifact["validation_mae"],
        "features_path": str(Path(features_path).expanduser().resolve()),
        "audio_path": str(Path(audio_path).expanduser().resolve()),
        "model_path": str(model_path),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Aspekty techniczne: {technical_score:.1f}/10")
    print(f"Ruch w rytm muzyki: {rhythm_score:.1f}/10")
    print(opinion["opinion"])
    print(f"Zapisano wynik: {output_path}")
    return output_path
