# Dancer AI — MVP dla cha-chy i rumby

Prosty system wspierający analizę pojedynczego tancerza tańca towarzyskiego.
Pierwsza wersja obsługuje dwa tańce: **cha-cha** i **rumba**.

## Założenia nagrania

- jedna osoba w kadrze,
- widoczna cała sylwetka (łącznie ze stopami),
- nieruchomy telefon ustawiony poziomo lub pionowo,
- 1080p i 60 fps,
- dobre, równomierne oświetlenie,
- muzyka słyszalna na nagraniu,
- bez cięć montażowych.

## Plan przepływu danych

1. Walidacja i wstępne przetwarzanie wideo — OpenCV.
2. Estymacja 33 punktów ciała — MediaPipe Pose Heavy.
3. Uzupełnianie krótkich braków, filtr Butterwortha i normalizacja — SciPy.
4. Cechy biomechaniczne i rytmiczne — NumPy/SciPy.
5. Beat tracking ścieżki audio — librosa.
6. Ocena modelu Random Forest — scikit-learn.
7. Raport i wykresy — Matplotlib.

Ważne: Random Forest wymaga oznaczonych przykładów treningowych. Do czasu
zebrania danych system może wyliczać obiektywne metryki, ale nie powinien
udawać wiarygodnej oceny sędziowskiej.

## Instalacja

Zalecany jest 64-bitowy Python 3.11 na Windows.

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Etap 1 — kontrola nagrania

```powershell
python -m dancer_ai inspect-video "C:\sciezka\taniec.mp4"
```

Domyślnie raport zostanie zapisany jako plik JSON obok nagrania. Inną
lokalizację można podać opcją `--output`.

```powershell
python -m dancer_ai inspect-video "taniec.mp4" --output "raport.json"
```

Kod zakończenia `0` oznacza poprawny materiał, a `2` — materiał możliwy do
odczytania, ale niespełniający przynajmniej jednego wymagania.

