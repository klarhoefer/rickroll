# RickRoll

## Data prep

```bash
ffmpeg -i song.mp3 song.wav

ffmpeg -i song.wav -ac 1 -ar 16000 song16.wav
```

## ML-Pipeline (PyTorch)

Ordnet einzelne Sekunden einem von mehreren Songs zu.

1. Songs als mono/16 kHz-`.wav` (siehe oben) in den Ordner `songs/` legen (alle mit gleicher Samplerate).
2. Trainieren:
   ```bash
   uv run rickroll train --songs songs --epochs 15
   ```
   - Eingabe: 1 s Audio -> Log-Mel-Spektrogramm -> kleines CNN -> Klasse = Song.
   - Training auf zufälligen 1-s-Fenstern (beliebiger Offset, Gain/Rausch-Augmentierung).
   - Validierung auf zurückgehaltenen 10-s-Zeitblöcken (jeder 5. Block), damit sich Fenster nicht überlappen.
   - Bestes Modell landet in `model.pt`.
3. Zuordnen:
   ```bash
   uv run rickroll predict songs/song1.wav            # alle Sekunden
   uv run rickroll predict songs/song1.wav --second 42
   ```
