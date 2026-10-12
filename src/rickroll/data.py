import os
import random

import torch
from torch.utils.data import Dataset

from .chunks import WavFile

BLOCK_SECONDS = 10  # Zeitblöcke für den Train/Val-Split


def load_songs(folder: str) -> tuple[list[str], list[torch.Tensor], int]:
    """Lädt alle .wav-Dateien eines Ordners. Rückgabe: (Namen, Signale in -1..1, Samplerate)."""
    names, signals, rate = [], [], None
    for fn in sorted(os.listdir(folder)):
        if not fn.lower().endswith('.wav'):
            continue
        wf = WavFile(os.path.join(folder, fn))
        if rate is None:
            rate = wf.rate
        elif wf.rate != rate:
            raise ValueError(f"{fn}: Samplerate {wf.rate} != {rate} (alle Dateien gleich konvertieren)")
        # WavFile liefert 0..1 -> auf -1..1 zentrieren
        signals.append(torch.from_numpy(wf.daten).float() * 2 - 1)
        names.append(fn)
    if not names:
        raise FileNotFoundError(f"Keine .wav-Dateien in '{folder}'")
    return names, signals, rate


def split_seconds(n_seconds: int, val_every: int) -> tuple[list[int], list[int]]:
    """Teilt die ganzen Sekunden eines Songs in zusammenhängende Blöcke:
    jeder `val_every`-te Block wird zur Validierung zurückgehalten, damit sich
    Trainings- und Validierungsfenster nie überlappen."""
    train, val = [], []
    for s in range(n_seconds):
        is_val = (s // BLOCK_SECONDS) % val_every == val_every - 1
        (val if is_val else train).append(s)
    return train, val


class TrainSeconds(Dataset):
    """Zufällige 1-s-Fenster (beliebiger Offset) innerhalb der Trainingssekunden, mit Augmentierung."""

    def __init__(self, signals, rate, val_every=5, length=20000, augment=True):
        self.signals, self.rate, self.length, self.augment = signals, rate, length, augment
        self.pool = []  # (song, sekunde), die komplett im Trainingsbereich liegen
        for i, sig in enumerate(signals):
            train, _ = split_seconds(len(sig) // rate, val_every)
            self.pool += [(i, s) for s in train]
        if not self.pool:
            raise ValueError("Keine Trainingssekunden - Songs zu kurz?")

    def __len__(self):
        return self.length

    def __getitem__(self, _):
        song, sec = random.choice(self.pool)
        sig = self.signals[song]
        # Fenster darf in die Folgesekunde ragen, solange es im selben Block bleibt
        start = sec * self.rate + random.randrange(self.rate // 2)
        start = min(start, len(sig) - self.rate)
        x = sig[start:start + self.rate].clone()
        if self.augment:
            x = x * random.uniform(0.5, 1.5)
            x = x + torch.randn_like(x) * random.uniform(0, 0.01)
        return x, song


class ValSeconds(Dataset):
    """Alle ganzen Sekunden der zurückgehaltenen Blöcke (so wie WavFile.get_second sie liefert)."""

    def __init__(self, signals, rate, val_every=5):
        self.items = []
        for i, sig in enumerate(signals):
            _, val = split_seconds(len(sig) // rate, val_every)
            self.items += [(sig[s * rate:(s + 1) * rate], i) for s in val]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]
