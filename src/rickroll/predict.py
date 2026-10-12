import torch

from .chunks import WavFile
from .model import SecondClassifier


def load_model(path='model.pt'):
    ckpt = torch.load(path, map_location='cpu')
    model = SecondClassifier(len(ckpt['names']), ckpt['rate'])
    model.load_state_dict(ckpt['state'])
    return model.eval(), ckpt['names'], ckpt['rate']


def predict(wav_path: str, model_path='model.pt', second: int | None = None) -> None:
    """Ordnet einzelne Sekunden einer .wav-Datei einem Song zu (alle Sekunden, falls `second` fehlt)."""
    model, names, rate = load_model(model_path)
    wf = WavFile(wav_path)
    if wf.rate != rate:
        raise ValueError(f"Samplerate {wf.rate} != {rate} des Modells")
    seconds = [second] if second is not None else range(len(wf.daten) // rate)
    for s in seconds:
        x = torch.from_numpy(wf.get_second(s)).float() * 2 - 1
        if len(x) < rate:
            continue
        with torch.no_grad():
            p = model(x[None]).softmax(1)[0]
        conf, idx = p.max(0)
        print(f"Sekunde {s:4d}: {names[idx]}  ({conf:.1%})")
